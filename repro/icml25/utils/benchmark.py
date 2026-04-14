from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as models

from .path_magnitude import get_path_magnitude_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the ICML 2025 pruning criteria runtime."
    )
    parser.add_argument("--saving-path", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    return parser.parse_args()


def benchmark_ms(fn, *, repeats: int, device: torch.device) -> float:
    for _ in range(3):
        result = fn()
        del result
        release_memory(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    timings = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
        else:
            begin = time.perf_counter()
            result = fn()
            timings.append((time.perf_counter() - begin) * 1000.0)
        del result
        release_memory(device)
    return float(statistics.median(timings))


def release_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def magnitude_scan(model: torch.nn.Module) -> list[torch.Tensor]:
    payload = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            payload.append(module.weight.detach().abs())
    return payload


def obd_named_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Match the historical invariant_pruning OBD parameter selection.

    The original code selected parameters by name using the tokens
    `conv` or `linear`, rather than all generic Linear modules.
    Keeping that convention makes the runtime table comparable to the
    original paper implementation and avoids pulling AlexNet/VGG
    classifier layers into the OBD benchmark.

    Torchvision AlexNet is a corner case because its convolutional
    weights live under `features.*` and its classifier uses
    `classifier.*`, so the historical token rule would return nothing.
    In that case we fall back to convolutional modules only, which is
    still consistent with the spirit of the historical implementation.
    """
    named = {
        name: param
        for name, param in model.named_parameters()
        if name.endswith(".weight")
        and ("conv" in name or "linear" in name)
    }
    if named:
        return named

    fallback = {}
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            weight_name = f"{module_name}.weight" if module_name else "weight"
            fallback[weight_name] = module.weight
    return fallback


def synthetic_obd_scores(
    model: torch.nn.Module,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    targets = torch.randint(0, 1000, (batch_size,), device=device)
    params_to_prune = obd_named_parameters(model)

    if not params_to_prune:
        raise RuntimeError(
            "No historically selected OBD weights were found for the synthetic OBD benchmark."
        )

    def loss_fn(params):
        logits = torch.func.functional_call(model, params, (images,))
        return F.cross_entropy(logits, targets)

    grad_fn = torch.func.grad(loss_fn)
    v = {
        name: torch.randint(
            0,
            2,
            param.shape,
            device=device,
            dtype=param.dtype,
        )
        * 2
        - 1
        for name, param in params_to_prune.items()
    }
    hvp = torch.func.jvp(grad_fn, (params_to_prune,), (v,))[1]
    return {
        name: 0.5 * hvp[name] * v[name] * (param.detach() ** 2)
        for name, param in params_to_prune.items()
    }


def benchmark_network(
    model_name: str,
    build_model,
    *,
    repeats: int,
    device: torch.device,
) -> dict[str, object]:
    obd_model = build_model()
    obd_param_count = int(
        sum(param.numel() for param in obd_named_parameters(obd_model).values())
    )
    del obd_model

    def benchmark_forward(batch_size: int) -> float:
        model = build_model().to(device).eval()
        batch = torch.randn(batch_size, 3, 224, 224, device=device)

        def run():
            with torch.no_grad():
                return model(batch)

        value = benchmark_ms(run, repeats=repeats, device=device)
        del batch, model
        release_memory(device)
        return value

    def benchmark_mag() -> float:
        model = build_model().to(device).eval()

        def run():
            return magnitude_scan(model)

        value = benchmark_ms(run, repeats=repeats, device=device)
        del model
        release_memory(device)
        return value

    def benchmark_path_mag() -> float:
        model = build_model().to(device).eval()

        def run():
            return get_path_magnitude_scores(
                model,
                input_shape=(1, 3, 224, 224),
                device=device,
            )

        value = benchmark_ms(run, repeats=repeats, device=device)
        del model
        release_memory(device)
        return value

    def benchmark_obd(batch_size: int) -> float:
        model = build_model().to(device).eval()

        def run():
            return synthetic_obd_scores(model, batch_size=batch_size, device=device)

        value = benchmark_ms(run, repeats=repeats, device=device)
        del model
        release_memory(device)
        return value

    forward_bs1_ms = benchmark_forward(1)
    forward_bs128_ms = benchmark_forward(128)
    mag_ms = benchmark_mag()
    obd_bs1_ms = benchmark_obd(1)
    obd_bs128_ms = benchmark_obd(128)
    path_mag_ms = benchmark_path_mag()

    return {
        "network": model_name,
        "obd_param_count": obd_param_count,
        "forward_bs1_ms": forward_bs1_ms,
        "forward_bs128_ms": forward_bs128_ms,
        "mag_ms": mag_ms,
        "obd_bs1_ms": obd_bs1_ms,
        "obd_bs128_ms": obd_bs128_ms,
        "path_mag_ms": path_mag_ms,
        "forward": f"{forward_bs1_ms:.1f}--{forward_bs128_ms:.1f}",
        "mag": f"{mag_ms:.1f}",
        "obd": f"{obd_bs1_ms:.1f}--{obd_bs128_ms:.1f}",
        "path_mag": f"{path_mag_ms:.1f}",
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = [
        benchmark_network("AlexNet", models.alexnet, repeats=args.repeats, device=device),
        benchmark_network("VGG16", models.vgg16, repeats=args.repeats, device=device),
        benchmark_network("ResNet18", models.resnet18, repeats=args.repeats, device=device),
    ]
    df = pd.DataFrame(rows)
    args.saving_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saving_path, index=False)
    print(f"Wrote {args.saving_path}.")
    print(df.loc[:, ["network", "forward", "mag", "obd", "path_mag"]].to_string(index=False))


if __name__ == "__main__":
    main()
