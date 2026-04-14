from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils import prune
import torchvision.models as models

from .path_magnitude import apply_path_magnitude_pruning
from .rescaling import random_rescale_resnet18


DEFAULT_ARCH = "resnet18"
DEFAULT_PRUNING_RATIOS = (0.1, 0.2, 0.4, 0.6, 0.8)
BODY_PRUNING_RATIOS = (0.4, 0.6, 0.8)
CURVE_PRUNING_RATIO = 0.4
DEFAULT_SWEEP_VARIANTS = (
    "path-magnitude",
    "path-magnitude-rescaled",
    "magnitude",
    "magnitude-rescaled",
)
TABLE_VARIANTS = (
    "path-magnitude",
    "magnitude",
    "magnitude-rescaled",
)
CURVE_VARIANTS = DEFAULT_SWEEP_VARIANTS
DEFAULT_EPOCHS = 90
DEFAULT_REWIND_EPOCH = 5
DEFAULT_LR = 0.1
DEFAULT_WEIGHT_DECAY = 0.0001
DEFAULT_LR_SCHEDULER = "multi-step"
DEFAULT_BATCH_SIZE = 1024
DEFAULT_WORKERS = 16

BODY_ROW_LABELS = {
    "path-magnitude": "Path-magnitude",
    "magnitude": "Magnitude",
    "magnitude-rescaled": "Magnitude (rescaled)",
}

FULL_ROW_LABELS = {
    "path-magnitude": "Path-Magnitude (*)",
    "magnitude": "Magnitude w/o Random Rescale",
    "magnitude-rescaled": "Magnitude w/ Random Rescale",
}

CURVE_LABELS = {
    "path-magnitude": "Path-magnitude",
    "path-magnitude-rescaled": "Path-magnitude (rescaled)",
    "magnitude": "Magnitude",
    "magnitude-rescaled": "Magnitude (rescaled)",
}


@dataclass(frozen=True)
class RunMetadata:
    variant: str
    pruning_ratio: float
    seed: int
    arch: str
    dense_dir: str
    rescaled_before_pruning: bool


def dense_run_dir(
    results_root: Path,
    *,
    seed: int,
    arch: str = DEFAULT_ARCH,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    epochs: int = DEFAULT_EPOCHS,
    lr_scheduler: str = DEFAULT_LR_SCHEDULER,
) -> Path:
    return (
        results_root
        / "dense"
        / f"seed={seed}"
        / arch
        / f"lr={lr}_wd={weight_decay}_epochs={epochs}_scheduler={lr_scheduler}"
    )


def pruning_run_dir(
    results_root: Path,
    *,
    variant: str,
    pruning_ratio: float,
    seed: int,
    arch: str = DEFAULT_ARCH,
) -> Path:
    return (
        results_root
        / "pruning"
        / f"seed={seed}"
        / arch
        / f"variant={variant}"
        / f"pruning_ratio={pruning_ratio}"
    )


def metadata_path(run_dir: Path) -> Path:
    return run_dir / "metadata.json"


def pruned_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "checkpoint_pruned.pth.tar"


def training_log_path(run_dir: Path) -> Path:
    return run_dir / "rank=0" / "csv" / "results.csv"


def dense_training_log_path(dense_dir: Path) -> Path:
    return dense_dir / "rank=0" / "csv" / "results.csv"


def best_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "model_best.pth.tar"


def epoch5_checkpoint_path(run_dir: Path) -> Path:
    return run_dir / "epoch_5.pth.tar"


def save_metadata(run_dir: Path, metadata: RunMetadata) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata_path(run_dir).write_text(json.dumps(asdict(metadata), indent=2) + "\n")


def load_metadata(run_dir: Path) -> RunMetadata | None:
    path = metadata_path(run_dir)
    if not path.exists():
        return None
    return RunMetadata(**json.loads(path.read_text()))


def variant_is_rescaled(variant: str) -> bool:
    return variant.endswith("-rescaled")


def variant_pruning_method(variant: str) -> str:
    if variant.startswith("path-magnitude"):
        return "path-magnitude"
    if variant.startswith("magnitude"):
        return "magnitude"
    raise ValueError(f"Unknown ICML variant: {variant!r}")


def build_resnet18(device: torch.device | str) -> torch.nn.Module:
    return models.resnet18(weights=None).to(device)


def parse_seed_from_path(path: Path) -> int | None:
    for part in path.parts:
        if part.startswith("seed="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def load_checkpoint(checkpoint_path: Path, device: torch.device | str) -> dict:
    return torch.load(checkpoint_path, map_location=torch.device(device))


def iter_prunable_modules(
    model: torch.nn.Module,
) -> Iterable[tuple[torch.nn.Module, str]]:
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            yield module, "weight"


def apply_magnitude_pruning(model: torch.nn.Module, amount: float) -> None:
    prune.global_unstructured(
        list(iter_prunable_modules(model)),
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


def apply_variant_pruning(
    model: torch.nn.Module,
    *,
    variant: str,
    pruning_ratio: float,
    device: torch.device,
) -> None:
    method = variant_pruning_method(variant)
    if method == "magnitude":
        apply_magnitude_pruning(model, pruning_ratio)
    elif method == "path-magnitude":
        apply_path_magnitude_pruning(
            model,
            pruning_ratio,
            input_shape=(1, 3, 224, 224),
            device=device,
        )
    else:
        raise ValueError(f"Unknown pruning method for variant {variant!r}")


def apply_rescaling_if_needed(
    model: torch.nn.Module,
    *,
    variant: str,
    seed: int,
) -> None:
    if not variant_is_rescaled(variant):
        return
    import numpy as np

    rng = np.random.default_rng(seed)
    random_rescale_resnet18(model, rng=rng)


def maybe_prepare_pruned_buffers(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    if not any(name.endswith("weight_orig") for name in state_dict):
        return
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and not prune.is_pruned(module):
            prune.l1_unstructured(module, name="weight", amount=0.0)


def rewind_pruned_model(
    model: torch.nn.Module,
    rewind_state_dict: dict[str, torch.Tensor],
) -> None:
    model_state = model.state_dict()
    for name, tensor in rewind_state_dict.items():
        if name.replace("weight", "weight_orig") in model_state:
            model_state[name.replace("weight", "weight_orig")] = tensor
        elif name in model_state:
            model_state[name] = tensor
    model.load_state_dict(model_state)


def summarize_training_log(log_path: Path) -> dict[str, float | int | str] | None:
    if not log_path.exists():
        return None
    df = pd.read_csv(log_path)
    if df.empty or "val/acc1" not in df.columns:
        return None
    numeric_columns = ["epoch", "train/acc1", "val/acc1", "test/acc1"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["epoch", "val/acc1", "test/acc1"])
    if df.empty:
        return None
    best_idx = df["val/acc1"].idxmax()
    row = df.loc[best_idx]
    return {
        "best_epoch": int(row["epoch"]),
        "best_val_acc1": float(row["val/acc1"]),
        "best_test_acc1": float(row["test/acc1"]),
        "source_log": str(log_path),
    }


def iter_dense_run_dirs(results_root: Path) -> list[Path]:
    dense_root = results_root / "dense"
    if not dense_root.exists():
        return []
    run_dirs = []
    for checkpoint_path in dense_root.rglob("model_best.pth.tar"):
        run_dirs.append(checkpoint_path.parent)
    return sorted(run_dirs)


def dense_baseline_summary(results_root: Path) -> dict[str, float | int | str] | None:
    summaries = []
    for dense_dir in iter_dense_run_dirs(results_root):
        summary = summarize_training_log(dense_training_log_path(dense_dir))
        if summary is None:
            continue
        summary = dict(summary)
        summary["seed"] = parse_seed_from_path(dense_dir)
        summary["dense_dir"] = str(dense_dir)
        summaries.append(summary)
    if not summaries:
        return None
    return {
        "mean_test_acc1": sum(s["best_test_acc1"] for s in summaries) / len(summaries),
        "mean_val_acc1": sum(s["best_val_acc1"] for s in summaries) / len(summaries),
        "mean_best_epoch": sum(s["best_epoch"] for s in summaries) / len(summaries),
        "num_runs": len(summaries),
    }


def build_dense_summary_frame(results_root: Path) -> pd.DataFrame:
    baseline = dense_baseline_summary(results_root)
    if baseline is None:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "row": "Dense baseline",
                "original_accuracy": baseline["mean_test_acc1"],
                "mean_val_acc1": baseline["mean_val_acc1"],
                "mean_best_epoch": baseline["mean_best_epoch"],
                "num_runs": baseline["num_runs"],
            }
        ]
    )


def iter_run_dirs(
    results_root: Path,
    *,
    variant: str,
    pruning_ratio: float,
) -> list[Path]:
    pruning_root = results_root / "pruning"
    if not pruning_root.exists():
        return []
    run_dirs = []
    for seed_dir in pruning_root.glob("seed=*"):
        candidate = seed_dir / DEFAULT_ARCH / f"variant={variant}" / f"pruning_ratio={pruning_ratio}"
        if candidate.exists():
            run_dirs.append(candidate)
    return sorted(run_dirs)


def aggregate_accuracy_rows(results_root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for variant in TABLE_VARIANTS:
        for pruning_ratio in DEFAULT_PRUNING_RATIOS:
            summaries = []
            for run_dir in iter_run_dirs(results_root, variant=variant, pruning_ratio=pruning_ratio):
                summary = summarize_training_log(training_log_path(run_dir))
                if summary is not None:
                    summaries.append(summary)
            if not summaries:
                continue
            rows.append(
                {
                    "variant": variant,
                    "pruning_ratio": pruning_ratio,
                    "mean_test_acc1": sum(s["best_test_acc1"] for s in summaries) / len(summaries),
                    "mean_val_acc1": sum(s["best_val_acc1"] for s in summaries) / len(summaries),
                    "mean_best_epoch": sum(s["best_epoch"] for s in summaries) / len(summaries),
                    "num_runs": len(summaries),
                }
            )
    return pd.DataFrame(rows)


def build_accuracy_tables(results_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    accuracy_rows = aggregate_accuracy_rows(results_root)
    baseline = dense_baseline_summary(results_root)
    if accuracy_rows.empty:
        return pd.DataFrame(), pd.DataFrame()

    def pivot_for(
        pruning_ratios: Iterable[float],
        *,
        row_labels: dict[str, str],
        include_none: bool,
    ) -> pd.DataFrame:
        subset = accuracy_rows.loc[accuracy_rows["pruning_ratio"].isin(tuple(pruning_ratios))].copy()
        if subset.empty:
            return pd.DataFrame()
        subset["paper_row"] = subset["variant"].map(row_labels)
        subset["pruning_level"] = subset["pruning_ratio"].map(lambda x: f"{int(100 * x)}%")
        table = subset.pivot(index="paper_row", columns="pruning_level", values="mean_test_acc1")
        desired_rows = [row_labels[v] for v in row_labels]
        desired_cols = [f"{int(100 * p)}%" for p in pruning_ratios]
        table = table.reindex(index=desired_rows, columns=desired_cols)
        if include_none and baseline is not None:
            table.insert(0, "none", baseline["mean_test_acc1"])
        return table.reset_index()

    body_df = pivot_for(
        BODY_PRUNING_RATIOS,
        row_labels=BODY_ROW_LABELS,
        include_none=False,
    )
    full_df = pivot_for(
        DEFAULT_PRUNING_RATIOS,
        row_labels=FULL_ROW_LABELS,
        include_none=True,
    )
    return body_df, full_df


def build_training_curve_payload(results_root: Path) -> dict[str, object] | None:
    ratio = CURVE_PRUNING_RATIO
    series_payload: list[dict[str, object]] = []
    for variant in CURVE_VARIANTS:
        run_dirs = iter_run_dirs(results_root, variant=variant, pruning_ratio=ratio)
        frames = []
        for run_dir in run_dirs:
            log_path = training_log_path(run_dir)
            if not log_path.exists():
                continue
            df = pd.read_csv(log_path)
            if "epoch" not in df.columns or "test/acc1" not in df.columns:
                continue
            df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
            df["test/acc1"] = pd.to_numeric(df["test/acc1"], errors="coerce")
            df = df.dropna(subset=["epoch", "test/acc1"])
            df = (
                df.sort_values("epoch")
                .drop_duplicates(subset=["epoch"], keep="last")
                .reset_index(drop=True)
            )
            if not df.empty:
                frames.append(df.loc[:, ["epoch", "test/acc1"]].reset_index(drop=True))
        if not frames:
            continue
        min_len = min(len(frame) for frame in frames)
        if min_len == 0:
            continue
        epochs = frames[0]["epoch"].iloc[:min_len].astype(int).tolist()
        stacked = torch.tensor(
            [frame["test/acc1"].iloc[:min_len].tolist() for frame in frames],
            dtype=torch.float64,
        )
        series_payload.append(
            {
                "variant": variant,
                "label": CURVE_LABELS[variant],
                "epoch": epochs,
                "test_acc1": stacked.mean(dim=0).tolist(),
                "num_runs": len(frames),
            }
        )
    if not series_payload:
        return None
    return {"pruning_ratio": ratio, "series": series_payload}
