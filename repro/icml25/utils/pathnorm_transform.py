from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import prune


OutputTransform = Callable[[Any], torch.Tensor]


@dataclass(frozen=True)
class PoolSpec:
    kind: str
    module_name: str
    in_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    divisor: int


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(
        model,
        (
            torch.nn.DataParallel,
            torch.nn.parallel.DistributedDataParallel,
        ),
    ):
        return model.module
    return model


def iter_leaf_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if name == "":
            continue
        if any(module.children()):
            continue
        yield name, module


def to_2tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def adaptive_pool_params(in_size: int, out_size: int) -> tuple[int, int]:
    if out_size <= 0:
        raise ValueError(
            f"Adaptive average pooling requires positive output size, got {out_size}."
        )
    stride = max(1, in_size // out_size)
    kernel = in_size - (out_size - 1) * stride
    if kernel <= 0:
        raise ValueError(
            f"Adaptive average pooling from size {in_size} to {out_size} "
            "cannot be rewritten with a fixed kernel/stride."
        )
    return stride, kernel


def normalize_adaptive_output_size(output_size: Any) -> tuple[int, int]:
    if isinstance(output_size, int):
        return int(output_size), int(output_size)
    if isinstance(output_size, tuple) and len(output_size) == 2:
        return int(output_size[0]), int(output_size[1])
    raise ValueError(f"Unsupported AdaptiveAvgPool2d output_size={output_size!r}.")


def replace_module(
    root_module: torch.nn.Module,
    module_name: str,
    new_module: torch.nn.Module,
) -> None:
    tokens = module_name.split(".")
    parent = root_module
    for token in tokens[:-1]:
        parent = parent._modules[token]
    parent._modules[tokens[-1]] = new_module


def infer_device(model: torch.nn.Module, device: Any | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    return torch.device("cpu")


def infer_dtype(
    model: torch.nn.Module,
    working_dtype: torch.dtype | None,
) -> torch.dtype:
    if working_dtype is not None:
        return working_dtype
    for tensor in list(model.parameters()) + list(model.buffers()):
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return torch.float32


def make_input_tensor(
    input_shape: Sequence[int] | None,
    input_tensor: torch.Tensor | None,
    *,
    constant_input: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if input_tensor is not None:
        cloned = input_tensor.detach().clone()
        if cloned.ndim in (1, 3):
            cloned = cloned.unsqueeze(0)
        return cloned.to(device=device, dtype=dtype)
    if input_shape is None:
        raise ValueError("Please provide either `input_shape` or `input_tensor`.")
    normalized_shape = tuple(int(v) for v in input_shape)
    if len(normalized_shape) == 0:
        raise ValueError("`input_shape` must contain at least one dimension.")
    if len(normalized_shape) in (1, 3):
        normalized_shape = (1, *normalized_shape)
    return torch.full(normalized_shape, constant_input, device=device, dtype=dtype)


def resolve_output_tensor(
    output: Any,
    output_transform: OutputTransform | None = None,
) -> torch.Tensor:
    if output_transform is not None:
        output = output_transform(output)

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict) and "out" in output and isinstance(output["out"], torch.Tensor):
        return output["out"]

    if (
        isinstance(output, (tuple, list))
        and len(output) == 1
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]

    raise TypeError(
        "Model output must be a tensor, a dict with key 'out', "
        "or a single-element tuple/list unless `output_transform` is provided."
    )


def collect_pool_specs(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> dict[str, PoolSpec]:
    specs: dict[str, PoolSpec] = {}
    hooks = []

    def register_hook(name: str, module: torch.nn.Module) -> None:
        def hook(_module, inputs, _output):
            x = inputs[0]
            if not isinstance(x, torch.Tensor) or x.ndim != 4:
                raise ValueError(
                    f"Pooling module `{name}` expects 4D tensor inputs, got {type(x).__name__}."
                )

            in_channels = int(x.shape[1])
            spatial_shape = tuple(int(v) for v in x.shape[-2:])

            if isinstance(module, torch.nn.MaxPool2d):
                specs[name] = PoolSpec(
                    kind="max",
                    module_name=name,
                    in_channels=in_channels,
                    kernel_size=to_2tuple(module.kernel_size),
                    stride=to_2tuple(module.stride or module.kernel_size),
                    padding=to_2tuple(module.padding),
                    dilation=to_2tuple(module.dilation),
                    divisor=1,
                )
            elif isinstance(module, torch.nn.AvgPool2d):
                kernel_size = to_2tuple(module.kernel_size)
                divisor = module.divisor_override or (kernel_size[0] * kernel_size[1])
                specs[name] = PoolSpec(
                    kind="avg",
                    module_name=name,
                    in_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=to_2tuple(module.stride or module.kernel_size),
                    padding=to_2tuple(module.padding),
                    dilation=(1, 1),
                    divisor=int(divisor),
                )
            elif isinstance(module, torch.nn.AdaptiveAvgPool2d):
                out_h, out_w = normalize_adaptive_output_size(module.output_size)
                stride_h, kernel_h = adaptive_pool_params(spatial_shape[0], out_h)
                stride_w, kernel_w = adaptive_pool_params(spatial_shape[1], out_w)
                specs[name] = PoolSpec(
                    kind="adaptive_avg",
                    module_name=name,
                    in_channels=in_channels,
                    kernel_size=(kernel_h, kernel_w),
                    stride=(stride_h, stride_w),
                    padding=(0, 0),
                    dilation=(1, 1),
                    divisor=int(kernel_h * kernel_w),
                )

        hooks.append(module.register_forward_hook(hook))

    for name, module in iter_leaf_modules(model):
        if isinstance(
            module,
            (
                torch.nn.MaxPool2d,
                torch.nn.AvgPool2d,
                torch.nn.AdaptiveAvgPool2d,
            ),
        ):
            register_hook(name, module)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(input_tensor)
    model.train(was_training)

    for hook in hooks:
        hook.remove()

    return specs


def set_batchnorm_for_path_norm(module: _BatchNorm, exponent: float) -> None:
    effective_weight = module.weight.detach() / torch.sqrt(
        module.running_var.detach() + module.eps
    )
    effective_bias = (
        module.bias.detach() - module.running_mean.detach() * effective_weight
    )

    effective_weight = torch.abs(effective_weight)
    effective_bias = torch.abs(effective_bias)

    if exponent != 1:
        effective_weight = torch.pow(effective_weight, exponent)
        effective_bias = torch.pow(effective_bias, exponent)

    module.running_mean.data.zero_()
    module.running_var.data.fill_(1.0)
    module.weight.data = effective_weight * torch.sqrt(
        module.running_var.detach() + module.eps
    )
    module.bias.data = effective_bias


def build_pool_replacement(
    spec: PoolSpec,
    exponent: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Conv2d:
    weight_value = 1.0 if spec.kind == "max" else float(1.0 / spec.divisor) ** exponent
    conv = torch.nn.Conv2d(
        in_channels=spec.in_channels,
        out_channels=spec.in_channels,
        kernel_size=spec.kernel_size,
        stride=spec.stride,
        padding=spec.padding,
        dilation=spec.dilation,
        groups=spec.in_channels,
        bias=False,
        device=device,
        dtype=dtype,
    )
    conv.weight.data.fill_(weight_value)
    return conv


def apply_path_norm_transform(
    model: torch.nn.Module,
    *,
    pool_specs: dict[str, PoolSpec],
    exponent: float,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if prune.is_pruned(module):
                module.weight_orig.data = torch.abs(module.weight_orig.detach())
                if exponent != 1:
                    module.weight_orig.data = torch.pow(
                        module.weight_orig.detach(), exponent
                    )
            else:
                module.weight.data = torch.abs(module.weight.detach())
                if exponent != 1:
                    module.weight.data = torch.pow(module.weight.detach(), exponent)

            if module.bias is not None:
                module.bias.data = torch.abs(module.bias.detach())
                if exponent != 1:
                    module.bias.data = torch.pow(module.bias.detach(), exponent)

        elif isinstance(module, _BatchNorm):
            set_batchnorm_for_path_norm(module, exponent)

    for name, spec in pool_specs.items():
        replacement = build_pool_replacement(
            spec,
            exponent,
            device=device,
            dtype=dtype,
        )
        replace_module(model, name, replacement)


def prepare_path_norm_model(
    model: torch.nn.Module,
    *,
    input_shape: Sequence[int] | None,
    input_tensor: torch.Tensor | None,
    device: Any | None,
    working_dtype: torch.dtype,
) -> tuple[torch.nn.Module, torch.Tensor, torch.device, torch.dtype]:
    base_model = unwrap_model(model)
    device = infer_device(base_model, device)
    dtype = infer_dtype(base_model, working_dtype)

    working_model = copy.deepcopy(base_model).to(device=device, dtype=dtype)
    for param in working_model.parameters():
        param.requires_grad_(True)

    x = make_input_tensor(
        input_shape=input_shape,
        input_tensor=input_tensor,
        constant_input=1.0,
        device=device,
        dtype=dtype,
    )
    pool_specs = collect_pool_specs(working_model, x)
    apply_path_norm_transform(
        working_model,
        pool_specs=pool_specs,
        exponent=1,
        device=device,
        dtype=dtype,
    )
    working_model.eval()
    return working_model, x, device, dtype
