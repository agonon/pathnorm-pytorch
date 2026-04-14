"""Standalone public API for path-norm computation on supported PyTorch models."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import prune


OutputTransform = Callable[[Any], torch.Tensor]


@dataclass
class PathNormSupportReport:
    supported: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.supported

    def format(self) -> str:
        lines = []
        if self.errors:
            lines.append("Unsupported path-norm model:")
            lines.extend(f"- {error}" for error in self.errors)
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines) if lines else "Model is supported."

    def raise_if_unsupported(self) -> None:
        if not self.supported:
            raise ValueError(self.format())


@dataclass(frozen=True)
class _PoolSpec:
    kind: str
    module_name: str
    in_channels: int
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    divisor: int
    reduce_dims: tuple[int, ...] = ()
    keepdim: bool = False


class _FunctionalMaxPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size: Any,
        stride: Any | None = None,
        padding: Any = 0,
        dilation: Any = 1,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            ceil_mode=self.ceil_mode,
            return_indices=False,
        )


class _FunctionalAvgPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size: Any,
        stride: Any | None = None,
        padding: Any = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
            count_include_pad=self.count_include_pad,
            divisor_override=self.divisor_override,
        )


class _FunctionalAdaptiveAvgPool2d(torch.nn.Module):
    def __init__(self, output_size: Any) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.adaptive_avg_pool2d(x, self.output_size)


class _SpatialMean(torch.nn.Module):
    def __init__(self, dim: tuple[int, ...], keepdim: bool) -> None:
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=self.dim, keepdim=self.keepdim)


class _MeanReduction(torch.nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
        keepdim: bool,
        scale: float,
    ) -> None:
        super().__init__()
        self.dims = dims
        self.keepdim = keepdim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=self.dims, keepdim=self.keepdim) * self.scale


@dataclass
class _OriginalState:
    tensor_state: dict[str, torch.Tensor] = field(default_factory=dict)
    module_state: dict[str, torch.nn.Module] = field(default_factory=dict)
    was_training: bool = False


_SUPPORTED_LEAF_MODULE_TYPES = (
    torch.nn.Conv2d,
    torch.nn.Linear,
    torch.nn.ReLU,
    torch.nn.ReLU6,
    torch.nn.Identity,
    torch.nn.Flatten,
    torch.nn.Dropout,
    torch.nn.Dropout1d,
    torch.nn.Dropout2d,
    torch.nn.Dropout3d,
    torch.nn.MaxPool2d,
    torch.nn.AvgPool2d,
    torch.nn.AdaptiveAvgPool2d,
    _FunctionalMaxPool2d,
    _FunctionalAvgPool2d,
    _FunctionalAdaptiveAvgPool2d,
    _SpatialMean,
    _BatchNorm,
)

_ALLOWED_GRAPH_FUNCTION_TOKENS = (
    "add",
    "getitem",
    "cat",
    "flatten",
    "getattr",
    "inceptionoutputs",
    "relu",
    "dropout",
)

_ALLOWED_GRAPH_METHOD_NAMES = {
    "contiguous",
    "flatten",
    "permute",
    "reshape",
    "size",
    "squeeze",
    "transpose",
    "unsqueeze",
    "view",
}

_UNSUPPORTED_GRAPH_TOKENS = (
    "attention",
    "batch_norm",
    "conv",
    "gelu",
    "group_norm",
    "interpolate",
    "layer_norm",
    "linear",
    "matmul",
    "mean",
    "scaled_dot_product_attention",
    "sigmoid",
    "silu",
    "softmax",
    "tanh",
    "upsample",
)


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(
        model,
        (
            torch.nn.DataParallel,
            torch.nn.parallel.DistributedDataParallel,
        ),
    ):
        return model.module
    return model


def _iter_leaf_modules(model: torch.nn.Module):
    for name, module in model.named_modules():
        if name == "":
            continue
        if any(module.children()):
            continue
        yield name, module


def _graph_target_name(target: Any) -> str:
    if isinstance(target, str):
        return target
    module_name = getattr(target, "__module__", "")
    qualname = getattr(target, "__qualname__", getattr(target, "__name__", ""))
    if module_name and qualname:
        return f"{module_name}.{qualname}"
    if qualname:
        return qualname
    return repr(target)


def _target_matches_any_token(target_name: str, tokens: Sequence[str]) -> bool:
    lowered = target_name.lower()
    return any(token in lowered for token in tokens)


def _node_name_or_target(node: Any) -> str:
    return f"{node.name} {_graph_target_name(getattr(node, 'target', ''))}"


def _node_matches_any_token(node: Any, tokens: Sequence[str]) -> bool:
    return _target_matches_any_token(_node_name_or_target(node), tokens)


def _adaptive_pool_params(in_size: int, out_size: int) -> tuple[int, int]:
    if out_size <= 0:
        raise ValueError(f"Adaptive average pooling requires positive output size, got {out_size}.")
    stride = max(1, in_size // out_size)
    kernel = in_size - (out_size - 1) * stride
    if kernel <= 0:
        raise ValueError(
            f"Adaptive average pooling from size {in_size} to {out_size} cannot be rewritten with a fixed kernel/stride."
        )
    return stride, kernel


def _normalize_adaptive_output_size(output_size: Any) -> tuple[int, int]:
    if isinstance(output_size, int):
        return int(output_size), int(output_size)
    if isinstance(output_size, tuple) and len(output_size) == 2:
        return int(output_size[0]), int(output_size[1])
    raise ValueError(f"Unsupported AdaptiveAvgPool2d output_size={output_size!r}.")


def _extract_mean_spec(node: Any) -> tuple[tuple[int, ...], bool] | None:
    if str(getattr(node, "target", "")) != "mean":
        return None

    dim = None
    keepdim = False
    if len(node.args) >= 2:
        dim = node.args[1]
    if "dim" in node.kwargs:
        dim = node.kwargs["dim"]
    if len(node.args) >= 3:
        keepdim = bool(node.args[2])
    if "keepdim" in node.kwargs:
        keepdim = bool(node.kwargs["keepdim"])
    if dim is None:
        return None

    if isinstance(dim, list):
        dim = tuple(dim)
    if isinstance(dim, tuple) and all(isinstance(value, int) for value in dim):
        return tuple(int(value) for value in dim), keepdim
    if isinstance(dim, int):
        return (int(dim),), keepdim
    return None


def _node_arg_or_kwarg(
    node: Any,
    position: int,
    keyword: str,
    default: Any = None,
) -> Any:
    if len(node.args) > position:
        return node.args[position]
    return node.kwargs.get(keyword, default)


def _build_rewrite_module_for_node(node: Any) -> torch.nn.Module | None:
    if node.op == "call_function":
        if _node_matches_any_token(node, ("max_pool",)):
            kernel_size = _node_arg_or_kwarg(node, 1, "kernel_size")
            if kernel_size is None:
                return None
            return _FunctionalMaxPool2d(
                kernel_size=kernel_size,
                stride=_node_arg_or_kwarg(node, 2, "stride"),
                padding=_node_arg_or_kwarg(node, 3, "padding", 0),
                dilation=_node_arg_or_kwarg(node, 4, "dilation", 1),
                ceil_mode=_node_arg_or_kwarg(node, 5, "ceil_mode", False),
            )
        if _node_matches_any_token(node, ("adaptive_avg_pool",)):
            output_size = _node_arg_or_kwarg(node, 1, "output_size")
            if output_size is None:
                return None
            return _FunctionalAdaptiveAvgPool2d(output_size=output_size)
        if _node_matches_any_token(node, ("avg_pool",)):
            kernel_size = _node_arg_or_kwarg(node, 1, "kernel_size")
            if kernel_size is None:
                return None
            return _FunctionalAvgPool2d(
                kernel_size=kernel_size,
                stride=_node_arg_or_kwarg(node, 2, "stride"),
                padding=_node_arg_or_kwarg(node, 3, "padding", 0),
                ceil_mode=_node_arg_or_kwarg(node, 4, "ceil_mode", False),
                count_include_pad=_node_arg_or_kwarg(node, 5, "count_include_pad", True),
                divisor_override=_node_arg_or_kwarg(node, 6, "divisor_override"),
            )

    if node.op == "call_method":
        mean_spec = _extract_mean_spec(node)
        if mean_spec is not None:
            dims, keepdim = mean_spec
            return _SpatialMean(dim=dims, keepdim=keepdim)

    return None


def _rewrite_supported_graph_ops(model: torch.nn.Module) -> tuple[torch.nn.Module, bool]:
    try:
        from torch.fx import symbolic_trace
    except Exception:
        return model, False

    try:
        traced = symbolic_trace(model)
    except Exception:
        return model, False

    graph = traced.graph
    rewrite_count = 0
    for node in list(graph.nodes):
        replacement_module = _build_rewrite_module_for_node(node)
        if replacement_module is None:
            continue

        module_name = f"_pathnorm_fx_{rewrite_count}"
        traced.add_module(module_name, replacement_module)
        input_node = node.args[0] if len(node.args) >= 1 else None
        if input_node is None:
            continue
        with graph.inserting_before(node):
            new_node = graph.call_module(module_name, args=(input_node,), kwargs={})
        node.replace_all_uses_with(new_node)
        graph.erase_node(node)
        rewrite_count += 1

    if rewrite_count == 0:
        return model, False

    graph.lint()
    traced.recompile()
    return traced, True


def _inspect_graph_support(model: torch.nn.Module) -> PathNormSupportReport:
    warnings: list[str] = []
    errors: list[str] = []

    try:
        from torch.fx import symbolic_trace

        traced = symbolic_trace(model)
    except Exception as exc:
        warnings.append(
            "FX graph inspection could not trace this model, so only module-level support checks were applied: "
            f"{type(exc).__name__}: {exc}"
        )
        return PathNormSupportReport(supported=True, warnings=warnings)

    traced_modules = dict(traced.named_modules())
    for node in traced.graph.nodes:
        if node.op in {"placeholder", "output", "get_attr"}:
            continue

        if node.op == "call_module":
            if str(node.target) not in traced_modules:
                warnings.append(
                    f"FX graph references module `{node.target}` at node `{node.name}`, "
                    "but it could not be resolved for support inspection."
                )
            continue

        if node.op == "call_function":
            target_name = _graph_target_name(node.target)
            if _target_matches_any_token(target_name, _ALLOWED_GRAPH_FUNCTION_TOKENS):
                continue
            if _build_rewrite_module_for_node(node) is not None:
                continue
            if _target_matches_any_token(target_name, _UNSUPPORTED_GRAPH_TOKENS):
                errors.append(
                    f"FX graph uses functional op `{target_name}` at node `{node.name}`, "
                    "which is unsupported for path-norm computation."
                )
            else:
                warnings.append(
                    f"FX graph uses function `{target_name}` at node `{node.name}`. "
                    "It was not classified as supported or unsupported by the graph checker."
                )
            continue

        if node.op == "call_method":
            method_name = str(node.target)
            if method_name in _ALLOWED_GRAPH_METHOD_NAMES:
                continue
            if _build_rewrite_module_for_node(node) is not None:
                continue
            if _target_matches_any_token(method_name, _UNSUPPORTED_GRAPH_TOKENS):
                errors.append(
                    f"FX graph uses tensor method `{method_name}` at node `{node.name}`, "
                    "which is unsupported for path-norm computation."
                )
            else:
                warnings.append(
                    f"FX graph uses tensor method `{method_name}` at node `{node.name}`. "
                    "It was not classified as supported or unsupported by the graph checker."
                )
            continue

        warnings.append(
            f"FX graph contains node `{node.name}` with op kind `{node.op}`, "
            "which is not inspected by the graph checker."
        )

    return PathNormSupportReport(
        supported=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def _to_2tuple(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple, got {value!r}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _replace_module(
    root_module: torch.nn.Module,
    module_name: str,
    new_module: torch.nn.Module,
) -> None:
    tokens = module_name.split(".")
    parent = root_module
    for token in tokens[:-1]:
        parent = parent._modules[token]
    parent._modules[tokens[-1]] = new_module


def _infer_device(model: torch.nn.Module, device: Any | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    for tensor in list(model.parameters()) + list(model.buffers()):
        return tensor.device
    return torch.device("cpu")


def _infer_dtype(
    model: torch.nn.Module,
    working_dtype: torch.dtype | None,
) -> torch.dtype:
    if working_dtype is not None:
        return working_dtype
    for tensor in list(model.parameters()) + list(model.buffers()):
        if torch.is_floating_point(tensor):
            return tensor.dtype
    return torch.float32


def _make_input_tensor(
    input_shape: Sequence[int] | None,
    input_tensor: torch.Tensor | None,
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


def _resolve_output_tensor(
    output: Any,
    output_transform: OutputTransform | None,
) -> torch.Tensor:
    if output_transform is not None:
        output = output_transform(output)

    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, dict) and "out" in output and isinstance(output["out"], torch.Tensor):
        return output["out"]

    if (
        isinstance(output, (tuple, list))
        and len(output) >= 1
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]

    raise TypeError(
        "Model output must be a tensor, a dict with key 'out', "
        "or a single-element tuple/list unless `output_transform` is provided."
    )


def validate_path_norm_support(
    model: torch.nn.Module,
    input_shape: Sequence[int] | None = None,
    input_tensor: torch.Tensor | None = None,
    *,
    strict_graph: bool = False,
) -> PathNormSupportReport:
    model = _unwrap_model(model)
    errors: list[str] = []
    warnings: list[str] = []

    if model.training:
        warnings.append(
            "The model is in training mode. Path-norm is computed with frozen/eval BatchNorm statistics."
        )

    for name, module in _iter_leaf_modules(model):
        if not isinstance(module, _SUPPORTED_LEAF_MODULE_TYPES):
            errors.append(
                f"`{name}` has unsupported type `{type(module).__name__}`. "
                "Currently supported leaves are Conv2d, Linear, ReLU, BatchNorm, "
                "MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Flatten, Identity, and Dropout."
            )
            continue

        if isinstance(module, _BatchNorm):
            if not module.affine or module.weight is None or module.bias is None:
                errors.append(
                    f"`{name}` has `affine=False`, which is unsupported for path-norm computation."
                )
            if module.running_mean is None or module.running_var is None:
                errors.append(
                    f"`{name}` does not expose frozen running statistics, so it cannot be folded into an affine map."
                )
            if not module.track_running_stats:
                errors.append(
                    f"`{name}` has `track_running_stats=False`, which is unsupported for path-norm computation."
                )

        if isinstance(module, torch.nn.MaxPool2d) and module.return_indices:
            errors.append(f"`{name}` uses `return_indices=True`, which is unsupported.")

        if isinstance(module, torch.nn.AvgPool2d):
            if module.ceil_mode:
                errors.append(f"`{name}` uses `ceil_mode=True`, which is unsupported.")
            padding = _to_2tuple(module.padding)
            if (
                module.divisor_override is None
                and not module.count_include_pad
                and padding != (0, 0)
            ):
                errors.append(
                    f"`{name}` uses `count_include_pad=False` with non-zero padding, "
                    "which cannot be rewritten as a fixed affine layer."
                )

        if isinstance(module, torch.nn.AdaptiveAvgPool2d):
            try:
                _normalize_adaptive_output_size(module.output_size)
            except ValueError as exc:
                errors.append(f"`{name}` is unsupported: {exc}")

    if strict_graph:
        graph_report = _inspect_graph_support(model)
        errors.extend(graph_report.errors)
        warnings.extend(graph_report.warnings)
    elif input_shape is None and input_tensor is None:
        warnings.append(
            "Validation used only the module structure. Functional ops inside `forward` were not inspected."
        )

    return PathNormSupportReport(
        supported=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def _collect_pool_specs(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
) -> dict[str, _PoolSpec]:
    specs: dict[str, _PoolSpec] = {}
    hooks = []

    def register_hook(name: str, module: torch.nn.Module) -> None:
        def hook(_module, inputs, _output):
            x = inputs[0]
            if not isinstance(x, torch.Tensor):
                raise ValueError(
                    f"Pooling module `{name}` expects tensor inputs, got {type(x).__name__}."
                )
            if isinstance(module, _SpatialMean):
                normalized_dims = tuple(dim if dim >= 0 else x.ndim + dim for dim in module.dim)
                divisor = 1
                for dim in normalized_dims:
                    divisor *= int(x.shape[dim])
                specs[name] = _PoolSpec(
                    kind="mean",
                    module_name=name,
                    in_channels=int(x.shape[1]) if x.ndim >= 2 else 1,
                    kernel_size=(1, 1),
                    stride=(1, 1),
                    padding=(0, 0),
                    dilation=(1, 1),
                    divisor=int(divisor),
                    reduce_dims=normalized_dims,
                    keepdim=module.keepdim,
                )
                return

            if x.ndim != 4:
                raise ValueError(
                    f"Pooling module `{name}` expects 4D tensor inputs, got {type(x).__name__}."
                )

            in_channels = int(x.shape[1])
            spatial_shape = tuple(int(v) for v in x.shape[-2:])

            if isinstance(module, (torch.nn.MaxPool2d, _FunctionalMaxPool2d)):
                specs[name] = _PoolSpec(
                    kind="max",
                    module_name=name,
                    in_channels=in_channels,
                    kernel_size=_to_2tuple(module.kernel_size),
                    stride=_to_2tuple(module.stride or module.kernel_size),
                    padding=_to_2tuple(module.padding),
                    dilation=_to_2tuple(module.dilation),
                    divisor=1,
                )
            elif isinstance(module, (torch.nn.AvgPool2d, _FunctionalAvgPool2d)):
                kernel_size = _to_2tuple(module.kernel_size)
                divisor = module.divisor_override or (kernel_size[0] * kernel_size[1])
                specs[name] = _PoolSpec(
                    kind="avg",
                    module_name=name,
                    in_channels=in_channels,
                    kernel_size=kernel_size,
                    stride=_to_2tuple(module.stride or module.kernel_size),
                    padding=_to_2tuple(module.padding),
                    dilation=(1, 1),
                    divisor=int(divisor),
                )
            elif isinstance(module, (torch.nn.AdaptiveAvgPool2d, _FunctionalAdaptiveAvgPool2d)):
                out_h, out_w = _normalize_adaptive_output_size(module.output_size)
                stride_h, kernel_h = _adaptive_pool_params(spatial_shape[0], out_h)
                stride_w, kernel_w = _adaptive_pool_params(spatial_shape[1], out_w)
                specs[name] = _PoolSpec(
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

    for name, module in _iter_leaf_modules(model):
        if isinstance(
            module,
            (
                torch.nn.MaxPool2d,
                torch.nn.AvgPool2d,
                torch.nn.AdaptiveAvgPool2d,
                _FunctionalMaxPool2d,
                _FunctionalAvgPool2d,
                _FunctionalAdaptiveAvgPool2d,
                _SpatialMean,
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


def _set_batchnorm_for_path_norm(module: _BatchNorm, exponent: float) -> None:
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


def _build_pool_replacement(
    spec: _PoolSpec,
    exponent: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    if spec.kind == "mean":
        return _MeanReduction(
            dims=spec.reduce_dims,
            keepdim=spec.keepdim,
            scale=float(1.0 / spec.divisor) ** exponent,
        )

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


def _apply_path_norm_transform(
    model: torch.nn.Module,
    pool_specs: dict[str, _PoolSpec],
    exponent: float,
    device: torch.device,
    dtype: torch.dtype,
    preserve_state: bool,
) -> _OriginalState:
    state = _OriginalState(was_training=model.training)

    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if prune.is_pruned(module):
                if preserve_state:
                    state.tensor_state[name + ".weight"] = module.weight_orig.detach().clone()
                module.weight_orig.data = torch.abs(module.weight_orig.detach())
                if exponent != 1:
                    module.weight_orig.data = torch.pow(module.weight_orig.detach(), exponent)
            else:
                if preserve_state:
                    state.tensor_state[name + ".weight"] = module.weight.detach().clone()
                module.weight.data = torch.abs(module.weight.detach())
                if exponent != 1:
                    module.weight.data = torch.pow(module.weight.detach(), exponent)

            if module.bias is not None:
                if preserve_state:
                    state.tensor_state[name + ".bias"] = module.bias.detach().clone()
                module.bias.data = torch.abs(module.bias.detach())
                if exponent != 1:
                    module.bias.data = torch.pow(module.bias.detach(), exponent)

        elif isinstance(module, _BatchNorm):
            if preserve_state:
                state.tensor_state[name + ".weight"] = module.weight.detach().clone()
                state.tensor_state[name + ".bias"] = module.bias.detach().clone()
                state.tensor_state[name + ".running_mean"] = module.running_mean.detach().clone()
                state.tensor_state[name + ".running_var"] = module.running_var.detach().clone()
            _set_batchnorm_for_path_norm(module, exponent)

    for name, spec in pool_specs.items():
        if preserve_state:
            original_module = dict(model.named_modules())[name]
            state.module_state[name] = copy.deepcopy(original_module)
        replacement = _build_pool_replacement(spec, exponent, device=device, dtype=dtype)
        _replace_module(model, name, replacement)

    return state


def _restore_model(model: torch.nn.Module, state: _OriginalState) -> None:
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            if prune.is_pruned(module) and name + ".weight" in state.tensor_state:
                module.weight_orig.data = state.tensor_state[name + ".weight"]
            elif name + ".weight" in state.tensor_state:
                module.weight.data = state.tensor_state[name + ".weight"]

            if module.bias is not None and name + ".bias" in state.tensor_state:
                module.bias.data = state.tensor_state[name + ".bias"]

        elif isinstance(module, _BatchNorm) and name + ".weight" in state.tensor_state:
            module.weight.data = state.tensor_state[name + ".weight"]
            module.bias.data = state.tensor_state[name + ".bias"]
            module.running_mean.data = state.tensor_state[name + ".running_mean"]
            module.running_var.data = state.tensor_state[name + ".running_var"]

    for name, original_module in state.module_state.items():
        _replace_module(model, name, original_module)

    model.train(state.was_training)


def compute_path_norm(
    model: torch.nn.Module,
    input_shape: Sequence[int] | None = None,
    *,
    input_tensor: torch.Tensor | None = None,
    exponent: float = 1,
    constant_input: float = 1.0,
    device: Any | None = None,
    in_place: bool = False,
    working_dtype: torch.dtype | None = torch.float64,
    output_transform: OutputTransform | None = None,
    validate: bool = True,
) -> float:
    """
    Compute the L^q path-norm of a supported PyTorch network.
    """
    base_model = _unwrap_model(model)

    if validate:
        validate_path_norm_support(
            base_model,
            input_shape=input_shape,
            input_tensor=input_tensor,
            strict_graph=True,
        ).raise_if_unsupported()

    device = _infer_device(base_model, device)
    model_dtype = _infer_dtype(base_model, None)

    if in_place:
        dtype = model_dtype if working_dtype is None else model_dtype
        working_model = base_model
    else:
        dtype = _infer_dtype(base_model, working_dtype)
        try:
            working_model = copy.deepcopy(base_model)
        except Exception as exc:
            raise RuntimeError(
                "Deep copy failed. Pass `in_place=True` if you want to transform the original model temporarily."
            ) from exc
        working_model = working_model.to(device=device, dtype=dtype)

    rewritten_model, graph_rewritten = _rewrite_supported_graph_ops(working_model)
    if in_place and graph_rewritten:
        raise ValueError(
            "Models that require rewriting functional pooling/mean ops are not supported with `in_place=True`."
        )
    working_model = rewritten_model

    shape_input = _make_input_tensor(
        input_shape=input_shape,
        input_tensor=input_tensor,
        constant_input=constant_input,
        device=device,
        dtype=dtype,
    )
    pool_specs = _collect_pool_specs(working_model, shape_input)

    state = _apply_path_norm_transform(
        working_model,
        pool_specs=pool_specs,
        exponent=exponent,
        device=device,
        dtype=dtype,
        preserve_state=in_place,
    )

    working_model.eval()

    x = _make_input_tensor(
        input_shape=input_shape,
        input_tensor=input_tensor,
        constant_input=constant_input,
        device=device,
        dtype=dtype,
    )

    with torch.no_grad():
        output = working_model(x)
        tensor_output = _resolve_output_tensor(output, output_transform)
        total = tensor_output.sum()
        if exponent == 0:
            path_norm = total.item()
        else:
            path_norm = torch.pow(total, 1.0 / exponent).item()

    if in_place:
        _restore_model(working_model, state)

    return path_norm


def compute_path_norms(
    model: torch.nn.Module,
    input_shape: Sequence[int] | None,
    exponents: Sequence[float],
    *,
    input_tensor: torch.Tensor | None = None,
    device: Any | None = None,
    in_place: bool = False,
    working_dtype: torch.dtype | None = torch.float64,
    output_transform: OutputTransform | None = None,
    validate: bool = True,
) -> list[float]:
    return [
        compute_path_norm(
            model,
            input_shape=input_shape,
            input_tensor=input_tensor,
            exponent=exponent,
            device=device,
            in_place=in_place,
            working_dtype=working_dtype,
            output_transform=output_transform,
            validate=validate,
        )
        for exponent in exponents
    ]


# Backward-compatible alias for older internal scripts.
get_path_norm = compute_path_norm


__all__ = [
    "compute_path_norm",
    "compute_path_norms",
]
