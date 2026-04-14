from __future__ import annotations

from typing import Any

import torch
from torch.nn.utils import prune

from pathnorm import validate_path_norm_support
from .pathnorm_transform import (
    infer_device,
    infer_dtype,
    prepare_path_norm_model,
    resolve_output_tensor,
    unwrap_model,
)


def get_path_magnitude_scores(
    model: torch.nn.Module,
    *,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    input_tensor: torch.Tensor | None = None,
    device: Any | None = None,
    working_dtype: torch.dtype = torch.float64,
) -> tuple[
    dict[tuple[torch.nn.Module, str], torch.Tensor],
    list[tuple[torch.nn.Module, str]],
]:
    """
    Compute the path-magnitude scores used in the ICML pruning paper:

        PhiCost(theta, i) = theta_i * d/d theta_i ||Phi(theta)||_1

    The returned `importance_scores` and `parameters_to_prune` are formatted
    for `torch.nn.utils.prune.global_unstructured`.
    """
    base_model = unwrap_model(model)
    validate_path_norm_support(
        base_model,
        input_shape=input_shape,
        input_tensor=input_tensor,
        strict_graph=True,
    ).raise_if_unsupported()
    device = infer_device(base_model, device)
    dtype = infer_dtype(base_model, working_dtype)
    working_model, x, _, _ = prepare_path_norm_model(
        base_model,
        input_shape=input_shape,
        input_tensor=input_tensor,
        device=device,
        working_dtype=dtype,
    )

    tensor_output = resolve_output_tensor(working_model(x), output_transform=None)
    tensor_output.sum().backward()

    original_modules = dict(base_model.named_modules())
    original_named_parameters = dict(base_model.named_parameters())

    importance_scores: dict[tuple[torch.nn.Module, str], torch.Tensor] = {}
    parameters_to_prune: list[tuple[torch.nn.Module, str]] = []

    for name, transformed_param in working_model.named_parameters():
        if transformed_param.grad is None:
            continue
        if "." not in name:
            continue

        module_name, param_name = name.rsplit(".", 1)
        original_module = original_modules.get(module_name)
        if not isinstance(original_module, (torch.nn.Conv2d, torch.nn.Linear)):
            continue

        if prune.is_pruned(original_module):
            if param_name != "weight_orig":
                continue
            prune_name = "weight"
            source_name = f"{module_name}.weight_orig"
        else:
            if param_name != "weight":
                continue
            prune_name = "weight"
            source_name = f"{module_name}.weight"

        original_param = original_named_parameters[source_name]
        score = torch.abs(original_param.detach()).to(
            device=transformed_param.grad.device,
            dtype=transformed_param.grad.dtype,
        ) * transformed_param.grad.detach()
        if torch.any(score < 0):
            raise ValueError(
                f"Negative path-magnitude score encountered for `{module_name}`."
            )

        entry = (original_module, prune_name)
        importance_scores[entry] = score
        parameters_to_prune.append(entry)

    return importance_scores, parameters_to_prune


def apply_path_magnitude_pruning(
    model: torch.nn.Module,
    amount: float,
    *,
    input_shape: tuple[int, ...] = (1, 3, 224, 224),
    input_tensor: torch.Tensor | None = None,
    device: Any | None = None,
    working_dtype: torch.dtype = torch.float64,
) -> dict[tuple[torch.nn.Module, str], torch.Tensor]:
    importance_scores, parameters_to_prune = get_path_magnitude_scores(
        model,
        input_shape=input_shape,
        input_tensor=input_tensor,
        device=device,
        working_dtype=working_dtype,
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        importance_scores=importance_scores,
        amount=amount,
    )
    return importance_scores
