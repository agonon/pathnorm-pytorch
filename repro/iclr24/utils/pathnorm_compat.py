from __future__ import annotations

import torch

from pathnorm import compute_path_norm as _compute_path_norm
from pathnorm import compute_path_norms as _compute_path_norms


def get_path_norm(
    model,
    name,
    device,
    exponent=1,
    constant_input=1,
    in_place=True,
    x=None,
):
    del name
    input_shape = tuple(x.shape) if x is not None else (1, 3, 224, 224)
    return _compute_path_norm(
        model,
        input_shape=input_shape,
        input_tensor=x,
        exponent=exponent,
        constant_input=constant_input,
        device=device,
        in_place=in_place,
        working_dtype=torch.float64,
    )


def compute_path_norms(model, name, exponents, device, data_parallel):
    del name, data_parallel
    return _compute_path_norms(
        model,
        input_shape=(1, 3, 224, 224),
        exponents=exponents,
        device=device,
        in_place=True,
        working_dtype=torch.float64,
    )
