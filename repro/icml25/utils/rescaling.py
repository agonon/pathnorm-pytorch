from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


DEFAULT_RESCALE_FACTORS = (1, 128, 4096)


def get_resnet18_block_triplets(
    model: torch.nn.Module,
) -> list[tuple[torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.Conv2d]]:
    return [
        (model.layer1[0].conv1, model.layer1[0].bn1, model.layer1[0].conv2),
        (model.layer1[1].conv1, model.layer1[1].bn1, model.layer1[1].conv2),
        (model.layer2[0].conv1, model.layer2[0].bn1, model.layer2[0].conv2),
        (model.layer2[1].conv1, model.layer2[1].bn1, model.layer2[1].conv2),
        (model.layer3[0].conv1, model.layer3[0].bn1, model.layer3[0].conv2),
        (model.layer3[1].conv1, model.layer3[1].bn1, model.layer3[1].conv2),
        (model.layer4[0].conv1, model.layer4[0].bn1, model.layer4[0].conv2),
        (model.layer4[1].conv1, model.layer4[1].bn1, model.layer4[1].conv2),
    ]


def _rescale_conv_bn_conv(
    conv1_weight: torch.Tensor,
    conv2_weight: torch.Tensor,
    running_mean: torch.Tensor,
    bn_bias: torch.Tensor,
    *,
    factors: Iterable[int] = DEFAULT_RESCALE_FACTORS,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = rng or np.random.default_rng()
    factors = np.asarray(tuple(factors), dtype=np.int64)

    rescaled_conv1 = conv1_weight.detach().clone()
    rescaled_conv2 = conv2_weight.detach().clone()
    rescaled_running_mean = running_mean.detach().clone()
    rescaled_bias = bn_bias.detach().clone()

    out_channels1 = int(rescaled_conv1.shape[0])
    in_channels2 = int(rescaled_conv2.shape[1])
    if out_channels1 != in_channels2:
        raise ValueError(
            "The output channels of the first convolution must match the input "
            "channels of the second convolution."
        )

    for channel in range(out_channels1):
        factor = int(rng.choice(factors))
        rescaled_conv1[channel, :, :, :] *= factor
        rescaled_conv2[:, channel, :, :] /= factor
        rescaled_running_mean[channel] *= factor
        rescaled_bias[channel] *= factor

    return (
        rescaled_conv1,
        rescaled_conv2,
        rescaled_running_mean,
        rescaled_bias,
    )


def random_rescale_resnet18(
    model: torch.nn.Module,
    *,
    factors: Iterable[int] = DEFAULT_RESCALE_FACTORS,
    rng: np.random.Generator | None = None,
) -> None:
    """
    Apply the random neuron-wise rescaling described in the ICML paper appendix
    to each `(conv1, bn1, conv2)` triplet of a torchvision ResNet-18 basic block.
    """
    for conv1, bn, conv2 in get_resnet18_block_triplets(model):
        (
            rescaled_conv1,
            rescaled_conv2,
            rescaled_running_mean,
            rescaled_bias,
        ) = _rescale_conv_bn_conv(
            conv1.weight,
            conv2.weight,
            bn.running_mean,
            bn.bias,
            factors=factors,
            rng=rng,
        )
        conv1.weight.data = rescaled_conv1
        conv2.weight.data = rescaled_conv2
        bn.running_mean.data = rescaled_running_mean
        bn.bias.data = rescaled_bias
