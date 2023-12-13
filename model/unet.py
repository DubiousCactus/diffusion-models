#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
My own U-Net implementation. I mostly used "Understanding Deep Learning" Chapter 18 as a reference,
the official TensorFlow implementation (https://github.com/hojonathanho/diffusion/) and the
lucidrains implementation for the nice temporal conditioning mechanism
(https://github.com/lucidrains/denoising-diffusion-pytorch).
"""


import abc
from typing import Tuple

import torch

from model.resnet import (
    TemporalDownScaleResidualBlock,
    TemporalIdentityResidualBlock,
    TemporalUpScaleResidualBlock,
)


class UNetBackboneModel(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        input_shape: Tuple[int],
        time_encoder: torch.nn.Module,
        temporal_channels: int,
        normalization: str = "group",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.time_encoder = time_encoder
        self.time_embedder = torch.nn.Identity()
        self.in_channels = input_shape[-1]
        assert input_shape[0] == input_shape[1]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.forward_impl(x, self.time_embedder(self.time_encoder(t)))


class UNetBackboneModelLarge(UNetBackboneModel):
    def __init__(
        self,
        input_shape: Tuple[int],
        time_encoder: torch.nn.Module,
        temporal_channels: int,
        normalization: str = "group",
        output_paddings: Tuple = (1, 1, 1, 1),
    ):
        super().__init__(
            input_shape, time_encoder, temporal_channels, normalization=normalization
        )
        self.time_embedder = torch.nn.Sequential(
            torch.nn.Linear(temporal_channels, temporal_channels),
            torch.nn.GELU(),
            torch.nn.Linear(temporal_channels, temporal_channels),
        )
        # TODO: Global self-attention bewtween some blocks (instead of residualblock?)
        self.identity1 = TemporalIdentityResidualBlock(
            self.in_channels,
            128,
            temporal_channels,
            normalization=normalization,
        )
        self.identity2 = TemporalIdentityResidualBlock(
            128, 128, temporal_channels, normalization=normalization
        )
        self.down1 = TemporalDownScaleResidualBlock(
            128,
            128,
            temporal_channels,
            normalization=normalization,
        )
        self.down2 = TemporalDownScaleResidualBlock(
            128,
            256,
            temporal_channels,
            normalization=normalization,
        )
        self.down3 = TemporalDownScaleResidualBlock(
            256,
            512,
            temporal_channels,
            normalization=normalization,
        )
        self.down4 = TemporalDownScaleResidualBlock(
            512,
            512,
            temporal_channels,
            normalization=normalization,
        )
        self.tunnel1 = TemporalIdentityResidualBlock(
            512, 512, temporal_channels, normalization=normalization
        )  # This is the middle 'bottleneck'
        self.tunnel2 = TemporalIdentityResidualBlock(
            512, 512, temporal_channels, normalization=normalization
        )
        self.up1 = TemporalUpScaleResidualBlock(
            512 + 512,
            256,
            temporal_channels,
            output_padding=output_paddings[0],
            normalization=normalization,
        )
        self.up2 = TemporalUpScaleResidualBlock(
            256 + 512,
            128,
            temporal_channels,
            output_padding=output_paddings[1],
            normalization=normalization,
        )
        self.up3 = TemporalUpScaleResidualBlock(
            256 + 128,
            64,
            temporal_channels,
            output_padding=output_paddings[2],
            normalization=normalization,
        )
        self.up4 = TemporalUpScaleResidualBlock(
            128 + 64,
            32,
            temporal_channels,
            output_padding=output_paddings[3],
            normalization=normalization,
        )
        self.identity3 = TemporalIdentityResidualBlock(
            32 + 128,
            32 + 128,
            temporal_channels,
        )
        self.identity4 = TemporalIdentityResidualBlock(
            32 + 128,
            128,
            temporal_channels,
        )  # This one changes the number of channels
        self.out_conv = torch.nn.Conv2d(128, self.in_channels, 1, padding=0, stride=1)

    def forward_impl(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        x1 = self.identity1(x, t_embed)
        x2 = self.identity2(x1, t_embed)
        x3 = self.down1(x2, t_embed)
        x4 = self.down2(x3, t_embed)
        x5 = self.down3(x4, t_embed)
        x6 = self.down4(x5, t_embed)
        x7 = self.tunnel1(x6, t_embed)
        x8 = self.tunnel2(x7, t_embed)
        x9 = self.up1(torch.cat((x8, x6), dim=1), t_embed)
        x10 = self.up2(torch.cat((x9, x5), dim=1), t_embed)
        x11 = self.up3(torch.cat((x10, x4), dim=1), t_embed)
        x12 = self.up4(torch.cat((x11, x3), dim=1), t_embed)
        x13 = self.identity3(torch.cat((x12, x2), dim=1), t_embed)
        x14 = self.identity4(x13, t_embed)
        return self.out_conv(x14)


class UNetBackboneModelSmall(UNetBackboneModel):
    def __init__(
        self,
        input_shape: Tuple[int],
        time_encoder: torch.nn.Module,
        temporal_channels: int,
        normalization: str = "group",
        output_paddings: Tuple = (1, 1, 1, 1),
    ):
        super().__init__(
            input_shape, time_encoder, temporal_channels, normalization=normalization
        )
        self.time_embedder = torch.nn.Sequential(
            torch.nn.Linear(temporal_channels, temporal_channels),
            torch.nn.GELU(),
            torch.nn.Linear(temporal_channels, temporal_channels),
        )
        self.identity1 = TemporalIdentityResidualBlock(
            self.in_channels,
            64,
            temporal_channels,
            normalization=normalization,
        )
        self.down1 = TemporalDownScaleResidualBlock(
            64,
            64,
            temporal_channels,
            normalization=normalization,
        )
        self.down2 = TemporalDownScaleResidualBlock(
            64,
            128,
            temporal_channels,
            normalization=normalization,
        )
        self.down3 = TemporalDownScaleResidualBlock(
            128,
            256,
            temporal_channels,
            normalization=normalization,
        )
        self.down4 = TemporalDownScaleResidualBlock(
            256,
            256,
            temporal_channels,
            normalization=normalization,
        )
        self.tunnel1 = TemporalIdentityResidualBlock(
            256, 256, temporal_channels, normalization=normalization
        )  # This is the middle 'bottleneck'
        self.tunnel2 = TemporalIdentityResidualBlock(
            256, 256, temporal_channels, normalization=normalization
        )
        self.up1 = TemporalUpScaleResidualBlock(
            512,
            128,
            temporal_channels,
            output_padding=output_paddings[0],
            normalization=normalization,
        )
        self.up2 = TemporalUpScaleResidualBlock(
            128 + 256,
            64,
            temporal_channels,
            output_padding=output_paddings[1],
            normalization=normalization,
        )
        self.up3 = TemporalUpScaleResidualBlock(
            64 + 128,
            32,
            temporal_channels,
            output_padding=output_paddings[2],
            normalization=normalization,
        )
        self.up4 = TemporalUpScaleResidualBlock(
            96,
            16,
            temporal_channels,
            output_padding=output_paddings[3],
            norm_groups=4,
            normalization=normalization,
        )
        self.identity3 = TemporalIdentityResidualBlock(
            16,
            16,
            temporal_channels,
            norm_groups=4,
            normalization=normalization,
        )
        self.out_conv = torch.nn.Conv2d(16, self.in_channels, 1, padding=0, stride=1)

    def forward_impl(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
        x1 = self.identity1(x, t_embed)
        x2 = self.down1(x1, t_embed)
        x3 = self.down2(x2, t_embed)
        x4 = self.down3(x3, t_embed)
        x5 = self.down4(x4, t_embed)
        x6 = self.tunnel1(x5, t_embed)
        x7 = self.tunnel2(x6, t_embed)
        # The output of the final downsampling layer is concatenated with the output of the final
        # tunnel layer because they have the same shape H and W. Then we upscale those features and
        # conctenate the upscaled features with the output of the previous downsampling layer, and
        # so on.
        x8 = self.up1(torch.cat((x7, x5), dim=1), t_embed)
        x9 = self.up2(torch.cat((x8, x4), dim=1), t_embed)
        x10 = self.up3(torch.cat((x9, x3), dim=1), t_embed)
        x11 = self.up4(torch.cat((x10, x2), dim=1), t_embed)
        x12 = self.identity3(x11, t_embed)
        return self.out_conv(x12)
