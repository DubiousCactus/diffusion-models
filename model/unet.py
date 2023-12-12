#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
My own U-Net implementation. I mostly used "Understanding Deep Learning" Chapter 18 as a reference.
"""


import abc
from typing import Tuple

import torch


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_dim: int,
        norm_groups: int = 8,
        normalization: str = "group",
        strides: Tuple[int, int] = (1, 1, 1),
        paddings: Tuple[int, int] = (1, 1, 0),
        kernels: Tuple[int, int] = (3, 3, 1),
        output_padding: int = 0,
        conv=torch.nn.Conv2d,
    ):
        super().__init__()
        kwargs = (
            {"output_padding": output_padding}
            if conv == torch.nn.ConvTranspose2d
            else {}
        )
        self.conv1 = conv(
            channels_in,
            channels_out,
            kernels[0],
            padding=paddings[0],
            stride=strides[0],
        )
        self.norm1 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if normalization == "group"
            else torch.nn.BatchNorm2d(channels_out)
        )
        self.nonlin = torch.nn.GELU()
        self.conv2 = conv(
            channels_out,
            channels_out,
            kernels[1],
            padding=paddings[1],
            stride=strides[1],
            **kwargs
        )
        self.norm2 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if normalization == "group"
            else torch.nn.BatchNorm2d(channels_out)
        )
        self.out_activation = torch.nn.GELU()
        self.temporal_projection = torch.nn.Linear(
            temporal_dim,
            channels_out * 2,
        )
        self.residual_scaling = (
            conv(
                channels_in,
                channels_out,
                kernels[2],
                padding=paddings[2],
                stride=strides[2],
                bias=False,
                **kwargs
            )
            if (channels_in != channels_out or strides[2] != 1 or paddings[2] != 0)
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        _x = x
        scale, shift = (
            self.temporal_projection(t_emb).unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        )
        # print(f"Starting with x.shape = {x.shape}")
        x = self.conv1(x)
        # print(f"After conv1, x.shape = {x.shape}")
        x = self.norm1(x)
        x *= (scale + 1) + shift
        x = self.nonlin(x)
        # print(f"Temb is {t_emb.shape}")
        # print(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x = self.conv2(x)
        # print(f"After conv2, x.shape = {x.shape}")
        # x = self.norm2(x)
        x *= (scale + 1) + shift
        # print(f"Adding _x of shape {_x.shape} (rescaled to {self.residual_scaling(_x).shape}) to x of shape {x.shape}")
        return self.out_activation(x + self.residual_scaling(_x))


class IdentityResidualBlock(ResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        norm_groups: int = 8,
        normalization: str = "group",
    ):
        super().__init__(
            channels_in, channels_out, temporal_channels, norm_groups, normalization
        )


class DownScaleResidualBlock(ResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        pooling: bool = False,  # TODO
        norm_groups: int = 8,
        normalization: str = "group",
    ):
        super().__init__(
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            strides=(2, 1, 2),
            paddings=(1, 1, 0),
            kernels=(3, 3, 1),
            conv=torch.nn.Conv2d,
        )


class UpScaleResidualBlock(ResidualBlock):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        upsampling: bool = False,  # TODO
        output_padding: int = 0,
        norm_groups: int = 8,
        normalization: str = "group",
    ):
        super().__init__(
            channels_in,
            channels_out,
            temporal_channels,
            norm_groups,
            normalization,
            strides=(1, 2, 2),
            paddings=(1, 1, 0),
            kernels=(3, 3, 1),
            output_padding=output_padding,
            conv=torch.nn.ConvTranspose2d,
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
        self.identity1 = IdentityResidualBlock(
            self.in_channels,
            128,
            temporal_channels,
            normalization=normalization,
        )
        self.identity2 = IdentityResidualBlock(
            128, 128, temporal_channels, normalization=normalization
        )
        self.down1 = DownScaleResidualBlock(
            128,
            128,
            temporal_channels,
            normalization=normalization,
        )
        self.down2 = DownScaleResidualBlock(
            128,
            256,
            temporal_channels,
            normalization=normalization,
        )
        self.down3 = DownScaleResidualBlock(
            256,
            512,
            temporal_channels,
            normalization=normalization,
        )
        self.down4 = DownScaleResidualBlock(
            512,
            512,
            temporal_channels,
            normalization=normalization,
        )
        self.tunnel1 = IdentityResidualBlock(
            512, 512, temporal_channels, normalization=normalization
        )  # This is the middle 'bottleneck'
        self.tunnel2 = IdentityResidualBlock(
            512, 512, temporal_channels, normalization=normalization
        )
        self.up1 = UpScaleResidualBlock(
            512 + 512,
            256,
            temporal_channels,
            output_padding=output_paddings[0],
            normalization=normalization,
        )
        self.up2 = UpScaleResidualBlock(
            256 + 512,
            128,
            temporal_channels,
            output_padding=output_paddings[1],
            normalization=normalization,
        )
        self.up3 = UpScaleResidualBlock(
            256 + 128,
            64,
            temporal_channels,
            output_padding=output_paddings[2],
            normalization=normalization,
        )
        self.up4 = UpScaleResidualBlock(
            128 + 64,
            32,
            temporal_channels,
            output_padding=output_paddings[3],
            normalization=normalization,
        )
        self.identity3 = IdentityResidualBlock(
            32 + 128,
            32 + 128,
            temporal_channels,
        )
        self.identity4 = IdentityResidualBlock(
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
        self.identity1 = IdentityResidualBlock(
            self.in_channels,
            64,
            temporal_channels,
            normalization=normalization,
        )
        self.down1 = DownScaleResidualBlock(
            64,
            64,
            temporal_channels,
            normalization=normalization,
        )
        self.down2 = DownScaleResidualBlock(
            64,
            128,
            temporal_channels,
            normalization=normalization,
        )
        self.down3 = DownScaleResidualBlock(
            128,
            256,
            temporal_channels,
            normalization=normalization,
        )
        self.down4 = DownScaleResidualBlock(
            256,
            256,
            temporal_channels,
            normalization=normalization,
        )
        self.tunnel1 = IdentityResidualBlock(
            256, 256, temporal_channels, normalization=normalization
        )  # This is the middle 'bottleneck'
        self.tunnel2 = IdentityResidualBlock(
            256, 256, temporal_channels, normalization=normalization
        )
        self.up1 = UpScaleResidualBlock(
            512,
            128,
            temporal_channels,
            output_padding=output_paddings[0],
            normalization=normalization,
        )
        self.up2 = UpScaleResidualBlock(
            128 + 256,
            64,
            temporal_channels,
            output_padding=output_paddings[1],
            normalization=normalization,
        )
        self.up3 = UpScaleResidualBlock(
            64 + 128,
            32,
            temporal_channels,
            output_padding=output_paddings[2],
            normalization=normalization,
        )
        self.up4 = UpScaleResidualBlock(
            96,
            16,
            temporal_channels,
            output_padding=output_paddings[3],
            norm_groups=4,
            normalization=normalization,
        )
        self.identity3 = IdentityResidualBlock(
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
