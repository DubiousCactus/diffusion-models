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


# TODO: Refactor these 3 following blocks into something DRY
class IdentityResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        temporal_projection_stride: int,
        norm_groups: int = 32,
        norm: str = "group",
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels_in, channels_out, 3, padding=1, stride=1)
        self.norm1 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if norm == "group"
            else torch.nn.BatchNorm2d(channels_out)
        )
        self.nonlin = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(channels_out, channels_out, 3, padding=1, stride=1)
        self.norm2 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if norm == "group"
            else torch.nn.BatchNorm2d(channels_out)
        )
        self.out_relu = torch.nn.LeakyReLU()
        self.temporal_projection = torch.nn.Conv2d(
            temporal_channels,
            channels_out,
            1,
            padding=0,
            stride=temporal_projection_stride,
        )
        self.residual_rescaling = (
            torch.nn.Conv2d(channels_in, channels_out, 1, padding=0, stride=1)
            if channels_in != channels_out
            else torch.nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        _x = x
        # print(f"Starting with x.shape = {x.shape}")
        x = self.conv1(x)
        # print(f"After conv1, x.shape = {x.shape}")
        x = self.norm1(x)
        x = self.nonlin(x)
        # print(f"Temb is {t_emb.shape}")
        # print(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x += self.temporal_projection(t_emb)
        x = self.conv2(x)
        # print(f"After conv2, x.shape = {x.shape}")
        x = self.norm2(x)
        # print(f"Adding _x of shape {_x.shape} to x of shape {x.shape}")
        return self.out_relu(x + self.residual_rescaling(_x))


class DownScaleResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        temporal_projection_stride: int,
        pooling: bool = True,
        norm_groups: int = 32,
        norm: str = "group",
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            channels_in, channels_out, 3, padding=1, stride=2 if not pooling else 1
        )
        self.norm1 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if norm == "group"
            else torch.nn.BatchNorm2d(channels_out)
        )
        self.nonlin = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(channels_out, channels_out, 3, padding=1, stride=1)
        self.norm2 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if norm == "group"
            else torch.nn.BatchNorm2d(channels_out)
        )
        self.out_relu = torch.nn.LeakyReLU()
        self.temporal_projection = torch.nn.Conv2d(
            temporal_channels,
            channels_out,
            1,
            padding=0,
            stride=temporal_projection_stride,
        )
        self.pooling = torch.nn.AvgPool2d(2) if pooling else torch.nn.Identity()
        self.residual_rescaling = torch.nn.Conv2d(
            channels_in, channels_out, 1, padding=0, stride=2
        )  # if pooling else torch.nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        _x = x
        # print(f"Starting with x.shape = {x.shape}")
        x = self.conv1(x)
        # print(f"After conv1, x.shape = {x.shape}")
        x = self.norm1(x)
        x = self.nonlin(x)
        # print(f"Temb is {t_emb.shape}")
        # print(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x += self.temporal_projection(t_emb)
        x = self.conv2(x)
        # print(f"After conv2, x.shape = {x.shape}")
        x = self.pooling(x)
        # print(f"After pooling, x.shape = {x.shape}")
        x = self.norm2(x)
        # print(f"Adding _x of shape {_x.shape} to x of shape {x.shape}")
        # print(f"Rescaled _x is {self.residual_rescaling(_x).shape}")
        return self.out_relu(x + self.residual_rescaling(_x))


class UpScaleResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        temporal_projection_stride: int,
        upsampling: bool = False,
        output_padding: int = 0,
        norm_groups: int = 32,
        norm: str = "group",
    ):
        super().__init__()
        self.upconv1 = torch.nn.ConvTranspose2d(
            channels_in, channels_out, 3, padding=1, stride=1
        )
        self.norm1 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if norm == "group"
            else torch.nn.BatchNorm2d(channels_out)
        )
        self.nonlin = torch.nn.LeakyReLU()
        self.upconv2 = torch.nn.ConvTranspose2d(
            channels_out,
            channels_out,
            3,
            padding=1,
            stride=2 if not upsampling else 1,
            output_padding=output_padding,
        )
        self.norm2 = (
            torch.nn.GroupNorm(norm_groups, channels_out)
            if norm == "group"
            else torch.nn.BatchNorm2d(channels_out)
        )
        self.out_relu = torch.nn.LeakyReLU()
        self.temporal_projection = torch.nn.Conv2d(
            temporal_channels,
            channels_out,
            1,
            padding=0,
            stride=temporal_projection_stride,
        )
        self.upsampling = (
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if upsampling
            else torch.nn.Identity()
        )
        self.residual_rescaling = torch.nn.ConvTranspose2d(
            channels_in,
            channels_out,
            1,
            padding=0,
            stride=2,
            output_padding=output_padding,
        )  # if pooling else torch.nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        _x = x
        # print(f"Starting with x.shape = {x.shape}")
        x = self.upconv1(x)
        # print(f"After upconv1, x.shape = {x.shape}")
        x = self.norm1(x)
        x = self.nonlin(x)
        # print(f"Temb is {t_emb.shape}")
        # print(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x += self.temporal_projection(t_emb)
        x = self.upconv2(x)
        # print(f"After upconv2, x.shape = {x.shape}")
        x = self.upsampling(x)
        # print(f"After upsampling, x.shape = {x.shape}")
        x = self.norm2(x)
        # print(f"Adding _x of shape {_x.shape} to x of shape {x.shape}")
        # print(f"Rescaled _x is {self.residual_rescaling(_x).shape}")
        return self.out_relu(x + self.residual_rescaling(_x))


class UNetBackboneModel(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(
        self,
        input_shape: Tuple[int],
        time_encoder: torch.nn.Module,
        temporal_channels: int,
        norm: str = "group",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.time_encoder = time_encoder
        self.temporal_network = torch.nn.Identity()
        self.in_channels = input_shape[-1]
        assert input_shape[0] == input_shape[1]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.temporal_network(self.time_encoder(t))
        t = t.unsqueeze(-1).unsqueeze(-1)
        t = t.repeat(1, 1, self.input_shape[0], self.input_shape[1])
        return self.forward_impl(x, t)


class UNetBackboneModelLarge(UNetBackboneModel):
    def __init__(
        self,
        input_shape: Tuple[int],
        time_encoder: torch.nn.Module,
        temporal_channels: int,
        norm: str = "group",
    ):
        super().__init__(input_shape, time_encoder, temporal_channels, norm=norm)
        self.temporal_network = torch.nn.Sequential(
            torch.nn.Linear(temporal_channels, temporal_channels),
            # torch.nn.BatchNorm1d(temporal_channels),
            torch.nn.GELU(),
            torch.nn.Linear(temporal_channels, temporal_channels),
            # torch.nn.BatchNorm1d(temporal_channels),
            # torch.nn.Tanh(),
        )
        # TODO: Global self-attention bewtween some blocks (instead of residualblock?)
        self.identity1 = IdentityResidualBlock(
            self.in_channels,
            128,
            temporal_channels,
            temporal_projection_stride=1,
            norm=norm,
        )
        self.identity2 = IdentityResidualBlock(
            128, 128, temporal_channels, temporal_projection_stride=1, norm=norm
        )
        self.down1 = DownScaleResidualBlock(
            128,
            128,
            temporal_channels,
            temporal_projection_stride=2,
            pooling=False,
            norm=norm,
        )
        self.down2 = DownScaleResidualBlock(
            128,
            256,
            temporal_channels,
            temporal_projection_stride=4,
            pooling=False,
            norm=norm,
        )
        self.down3 = DownScaleResidualBlock(
            256,
            512,
            temporal_channels,
            temporal_projection_stride=8,
            pooling=False,
            norm=norm,
        )
        self.down4 = DownScaleResidualBlock(
            512,
            512,
            temporal_channels,
            temporal_projection_stride=16,
            pooling=False,
            norm=norm,
        )
        self.tunnel1 = IdentityResidualBlock(
            512, 512, temporal_channels, temporal_projection_stride=16, norm=norm
        )  # This is the middle 'bottleneck'
        self.tunnel2 = IdentityResidualBlock(
            512, 512, temporal_channels, temporal_projection_stride=16, norm=norm
        )
        self.up1 = UpScaleResidualBlock(
            512 + 512,
            256,
            temporal_channels,
            temporal_projection_stride=16,
            upsampling=False,
            output_padding=1,
            norm=norm,
        )
        self.up2 = UpScaleResidualBlock(
            256 + 512,
            128,
            temporal_channels,
            temporal_projection_stride=8,
            upsampling=False,
            output_padding=1,
            norm=norm,
        )
        self.up3 = UpScaleResidualBlock(
            256 + 128,
            64,
            temporal_channels,
            temporal_projection_stride=4,
            upsampling=False,
            output_padding=1,
            norm=norm,
        )
        self.up4 = UpScaleResidualBlock(
            128 + 64,
            32,
            temporal_channels,
            temporal_projection_stride=2,
            upsampling=False,
            output_padding=1,
            norm_groups=16,
            norm=norm,
        )
        self.identity3 = IdentityResidualBlock(
            32 + 128, 32 + 128, temporal_channels, temporal_projection_stride=1
        )
        self.identity4 = IdentityResidualBlock(
            32 + 128, 128, temporal_channels, temporal_projection_stride=1
        )  # This one changes the number of channels
        self.out_conv = torch.nn.Conv2d(128, self.in_channels, 1, padding=0, stride=1)

    def forward_impl(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x1 = self.identity1(x, t)
        x2 = self.identity2(x1, t)
        x3 = self.down1(x2, t)
        x4 = self.down2(x3, t)
        x5 = self.down3(x4, t)
        x6 = self.down4(x5, t)
        x7 = self.tunnel1(x6, t)
        x8 = self.tunnel2(x7, t)
        x9 = self.up1(torch.cat((x8, x6), dim=1), t)
        x10 = self.up2(torch.cat((x9, x5), dim=1), t)
        x11 = self.up3(torch.cat((x10, x4), dim=1), t)
        x12 = self.up4(torch.cat((x11, x3), dim=1), t)
        x13 = self.identity3(torch.cat((x12, x2), dim=1), t)
        x14 = self.identity4(x13, t)
        return self.out_conv(x14)


class UNetBackboneModelSmall(UNetBackboneModel):
    def __init__(
        self,
        input_shape: Tuple[int],
        time_encoder: torch.nn.Module,
        temporal_channels: int,
        norm: str = "group",
    ):
        super().__init__(input_shape, time_encoder, temporal_channels, norm=norm)
        self.temporal_network = torch.nn.Sequential(
            torch.nn.Linear(temporal_channels, temporal_channels),
            # torch.nn.BatchNorm1d(temporal_channels),
            torch.nn.GELU(),
            torch.nn.Linear(temporal_channels, temporal_channels),
            # torch.nn.BatchNorm1d(temporal_channels),
            # torch.nn.Tanh(),
        )
        self.identity1 = IdentityResidualBlock(
            self.in_channels,
            64,
            temporal_channels,
            temporal_projection_stride=1,
            norm=norm,
        )
        self.down1 = DownScaleResidualBlock(
            64,
            64,
            temporal_channels,
            temporal_projection_stride=2,
            pooling=False,
            norm=norm,
        )
        self.down2 = DownScaleResidualBlock(
            64,
            128,
            temporal_channels,
            temporal_projection_stride=4,
            pooling=False,
            norm=norm,
        )
        self.down3 = DownScaleResidualBlock(
            128,
            256,
            temporal_channels,
            temporal_projection_stride=8,
            pooling=False,
            norm=norm,
        )
        self.down4 = DownScaleResidualBlock(
            256,
            256,
            temporal_channels,
            temporal_projection_stride=16,
            pooling=False,
            norm=norm,
        )
        self.tunnel1 = IdentityResidualBlock(
            256, 256, temporal_channels, temporal_projection_stride=16, norm=norm
        )  # This is the middle 'bottleneck'
        self.tunnel2 = IdentityResidualBlock(
            256, 256, temporal_channels, temporal_projection_stride=16, norm=norm
        )
        self.up1 = UpScaleResidualBlock(
            512,
            128,
            temporal_channels,
            temporal_projection_stride=16,
            upsampling=False,
            output_padding=1,
            norm=norm,
        )
        self.up2 = UpScaleResidualBlock(
            128 + 256,
            64,
            temporal_channels,
            temporal_projection_stride=8,
            upsampling=False,
            output_padding=1,
            norm_groups=16,
            norm=norm,
        )
        self.up3 = UpScaleResidualBlock(
            64 + 128,
            32,
            temporal_channels,
            temporal_projection_stride=4,
            upsampling=False,
            output_padding=1,
            norm_groups=8,
            norm=norm,
        )
        self.up4 = UpScaleResidualBlock(
            96,
            16,
            temporal_channels,
            temporal_projection_stride=2,
            upsampling=False,
            output_padding=1,
            norm_groups=4,
            norm=norm,
        )
        self.identity3 = IdentityResidualBlock(
            16,
            16,
            temporal_channels,
            temporal_projection_stride=1,
            norm_groups=4,
            norm=norm,
        )
        self.out_conv = torch.nn.Conv2d(16, self.in_channels, 1, padding=0, stride=1)

    def forward_impl(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x1 = self.identity1(x, t)
        x2 = self.down1(x1, t)
        x3 = self.down2(x2, t)
        x4 = self.down3(x3, t)
        x5 = self.down4(x4, t)
        x6 = self.tunnel1(x5, t)
        x7 = self.tunnel2(x6, t)
        # The output of the final downsampling layer is concatenated with the output of the final
        # tunnel layer because they have the same shape H and W. Then we upscale those features and
        # conctenate the upscaled features with the output of the previous downsampling layer, and
        # so on.
        x8 = self.up1(torch.cat((x7, x5), dim=1), t)
        x9 = self.up2(torch.cat((x8, x4), dim=1), t)
        x10 = self.up3(torch.cat((x9, x3), dim=1), t)
        x11 = self.up4(torch.cat((x10, x2), dim=1), t)
        x12 = self.identity3(x11, t)
        return self.out_conv(x12)
