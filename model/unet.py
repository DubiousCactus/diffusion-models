#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
My own U-Net implementation.
"""


from typing import Tuple

import torch


class IdentityResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        temporal_projection_stride: int,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels_in, channels_out, 3, padding=1, stride=1)
        self.norm1 = torch.nn.GroupNorm(32, channels_out)
        self.nonlin = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(channels_out, channels_out, 3, padding=1, stride=1)
        self.norm2 = torch.nn.GroupNorm(32, channels_out)
        self.out_relu = torch.nn.LeakyReLU()
        self.temporal_projection = torch.nn.Conv2d(
            temporal_channels,
            channels_out,
            1,
            padding=0,
            stride=temporal_projection_stride,
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        _x = x
        print(f"Starting with x.shape = {x.shape}")
        x = self.conv1(x)
        print(f"After conv1, x.shape = {x.shape}")
        x = self.norm1(x)
        x = self.nonlin(x)
        print(f"Temb is {t_emb.shape}")
        print(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x += self.temporal_projection(t_emb)
        x = self.conv2(x)
        print(f"After conv2, x.shape = {x.shape}")
        x = self.norm2(x)
        print(f"Adding _x of shape {_x.shape} to x of shape {x.shape}")
        return self.out_relu(x + _x)  # TODO: Rescale _x?


class DownScaleResidualBlock(torch.nn.Module):
    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        temporal_channels: int,
        temporal_projection_stride: int,
        pooling: bool = True,
    ):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            channels_in, channels_out, 3, padding=1, stride=2 if not pooling else 1
        )
        self.norm1 = torch.nn.GroupNorm(32, channels_out)
        self.nonlin = torch.nn.LeakyReLU()
        self.conv2 = torch.nn.Conv2d(channels_out, channels_out, 3, padding=1, stride=1)
        self.norm2 = torch.nn.GroupNorm(32, channels_out)
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
        print(f"Starting with x.shape = {x.shape}")
        x = self.conv1(x)
        print(f"After conv1, x.shape = {x.shape}")
        x = self.norm1(x)
        x = self.nonlin(x)
        print(f"Temb is {t_emb.shape}")
        print(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x += self.temporal_projection(t_emb)
        x = self.conv2(x)
        print(f"After conv2, x.shape = {x.shape}")
        x = self.pooling(x)
        print(f"After pooling, x.shape = {x.shape}")
        x = self.norm2(x)
        print(f"Adding _x of shape {_x.shape} to x of shape {x.shape}")
        print(f"Rescaled _x is {self.residual_rescaling(_x).shape}")
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
    ):
        super().__init__()
        self.upconv1 = torch.nn.ConvTranspose2d(
            channels_in, channels_out, 3, padding=1, stride=1
        )
        self.norm1 = torch.nn.GroupNorm(32, channels_out)
        self.nonlin = torch.nn.LeakyReLU()
        self.upconv2 = torch.nn.ConvTranspose2d(
            channels_out,
            channels_out,
            3,
            padding=1,
            stride=2 if not upsampling else 1,
            output_padding=output_padding,
        )
        self.norm2 = torch.nn.GroupNorm(32, channels_out)
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
        print(f"Starting with x.shape = {x.shape}")
        x = self.upconv1(x)
        print(f"After upconv1, x.shape = {x.shape}")
        x = self.norm1(x)
        x = self.nonlin(x)
        print(f"Temb is {t_emb.shape}")
        print(f"Temb projected is {self.temporal_projection(t_emb).shape}")
        x += self.temporal_projection(t_emb)
        x = self.upconv2(x)
        print(f"After upconv2, x.shape = {x.shape}")
        x = self.upsampling(x)
        print(f"After upsampling, x.shape = {x.shape}")
        x = self.norm2(x)
        print(f"Adding _x of shape {_x.shape} to x of shape {x.shape}")
        print(f"Rescaled _x is {self.residual_rescaling(_x).shape}")
        return self.out_relu(x + self.residual_rescaling(_x))


class UNetBackboneModel(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int],
        time_encoder: torch.nn.Module,
        temporal_channels: int,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.time_encoder = time_encoder
        self.temporal_network = torch.nn.Sequential(
            torch.nn.Linear(temporal_channels, temporal_channels),
            torch.nn.BatchNorm1d(temporal_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(temporal_channels, temporal_channels),
            torch.nn.BatchNorm1d(temporal_channels),
            torch.nn.Tanh(),
        )
        assert input_shape[0] == input_shape[1]
        in_channels = input_shape[-1]

        # TODO: Global self-attention bewtween some blocks (instead of residualblock?)
        self.identity1 = IdentityResidualBlock(
            in_channels, 128, temporal_channels, temporal_projection_stride=1
        )
        self.identity2 = IdentityResidualBlock(
            128, 128, temporal_channels, temporal_projection_stride=1
        )
        self.down1 = DownScaleResidualBlock(
            128,
            128,
            temporal_channels,
            temporal_projection_stride=2,
            pooling=False,
        )
        self.down2 = DownScaleResidualBlock(
            128, 256, temporal_channels, temporal_projection_stride=4, pooling=False
        )
        self.down3 = DownScaleResidualBlock(
            256, 512, temporal_channels, temporal_projection_stride=8, pooling=False
        )
        self.down4 = DownScaleResidualBlock(
            512, 512, temporal_channels, temporal_projection_stride=16, pooling=False
        )
        self.tunnel1 = IdentityResidualBlock(
            512, 512, temporal_channels, temporal_projection_stride=16
        )  # This is the middle 'bottleneck'
        self.tunnel2 = IdentityResidualBlock(
            512, 512, temporal_channels, temporal_projection_stride=16
        )
        self.up1 = UpScaleResidualBlock(
            512 + 512,
            256,
            temporal_channels,
            temporal_projection_stride=16,
            upsampling=False,
            output_padding=1,
        )
        self.up2 = UpScaleResidualBlock(
            256 + 512,
            128,
            temporal_channels,
            temporal_projection_stride=8,
            upsampling=False,
            output_padding=0,
        )
        self.up3 = UpScaleResidualBlock(
            256 + 128,
            64,
            temporal_channels,
            temporal_projection_stride=4,
            upsampling=False,
            output_padding=1,
        )
        self.identity3 = IdentityResidualBlock(
            64 + 128, 64, temporal_channels, temporal_projection_stride=2
        )
        self.identity4 = IdentityResidualBlock(
            64 + 256, 64, temporal_channels, temporal_projection_stride=2
        )
        self.out_conv = torch.nn.Conv2d(64, 1, 1, padding=0, stride=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        t = self.temporal_network(self.time_encoder(t))
        # t = self.time_encoder(t)
        # print(f"t.shape = {t.shape}")
        t = t.unsqueeze(-1).unsqueeze(-1)
        t = t.repeat(
            1, 1, self.input_shape[0], self.input_shape[1]
        )  # TODO: Is this how it should be done???
        # print(f"New t.shape = {t.shape}")
        x1 = self.identity1(x, t)
        x2 = self.identity2(x1, t)
        x3 = self.down1(x2, t)
        x4 = self.down2(x3, t)
        x5 = self.down3(x4, t)
        x6 = self.down4(x5, t)
        x7 = self.tunnel1(x6, t)
        x8 = torch.cat((self.tunnel2(x7, t), x6), dim=1)
        x9 = torch.cat((self.up1(x8, t), x5), dim=1)
        x10 = torch.cat((self.up2(x9, t), x4), dim=1)
        x11 = torch.cat((self.up3(x10, t), x3), dim=1)
        x12 = torch.cat((self.identity3(x11, t), x2), dim=1)
        x13 = torch.cat((self.identity4(x12, t), x), dim=1)
        x14 = self.out_conv(x13)
        print(x14.shape)
        return x14.view(input_shape)
