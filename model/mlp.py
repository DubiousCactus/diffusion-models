#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
MLP UNet and variants for the PDM.
"""

from functools import reduce
from typing import Tuple

import torch

from model.resnet import TemporalResidualBlock1D


class MLPBackboneModel(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int],
        time_encoder: torch.nn.Module,
        time_dim: int,
    ):
        super().__init__()
        flat_shape = reduce(lambda x, y: x * y, input_shape)
        self.time_encoder = time_encoder
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_dim, time_dim),
            torch.nn.BatchNorm1d(time_dim),
            # torch.nn.GroupNorm(32, time_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(time_dim, time_dim),
            torch.nn.BatchNorm1d(time_dim),
            # torch.nn.GroupNorm(32, time_dim),
            torch.nn.Tanh(),
        )
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(flat_shape, 2048)]
            + [torch.nn.Linear(2048, 2048)] * 4
            + [torch.nn.Linear(2048, flat_shape)]
        )
        # self.bn = torch.nn.ModuleList([torch.nn.BatchNorm1d(2048)] * 3)
        self.bn = torch.nn.ModuleList([torch.nn.GroupNorm(32, 2048)] * 5)
        self.input_time_proj = torch.nn.Linear(time_dim, flat_shape)
        self.time_proj = torch.nn.ModuleList([torch.nn.Linear(time_dim, 2048)] * 5)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(x.shape[0], -1)
        # time_embed = self.time_mlp(self.time_encoder(t))
        time_embed = self.time_encoder(t)
        # x += self.input_time_proj(time_embed)
        for i, (layer, bn, time_proj) in enumerate(
            zip(self.layers, self.bn, self.time_proj)
        ):
            _x = x
            x = layer(x) + time_proj(time_embed)
            x = bn(x)
            if i > 0 and i % 2 == 0:
                x += _x  # Residual connection
            x = torch.nn.functional.leaky_relu(x)
        x = torch.tanh(self.layers[-1](x))
        return x.view(input_shape)


class MLPUNetBackboneModel(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int],
        time_encoder: torch.nn.Module,
        temporal_channels: int,
    ):
        super().__init__()
        flat_shape = reduce(lambda x, y: x * y, input_shape)
        self.time_encoder = time_encoder
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(temporal_channels, temporal_channels),
            torch.nn.BatchNorm1d(temporal_channels),
            torch.nn.GELU(),
            torch.nn.Linear(temporal_channels, temporal_channels),
        )
        self.input_layer = torch.nn.Linear(flat_shape, 512)
        self.identity_1 = TemporalResidualBlock1D(512, 512, temporal_channels)
        self.down_1 = TemporalResidualBlock1D(512, 256, temporal_channels)
        self.down_2 = TemporalResidualBlock1D(256, 128, temporal_channels)
        self.down_3 = TemporalResidualBlock1D(128, 64, temporal_channels)
        self.tunnel_1 = TemporalResidualBlock1D(64, 64, temporal_channels)
        self.tunnel_2 = TemporalResidualBlock1D(64, 64, temporal_channels)
        self.up_1 = TemporalResidualBlock1D(64 + 64, 128, temporal_channels)
        self.up_2 = TemporalResidualBlock1D(128 + 128, 256, temporal_channels)
        self.up_3 = TemporalResidualBlock1D(256 + 256, 512, temporal_channels)
        self.identity_2 = TemporalResidualBlock1D(512, 512, temporal_channels)
        self.output_layer = torch.nn.Linear(512, flat_shape)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(x.shape[0], -1)
        time_embed = self.time_mlp(self.time_encoder(t))
        x1 = self.input_layer(x)
        x2 = self.identity_1(x1, time_embed)
        x3 = self.down_1(x2, time_embed)
        x4 = self.down_2(x3, time_embed)
        x5 = self.down_3(x4, time_embed)
        x6 = self.tunnel_1(x5, time_embed)
        x7 = self.tunnel_2(x6, time_embed)
        x8 = self.up_1(torch.cat([x7, x5], dim=1), time_embed)
        x9 = self.up_2(torch.cat([x8, x4], dim=1), time_embed)
        x10 = self.up_3(torch.cat([x9, x3], dim=1), time_embed)
        x11 = self.identity_2(x10, time_embed)
        return self.output_layer(x11).view(input_shape)
