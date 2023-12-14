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
        time_dim: int,
    ):
        super().__init__()
        flat_shape = reduce(lambda x, y: x * y, input_shape)
        self.time_encoder = time_encoder
        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(time_dim, time_dim),
            torch.nn.BatchNorm1d(time_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(time_dim, time_dim),
            torch.nn.BatchNorm1d(time_dim),
            torch.nn.Tanh(),
        )
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(flat_shape, 512),
                torch.nn.Linear(512, 256),
                torch.nn.Linear(256, 128),
                torch.nn.Linear(128, 64),
                torch.nn.Linear(64, 128),
                torch.nn.Linear(128, 256),
                torch.nn.Linear(256, 512),
                torch.nn.Linear(512, flat_shape),
            ]
        )
        self.bn = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(512),
                torch.nn.BatchNorm1d(256),
                torch.nn.BatchNorm1d(128),
                torch.nn.BatchNorm1d(64),
                torch.nn.BatchNorm1d(128),
                torch.nn.BatchNorm1d(256),
                torch.nn.BatchNorm1d(512),
            ]
        )
        # self.bn = torch.nn.ModuleList([torch.nn.GroupNorm(32, 2048)] * 5)
        self.input_time_proj = torch.nn.Linear(time_dim, flat_shape)
        self.time_proj = torch.nn.ModuleList(
            [
                torch.nn.Linear(time_dim, 512, bias=False),
                torch.nn.Linear(time_dim, 256, bias=False),
                torch.nn.Linear(time_dim, 128, bias=False),
                torch.nn.Linear(time_dim, 64, bias=False),
                torch.nn.Linear(time_dim, 128, bias=False),
                torch.nn.Linear(time_dim, 256, bias=False),
                torch.nn.Linear(time_dim, 512, bias=False),
            ]
        )
        # 3 Skip connections, 1 every 2 layers:
        self.residual_proj = torch.nn.ModuleList(
            [
                torch.nn.Linear(flat_shape, 128, bias=False),
                torch.nn.Identity(),
                torch.nn.Linear(128, 512, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(x.shape[0], -1)
        _x = x
        # time_embed = self.time_mlp(self.time_encoder(t))
        time_embed = self.time_encoder(t)
        # x += self.input_time_proj(time_embed)
        for i, (layer, bn, time_proj) in enumerate(
            zip(self.layers, self.bn, self.time_proj)
        ):
            x = layer(x) + time_proj(time_embed)
            x = bn(x)
            if i > 0 and i % 2 == 0:
                x = torch.nn.functional.leaky_relu(
                    x + self.residual_proj[i // 2 - 1](_x)
                )
                _x = x
            else:
                x = torch.nn.functional.leaky_relu(x)
        # x = torch.tanh(self.layers[-1](x))
        x = self.layers[-1](x)
        return x.view(input_shape)
