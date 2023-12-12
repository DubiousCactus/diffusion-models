#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Diffusion model and backbones.
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


class DiffusionModel(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        input_shape: Tuple[int],
        time_steps: int,
        beta_1: float,
        beta_T: float,
        temporal_channels: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.time_steps = time_steps
        self.beta = torch.nn.Parameter(
            torch.linspace(beta_1, beta_T, time_steps), requires_grad=False
        )
        self.alpha = torch.nn.Parameter(
            torch.exp(
                torch.tril(torch.ones((time_steps, time_steps)))
                @ torch.log(1 - self.beta)
            ),
            requires_grad=False,
        )
        self.sigma = torch.nn.Parameter(torch.sqrt(self.beta), requires_grad=False)
        assert not torch.isnan(self.alpha).any(), "Alphas contains nan"
        assert not (self.alpha < 0).any(), "Alphas contain neg"
        self._input_shape = None

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._input_shape is None:
            self._input_shape = x.shape[1:]
        # ===== Training =========
        # 1. Sample timestep t with shape (B, 1)
        t = (
            torch.randint(0, self.time_steps, (x.shape[0], 1))
            .to(x.device)
            .requires_grad_(False)
        )
        # 2. Sample the noise with shape x.shape
        eps = torch.randn_like(x).to(x.device).requires_grad_(False)
        # 3. Diffuse the image
        batched_alpha = self.alpha[t]

        # assert not torch.isnan(batched_alpha).any(), "Bartched Alphas contains nan"
        # assert not (self.alpha < 0).any(), "Alpha contains neg"
        # assert not (batched_alpha < 0).any(), "Bartched Alphas contains neg"
        # assert not torch.isnan(
        # torch.sqrt(1 - batched_alpha)
        # ).any(), "Sqrt 1-alpha is nan"

        # assert not torch.isnan(torch.sqrt(batched_alpha)).any(), "Sqrt alpha is nan"
        # assert not torch.isnan(
        # torch.sqrt(1 - batched_alpha)
        # ).any(), "Sqrt 1-alpha is nan"
        diffused_x = (
            torch.sqrt(batched_alpha)[..., None, None] * x
            + torch.sqrt(1 - batched_alpha)[..., None, None] * eps
        )
        # assert not torch.isnan(diffused_x).any(), "Diffused x contains nan"
        # 4. Predict the noise sample
        eps_hat = self.backbone(diffused_x, t)
        return eps_hat, eps

    def generate(self, n: int) -> torch.Tensor:
        assert self._input_shape is not None, "Must call forward first"
        with torch.no_grad():
            device = next(self.parameters()).device
            z_current = torch.randn(n, *self._input_shape).to(device)
            for t in range(self.time_steps - 1, 0, -1):  # Reversed from T to 1
                eps_hat = self.backbone(
                    z_current, torch.tensor(t).view(1, 1).repeat(n, 1).to(device)
                )
                z_prev_hat = (1 / (torch.sqrt(1 - self.beta[t]))) * z_current - (
                    self.beta[t]
                    / (torch.sqrt(1 - self.alpha[t]) * torch.sqrt(1 - self.beta[t]))
                ) * eps_hat
                eps = torch.randn_like(z_current)
                z_current = z_prev_hat + eps * self.sigma[t]
            # Now for z_0:
            eps_hat = self.backbone(
                z_current, torch.tensor(0).view(1, 1).repeat(n, 1).to(device)
            )
            x_hat = (1 / (torch.sqrt(1 - self.beta[0]))) * z_current - (
                self.beta[0]
                / (torch.sqrt(1 - self.alpha[0]) * torch.sqrt(1 - self.beta[0]))
            ) * eps_hat
            return x_hat.view(n, *self._input_shape)
