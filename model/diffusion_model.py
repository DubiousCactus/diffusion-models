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
        self, input_shape: Tuple[int], latent_dim: int, time_dim: int, time_steps: int
    ):
        super().__init__()
        flat_shape = reduce(lambda x, y: x * y, input_shape)
        self.time_steps = time_steps
        self.time_embedder = torch.nn.Sequential(
            torch.nn.Linear(1, time_dim // 2),
            torch.nn.BatchNorm1d(time_dim // 2),
            torch.nn.Sigmoid(),
            torch.nn.Linear(time_dim // 2, time_dim),
        )
        self.encoder = torch.nn.ModuleList(
            [
                torch.nn.Linear(flat_shape + time_dim, 1024),
                torch.nn.Linear(1024 + time_dim, 512),
                torch.nn.Linear(512 + time_dim, 256),
                torch.nn.Linear(256, latent_dim),
            ]
        )
        self.encoder_bn = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(1024),
                torch.nn.BatchNorm1d(512),
                torch.nn.BatchNorm1d(256),
            ]
        )
        self.decoder = torch.nn.ModuleList(
            [
                torch.nn.Linear(latent_dim + time_dim, 256),
                torch.nn.Linear(256 + time_dim, 512),
                torch.nn.Linear(512 + time_dim, 1024),
                torch.nn.Linear(1024, flat_shape),
            ]
        )
        self.decoder_bn = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(256),
                torch.nn.BatchNorm1d(512),
                torch.nn.BatchNorm1d(1024),
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        time_embedding = self.time_embedder(t.float() / self.time_steps)
        x = x.view(x.shape[0], -1)
        for i, (layer, bn) in enumerate(zip(self.encoder, self.encoder_bn + [None])):
            if i < (len(self.encoder) - 1):
                x = torch.cat((x, time_embedding), dim=-1)
            x = layer(x)
            if i < (len(self.encoder) - 1):
                x = bn(x)
                x = torch.relu(x)

        for i, (layer, bn) in enumerate(zip(self.decoder, self.decoder_bn + [None])):
            if i < (len(self.encoder) - 1):
                x = torch.cat((x, time_embedding), dim=-1)
            x = layer(x)
            if i < (len(self.encoder) - 1):
                x = bn(x)
                x = torch.relu(x)

        return x.view(input_shape)


class DiffusionModel(torch.nn.Module):
    def __init__(
        self,
        # backbone: torch.nn.Module,
        input_shape: Tuple[int],
        timesteps: int,
        beta_1: float,
        beta_T: float,
    ):
        super().__init__()
        self.backbone = MLPBackboneModel(input_shape, 128, 32, timesteps)
        self.timesteps = timesteps
        self.beta = torch.nn.Parameter(
            torch.linspace(beta_1, beta_T, timesteps), requires_grad=False
        )
        self.alpha = torch.nn.Parameter(
            torch.exp(
                torch.tril(torch.ones((timesteps, timesteps)))
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
        t = torch.randint(1, self.timesteps, (x.shape[0], 1)).to(x.device)
        # 2. Sample the noise with shape x.shape
        eps = torch.randn_like(x).to(x.device)
        # 3. Diffuse the image
        batched_alpha = self.alpha[t]

        assert not torch.isnan(batched_alpha).any(), "Bartched Alphas contains nan"
        assert not (self.alpha < 0).any(), "Alpha contains neg"
        assert not (batched_alpha < 0).any(), "Bartched Alphas contains neg"
        assert not torch.isnan(
            torch.sqrt(1 - batched_alpha)
        ).any(), "Sqrt 1-alpha is nan"

        assert not torch.isnan(torch.sqrt(batched_alpha)).any(), "Sqrt alpha is nan"
        assert not torch.isnan(
            torch.sqrt(1 - batched_alpha)
        ).any(), "Sqrt 1-alpha is nan"
        diffused_x = (
            torch.sqrt(batched_alpha)[..., None, None] * x
            + torch.sqrt(1 - batched_alpha)[..., None, None] * eps
        )
        assert not torch.isnan(diffused_x).any(), "Diffused x contains nan"
        # 4. Predict the noise sample
        eps_hat = self.backbone(diffused_x, t)
        return eps_hat, eps

    def generate(self, n: int) -> torch.Tensor:
        device = next(self.parameters()).device
        z_current = torch.randn(n, *self._input_shape).to(device)
        for t in range(self.timesteps - 1, 0, -1):  # Reversed from T to 1
            eps_hat = self.backbone(
                z_current, torch.tensor(t).view(1, 1).repeat(n, 1).to(device)
            )
            z_prev_hat = (
                1 / (torch.sqrt(1 - self.beta[t])) * z_current
                - self.beta[t]
                / (torch.sqrt(1 - self.alpha[t] * torch.sqrt(1 - self.beta[t])))
                * eps_hat
            )
            eps = torch.randn_like(z_current)
            z_current = z_prev_hat + eps * self.sigma[t]
        # Now for z_0:
        x_hat = (
            1 / (torch.sqrt(1 - self.beta[t])) * z_current
            - self.beta[t]
            / (torch.sqrt(1 - self.alpha[t] * torch.sqrt(1 - self.beta[t])))
            * eps_hat
        )
        return x_hat.view(n, *self._input_shape)
