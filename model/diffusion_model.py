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


class SinusoidalTimeEncoder(torch.nn.Module):
    def __init__(self, time_steps: int, input_shape: Tuple[int]) -> None:
        super().__init__()
        model_dim = reduce(lambda x, y: x * y, input_shape)
        self.time_steps = time_steps
        # Little trick to simplify computation:
        constants = torch.exp(
            -torch.arange(0, model_dim, 2)
            * (torch.log(torch.tensor(10000.0)) / model_dim)
        )
        # assert torch.allclose(
        # constants,
        # torch.tensor(
        # [10000**((2*i)/model_dim) for i in range(time_steps//2)]
        # ),
        # ), "Oops we computed it wrong!"
        self.time_embeddings = torch.nn.Parameter(
            torch.zeros(time_steps, model_dim), requires_grad=False
        )
        self.time_embeddings[:, ::2] = torch.sin(
            torch.arange(0, time_steps).unsqueeze(1).repeat(1, model_dim // 2)
            * constants
        )
        self.time_embeddings[:, 1::2] = torch.cos(
            torch.arange(0, time_steps).unsqueeze(1).repeat(1, model_dim // 2)
            * constants
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        assert type(t) == torch.Tensor
        assert x.shape[0] == t.shape[0], "Batch size must be the same for x and t"
        assert len(t.shape) == 2, "t must be (B, 1)"
        return x + self.time_embeddings[t].squeeze()


class MLPBackboneModel(torch.nn.Module):
    def __init__(
        self, input_shape: Tuple[int], latent_dim: int, time_encoder: torch.nn.Module
    ):
        super().__init__()
        flat_shape = reduce(lambda x, y: x * y, input_shape)
        self.time_encoder = time_encoder
        self.predictor = torch.nn.Sequential(
            torch.nn.Linear(flat_shape, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, flat_shape),
            torch.nn.Tanh(),  # We don't need to predict noise samples beyond [-1, 1]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(x.shape[0], -1)
        x = self.time_encoder(x, t)
        x = self.predictor(x)
        return x.view(input_shape)


class MLPUNetBackboneModel(torch.nn.Module):
    def __init__(
        self, input_shape: Tuple[int], latent_dim: int, time_encoder: torch.nn.Module
    ):
        super().__init__()
        flat_shape = reduce(lambda x, y: x * y, input_shape)
        self.time_encoder = time_encoder
        self.encoder = torch.nn.ModuleList(
            [
                torch.nn.Linear(flat_shape, 512),
                torch.nn.Linear(512, 256),
                torch.nn.Linear(256, latent_dim),
            ]
        )
        self.encoder_bn = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(512),
                torch.nn.BatchNorm1d(256),
            ]
        )
        self.decoder = torch.nn.ModuleList(
            [
                torch.nn.Linear(latent_dim, 256),
                torch.nn.Linear(256, 512),
                torch.nn.Linear(512, flat_shape),
            ]
        )
        self.decoder_bn = torch.nn.ModuleList(
            [
                torch.nn.BatchNorm1d(256),
                torch.nn.BatchNorm1d(512),
            ]
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape
        x = x.view(x.shape[0], -1)
        x = self.time_encoder(x, t)
        for i, (layer, bn) in enumerate(zip(self.encoder, self.encoder_bn + [None])):
            if i < (len(self.encoder) - 1):
                pass  # TODO: Skip connection
            x = layer(x)
            if i < (len(self.encoder) - 1):
                x = bn(x)
                x = torch.relu(x)

        for i, (layer, bn) in enumerate(zip(self.decoder, self.decoder_bn + [None])):
            if i < (len(self.encoder) - 1):
                pass  # TODO: Skip connection
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
        self.backbone = MLPBackboneModel(
            input_shape, 128, SinusoidalTimeEncoder(timesteps, input_shape)
        )
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
        t = (
            torch.randint(0, self.timesteps, (x.shape[0], 1))
            .to(x.device)
            .requires_grad_(False)
        )
        # 2. Sample the noise with shape x.shape
        eps = torch.randn_like(x).to(x.device).requires_grad_(False)
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
        assert self._input_shape is not None, "Must call forward first"
        with torch.no_grad():
            device = next(self.parameters()).device
            z_current = torch.randn(n, *self._input_shape).to(device)
            for t in range(self.timesteps - 1, 0, -1):  # Reversed from T to 1
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
