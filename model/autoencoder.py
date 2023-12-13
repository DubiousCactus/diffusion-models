#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Simple autoencoders for the LDM.
"""


from typing import Tuple

import torch

from model.resnet import (
    DownScaleResidualBlock,
    IdentityResidualBlock,
    UpScaleResidualBlock,
)


class ImageAutoEncoderModel(torch.nn.Module):
    def __init__(self, input_shape: Tuple[int]) -> None:
        super().__init__()
        self._input_shape = input_shape
        self._input_channels = input_shape[-1]
        self.encoder = torch.nn.Sequential(
            IdentityResidualBlock(self._input_channels, 32, kernels=(3, 3, 1)),
            DownScaleResidualBlock(32, 64, kernels=(3, 3, 1)),
            DownScaleResidualBlock(64, 128, kernels=(3, 3, 1)),
            DownScaleResidualBlock(128, 256, kernels=(3, 3, 1)),
            DownScaleResidualBlock(256, 512, kernels=(3, 3, 1)),
            IdentityResidualBlock(512, 512, kernels=(3, 3, 1)),
        )
        self.decoder = torch.nn.Sequential(
            UpScaleResidualBlock(512, 256, kernels=(3, 3, 1)),
            UpScaleResidualBlock(256, 128, kernels=(3, 3, 1)),
            UpScaleResidualBlock(128, 64, kernels=(3, 3, 1)),
            UpScaleResidualBlock(64, 32, kernels=(3, 3, 1)),
            IdentityResidualBlock(32, 32, kernels=(3, 3, 1)),
            torch.nn.Conv2d(32, self._input_channels, 1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_map = self.encoder(x)
        return self.decoder(latent_map), latent_map
