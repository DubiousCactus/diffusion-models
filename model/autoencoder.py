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
            IdentityResidualBlock(self._input_channels, 64),
            DownScaleResidualBlock(64, 128),
            DownScaleResidualBlock(128, 256),
            # DownScaleResidualBlock(256, 512),
            # IdentityResidualBlock(512, 512),
            IdentityResidualBlock(256, 256),
        )
        self.decoder = torch.nn.Sequential(
            # UpScaleResidualBlock(512, 256, output_padding=0),
            UpScaleResidualBlock(256, 128, output_padding=1),
            UpScaleResidualBlock(128, 64, output_padding=1),
            IdentityResidualBlock(64, 64),
            torch.nn.Conv2d(64, self._input_channels, 1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_map = self.encoder(x)
        return self.decoder(latent_map), latent_map
