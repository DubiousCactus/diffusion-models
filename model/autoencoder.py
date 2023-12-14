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
    def __init__(
        self, input_shape: Tuple[int], output_paddings: Tuple[int] = (1, 1, 1)
    ) -> None:
        super().__init__()
        self._input_shape = input_shape
        self._input_channels = input_shape[-1]
        self.encoder = torch.nn.Sequential(
            IdentityResidualBlock(self._input_channels, 8),
            DownScaleResidualBlock(8, 16),
            DownScaleResidualBlock(16, 32),
            DownScaleResidualBlock(32, 32),
            IdentityResidualBlock(32, 32),
            torch.nn.Conv2d(32, 8, 1),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Conv2d(8, 32, 1),
            UpScaleResidualBlock(32, 32, output_padding=output_paddings[0]),
            UpScaleResidualBlock(32, 16, output_padding=output_paddings[1]),
            UpScaleResidualBlock(16, 8, output_padding=output_paddings[2]),
            IdentityResidualBlock(8, 8),
            torch.nn.Conv2d(8, self._input_channels, 1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_map = self.encoder(x)
        return self.decoder(latent_map), latent_map
