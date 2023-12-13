#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Autoencoder trainer.
"""

from typing import Dict, List, Tuple, Union

import torch

from src.base_trainer import BaseTrainer
from utils import to_cuda
from utils.training import visualize_model_reconstructions


class AETrainer(BaseTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @to_cuda
    def _visualize(
        self,
        batch: Union[Tuple, List, torch.Tensor],
        epoch: int,
    ) -> None:
        """Visualize the model predictions.
        Args:
            batch: The batch to process.
            epoch: The current epoch.
        """
        visualize_model_reconstructions(
            self._model, batch, epoch, self._val_loader.dataset.denormalize
        )  # User implementation goes here (utils/training.py)

    @to_cuda
    def _train_val_iteration(
        self,
        batch: Union[Tuple, List, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Training or validation procedure for one batch. We want to keep the code DRY and avoid
        making mistakes, so write this code only once at the cost of many function calls!
        Args:
            batch: The batch to process.
        Returns:
            torch.Tensor: The loss for the batch.
            Dict[str, torch.Tensor]: The loss components for the batch.
        """
        x, _ = batch
        x_hat, latent_map = self._model(x)
        loss, loss_components = torch.nn.functional.mse_loss(x_hat, x), {}
        return loss, loss_components
