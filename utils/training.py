#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training utilities. This is a good place for your code that is used in training (i.e. custom loss
function, visualization code, etc.)
"""

from typing import Callable, List, Tuple, Union

import matplotlib.pyplot as plt
import torch

import conf.project as project_conf
from utils import to_cuda_


def visualize_model_generations(
    model: torch.nn.Module,
    batch: Union[Tuple, List, torch.Tensor],
    step: int,
    denormalize: Callable,
) -> None:
    x, y = to_cuda_(batch)  # type: ignore
    if not project_conf.HEADLESS:
        x_hat = model.generate(10)
        x_hat = denormalize(x_hat)
        # Clip to [0, 1]: (MatPlotLib does this automatically though)
        x_hat = torch.clamp(x_hat, 0, 1)
        fig, axs = plt.subplots(1, 10)
        for i in range(10):
            axs[i].imshow(x_hat[i].swapaxes(0, 2).swapaxes(0, 1).cpu().numpy())
            axs[i].axis("off")
        plt.show()

    if project_conf.USE_WANDB:
        # TODO: Log a few predictions and the ground truth to wandb.
        # wandb.log({"pointcloud": wandb.Object3D(ptcld)}, step=step)
        # raise NotImplementedError("Visualization is not implemented for wandb.")
        pass


def visualize_model_reconstructions(
    model: torch.nn.Module,
    batch: Union[Tuple, List, torch.Tensor],
    step: int,
    denormalize: Callable,
) -> None:
    x, _ = to_cuda_(batch)  # type: ignore
    if not project_conf.HEADLESS:
        x_hat, latent_map = model(x)
        x = denormalize(x)
        x_hat = denormalize(x_hat)
        # Clip to [0, 1]: (MatPlotLib does this automatically though)
        x_hat = torch.clamp(x_hat, 0, 1)
        fig, axs = plt.subplots(2, 10)
        print(
            f"Latent map shape: {latent_map.shape}. Downsampling factor is {x.shape[-1] / latent_map.shape[-1]}"
        )
        for i in range(10):
            axs[0, i].imshow(x[i].swapaxes(0, 2).swapaxes(0, 1).cpu().numpy())
            axs[0, i].axis("off")
            axs[1, i].imshow(x_hat[i].swapaxes(0, 2).swapaxes(0, 1).cpu().numpy())
            axs[1, i].axis("off")
            # axs[2,i].imshow(latent_map[i].swapaxes(0, 2).swapaxes(0, 1).cpu().numpy())
            # axs[2,i].axis("off")
        plt.show()

    if project_conf.USE_WANDB:
        # TODO: Log a few predictions and the ground truth to wandb.
        # wandb.log({"pointcloud": wandb.Object3D(ptcld)}, step=step)
        # raise NotImplementedError("Visualization is not implemented for wandb.")
        pass
