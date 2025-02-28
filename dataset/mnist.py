#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
MNIST dataset.
"""


from typing import Optional, Tuple, Union

import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from dataset.base.image import ImageDataset


class MNISTDataset(ImageDataset):
    IMG_SIZE = (28, 28, 1)

    def __init__(
        self,
        dataset_name: str,
        dataset_root: str,
        split: str,
        img_dim: Optional[Tuple[int]] = None,
        augment=False,
        normalize=False,
        debug=False,
        tiny=False,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            dataset_name,
            dataset_root,
            split,
            img_dim,
            augment,
            normalize,
            debug,
            tiny,
            seed=seed,
        )
        self._normalization = transforms.Normalize(0.1307, 0.3081)
        self.denormalize = (
            transforms.Compose(
                [
                    transforms.Normalize(0, 1 / 0.3081),
                    transforms.Normalize(-0.1307, 1),
                ]
            )
            if normalize
            else lambda x: x
        )

    def _load(
        self,
        dataset_root: str,
        tiny: bool,
        split: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Union[dict, list, torch.Tensor], Union[dict, list, torch.Tensor]]:
        if split == "test":
            dataset = MNIST(root=dataset_root, train=False, download=True)
            samples, labels = dataset.data, dataset.targets
        else:
            dataset = MNIST(
                root=dataset_root, train=split in ["train", "val"], download=True
            )
            data_len = len(dataset)
            train_len = int(0.7 * data_len)
            samples, labels = dataset.data, dataset.targets
            indices = np.arange(data_len)
            np.random.seed(42)  # Avoid data leakage
            np.random.shuffle(indices)
            samples, labels = samples[indices], labels[indices]
            samples = samples[:train_len] if split == "train" else samples[train_len:]
            labels = labels[:train_len] if split == "train" else labels[train_len:]
        return samples, labels

    def __getitem__(self, index: int):
        sample, label = self._samples[index], self._labels[index]
        sample = (sample / 255.0).float().unsqueeze(0)
        if self._augment:
            sample = self._augs(image=sample)["image"]
        if self._normalize:
            sample = self._normalization(sample)
        return sample, label
