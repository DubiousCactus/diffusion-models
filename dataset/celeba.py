#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
CelebA dataset.
"""


from typing import Optional, Tuple, Union

import torch
from torchvision.datasets import CelebA
from torchvision.transforms import transforms

from dataset.base.image import ImageDataset


class CelebADataset(ImageDataset):
    IMG_SIZE = (64, 64, 3)

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
        split = "valid" if split == "val" else split
        self._dataset = CelebA(
            root=dataset_root,
            split=split,
            download=False,
            target_type="identity",
            transform=transforms.Compose(
                [
                    transforms.Resize(
                        self.IMG_SIZE[:2] if img_dim is None else img_dim[:2]
                    ),
                    transforms.CenterCrop(
                        self.IMG_SIZE[:2] if img_dim is None else img_dim[:2]
                    ),
                    transforms.ToTensor(),
                ]
            ),
        )
        self._normalization = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.denormalize = (
            transforms.Compose(
                [
                    transforms.Normalize((0, 0, 0), (1 / 0.5, 1 / 0.5, 1 / 0.5)),
                    transforms.Normalize((-0.5, -0.5, -0.5), (1, 1, 1)),
                ]
            )
            if normalize
            else lambda x: x
        )

    def _load(
        self, dataset_root: str, tiny: bool, split: str, seed: int
    ) -> Tuple[Union[dict, list, torch.Tensor], Union[dict, list, torch.Tensor]]:
        return [], []

    def __getitem__(self, index: int):
        sample, label = self._dataset[index]
        if self._augment:
            sample = self._augs(image=sample)["image"]
        if self._normalize:
            sample = self._normalization(sample)
        return sample, label

    def __len__(self):
        return len(self._dataset)
