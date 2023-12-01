#! /usr/bin/env python
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Some utility classes and functions for your convenience.
"""

import os
import os.path as osp
import pickle
from collections.abc import Callable
from typing import Any, Optional, Union

import blosc
from hydra.core.hydra_config import HydraConfig


class BestNModelSaver:
    def __init__(self, n: int, save_callback: Callable) -> None:
        self._n = n
        self._best_n_models = {}
        self._save_callback = save_callback
        self._min_val_loss = float("inf")
        self._min_val_loss_epoch: int = 0
        self._best_metrics = {}
        self._min_val_metric = float("inf")

    def __call__(
        self,
        epoch: int,
        val_loss: float,
        metrics: Optional[dict] = None,
        minimize_metric: str = "loss",
    ) -> Any:
        if (val_loss < self._min_val_loss) or (
            metrics.get(minimize_metric, val_loss) < self._min_val_metric
        ):
            ckpt_path = osp.join(
                HydraConfig.get().runtime.output_dir,
                f"epoch_{epoch:03d}_val-loss_{val_loss:06f}.ckpt",
            )
            self._save_if_best_model(
                val_loss, ckpt_path
            ) if self._n > 0 else self._save_callback(val_loss, ckpt_path)
            self._min_val_loss = val_loss
            self._min_val_metric = metrics.get(minimize_metric, val_loss)
            self._min_val_loss_epoch = epoch
            self._best_metrics = metrics

    def _save_if_best_model(self, val_loss: float, ckpt_path: str):
        if len(self._best_n_models) < self._n or val_loss < max(
            self._best_n_models.keys()
        ):
            if len(self._best_n_models) == self._n:
                worst_of_best = max(self._best_n_models.keys())
                del_fname = self._best_n_models[worst_of_best]
                os.remove(del_fname)
                del self._best_n_models[worst_of_best]
            last_ckpt_path = osp.join(HydraConfig.get().runtime.output_dir, "last.ckpt")
            if osp.isfile(last_ckpt_path):
                os.remove(last_ckpt_path)
            self._best_n_models[val_loss] = ckpt_path
            self._save_callback(val_loss, ckpt_path)

    @property
    def min_val_loss(self):
        return self._min_val_loss

    @property
    def min_val_loss_epoch(self):
        return self._min_val_loss_epoch

    @property
    def best_metrics(self) -> dict:
        best = {"loss": self._min_val_loss}
        if self._best_metrics is not None:
            for k, v in self._best_metrics.items():
                best[k] = v
        return best


class BestMetricTracker:
    def __init__(self, comp: str) -> None:
        if comp not in ["max", "min"]:
            raise ValueError(f"Invalid comp: {comp}. Must be 'max' or 'min'.")
        self._comp = (lambda x, y: x > y) if comp == "max" else (lambda x, y: x < y)
        self._best_value = float("-inf") if comp == "max" else float("inf")
        self._best_at_epoch: int = 0
        self._last_value = float("-inf") if comp == "max" else float("inf")

    def update(self, epoch: int, metric: Union[float, int]) -> None:
        self._last_value = metric
        if self._comp(metric, self._best_value):
            self._best_value = metric
            self._best_at_epoch = epoch

    @property
    def value(self) -> Union[float, int]:
        return self._best_value

    @property
    def at_epoch(self) -> int:
        return self._best_at_epoch

    @property
    def last_value(self) -> float:
        return self._last_value


def compressed_read(file_path: str) -> Any:
    """
    Read a pickle file compressed with blosc and return its content.
    """
    with open(file_path, "rb") as f:
        return pickle.loads(blosc.decompress(f.read()))


def compressed_write(path: str, *args, **kwargs) -> None:
    """
    Write data compressed with blosc to a pickle file.
    """
    with open(path, "wb") as f:
        f.write(blosc.compress(pickle.dumps(*args, **kwargs)))
