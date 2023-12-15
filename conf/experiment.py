#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Configurations for the experiments and config groups, using hydra-zen.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from hydra.conf import HydraConf, JobConf, RunDir
from hydra_zen import (
    MISSING,
    ZenStore,
    builds,
    make_config,
    make_custom_builds_fn,
    store,
)
from torch.utils.data import DataLoader
from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, NAMES

from dataset.celeba import CelebADataset
from dataset.mnist import MNISTDataset
from launch_experiment import launch_experiment
from model.autoencoder import ImageAutoEncoderModel
from model.diffusion_model import DiffusionModel, LatentDiffusionModel
from model.mlp import MLPBackboneModel, MLPResNetBackboneModel, MLPUNetBackboneModel
from model.time_encoding import SinusoidalTimeEncoder
from model.unet import (
    UNetBackboneModelLarge,
    UNetBackboneModelMicro,
    UNetBackboneModelMini,
    UNetBackboneModelSmall,
)
from src.ae_trainer import AETrainer
from src.base_tester import BaseTester
from src.base_trainer import BaseTrainer

# Set hydra.job.chdir=True using store():
hydra_store = ZenStore(overwrite_ok=True)
hydra_store(HydraConf(job=JobConf(chdir=True)), name="config", group="hydra")
# We'll generate a unique name for the experiment and use it as the run name
hydra_store(
    HydraConf(
        run=RunDir(
            f"runs/{get_random_name(combo=[ADJECTIVES, NAMES], separator='-', style='lowercase')}"
        )
    ),
    name="config",
    group="hydra",
)
hydra_store.add_to_hydra_store()
pbuilds = make_custom_builds_fn(zen_partial=True, populate_full_signature=False)

" ================== Dataset ================== "


# Dataclasses are a great and simple way to define a base config group with default values.
@dataclass
class ImageDatasetConf:
    dataset_name: str = "image_dataset"
    dataset_root: str = "data/a"
    tiny: bool = False
    normalize: bool = True
    augment: bool = False
    debug: bool = False
    img_dim: Tuple[int] = MISSING


# Pre-set the group for store's dataset entries
dataset_store = store(group="dataset")
dataset_store(
    pbuilds(
        CelebADataset,
        builds_bases=(ImageDatasetConf,),
        dataset_name="celeba",
        dataset_root="data/celeba",
        img_dim=CelebADataset.IMG_SIZE,
    ),
    name="celeba",
)
dataset_store(
    pbuilds(
        MNISTDataset,
        builds_bases=(ImageDatasetConf,),
        dataset_name="mnist",
        dataset_root="data/mnist",
        img_dim=MNISTDataset.IMG_SIZE,
    ),
    name="mnist",
)

" ================== Dataloader & sampler ================== "


@dataclass
class SamplerConf:
    batch_size: int = 16
    drop_last: bool = True
    shuffle: bool = True


@dataclass
class DataloaderConf:
    batch_size: int = 16
    drop_last: bool = True
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


" ================== Model ================== "


@dataclass
class BackboneConf:
    input_shape: Tuple[int]
    temporal_channels: int
    output_paddings: Tuple[int]
    normalization: str = "group"


@dataclass
class TimeEncoderConf:
    time_steps: int
    model_dim: int


@dataclass
class DiffusionModelConf:
    backbone: torch.nn.Module
    time_steps: int = 1000
    beta_1: float = 1e-4
    beta_T: float = 0.02


# Pre-set the group for store's model entries
model_store = store(group="model")
backbone_store = store(group="backbone")
time_encoder_store = store(group="time_encoder")
autoencoder_store = store(group="autoencoder")

# Not that encoder_input_dim depend on dataset.img_dim, so we need to use a partial to set them in
# the launch_experiment function.
model_store(
    pbuilds(MLPBackboneModel, input_shape=MISSING),
    name="mlp_backend",
)

model_store(
    pbuilds(
        DiffusionModel,
        builds_bases=(DiffusionModelConf,),
        backbone=MISSING,
    ),
    name="diffusion_model",
)

model_store(
    pbuilds(
        LatentDiffusionModel,
        builds_bases=(DiffusionModelConf,),
        backbone=MISSING,
    ),
    name="latent_diffusion_model",
)

model_store(
    pbuilds(ImageAutoEncoderModel, input_shape=MISSING, populate_full_signature=True),
    name="img_autoencoder",
)  # Can be part of the model store because we need to train it as one.

autoencoder_store(
    pbuilds(ImageAutoEncoderModel, input_shape=MISSING, populate_full_signature=True),
    name="img_autoencoder",
)  # But is also part of the autoencoder store because we use it for the LDM.

backbone_store(
    pbuilds(
        UNetBackboneModelSmall,
        builds_bases=(BackboneConf,),
    ),
    name="unet_small",
)

backbone_store(
    pbuilds(
        UNetBackboneModelMini,
        builds_bases=(BackboneConf,),
    ),
    name="unet_mini",
)

backbone_store(
    pbuilds(
        UNetBackboneModelMicro,
        builds_bases=(BackboneConf,),
    ),
    name="unet_micro",
)

backbone_store(
    pbuilds(
        UNetBackboneModelLarge,
        builds_bases=(BackboneConf,),
    ),
    name="unet_large",
)

backbone_store(
    pbuilds(
        MLPUNetBackboneModel,
        input_shape=MISSING,
        time_encoder=MISSING,
        temporal_channels=256,
    ),
    name="mlp_unet",
)

backbone_store(
    pbuilds(
        MLPBackboneModel,
        input_shape=MISSING,
        time_encoder=MISSING,
        temporal_channels=256,
    ),
    name="mlp",
)

backbone_store(
    pbuilds(
        MLPResNetBackboneModel,
        input_shape=MISSING,
        time_encoder=MISSING,
        temporal_channels=256,
        hidden_dim=2048,
    ),
    name="mlp_resnet",
)


time_encoder_store(
    pbuilds(
        SinusoidalTimeEncoder,
        builds_bases=(TimeEncoderConf,),
    ),
    name="sinusoidal",
)

" ================== Optimizer ================== "


@dataclass
class Optimizer:
    lr: float = 1e-3
    weight_decay: float = 0.0


opt_store = store(group="optimizer")
opt_store(
    pbuilds(
        torch.optim.Adam,
        builds_bases=(Optimizer,),
    ),
    name="adam",
)
opt_store(
    pbuilds(
        torch.optim.SGD,
        builds_bases=(Optimizer,),
    ),
    name="sgd",
)

opt_store(
    pbuilds(
        torch.optim.SGD,
        builds_bases=(Optimizer,),
        momentum=0.9,
        nesterov=True,
    ),
    name="sgd-momentum",
)

" ================== Scheduler ================== "
sched_store = store(group="scheduler")
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.StepLR,
        step_size=100,
        gamma=0.5,
    ),
    name="step",
)
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        mode="min",
        factor=0.5,
        patience=10,
    ),
    name="plateau",
)
sched_store(
    pbuilds(
        torch.optim.lr_scheduler.CosineAnnealingLR,
    ),
    name="cosine",
)

" ================== Experiment ================== "


@dataclass
class RunConfig:
    epochs: int = 200
    seed: int = 42
    val_every: int = 1
    viz_every: int = 10
    viz_train_every: int = 0
    viz_num_samples: int = 5
    load_from: Optional[str] = None
    load_ae_from: Optional[str] = None
    training_mode: bool = True


run_store = store(group="run")
run_store(RunConfig, name="default")


trainer_store = store(group="trainer")
trainer_store(pbuilds(BaseTrainer, populate_full_signature=True), name="base")
trainer_store(pbuilds(AETrainer, populate_full_signature=True), name="ae_trainer")

tester_store = store(group="tester")
tester_store(pbuilds(BaseTester, populate_full_signature=True), name="base")

Experiment = builds(
    launch_experiment,
    populate_full_signature=True,
    hydra_defaults=[
        "_self_",
        {"trainer": "base"},
        {"tester": "base"},
        {"dataset": "image_a"},
        {"model": "model_a"},
        {"backbone": "unet_small"},
        {"time_encoder": "sinusoidal"},
        {"autoencoder": "img_autoencoder"},
        {"optimizer": "adam"},
        {"scheduler": "step"},
        {"run": "default"},
    ],
    trainer=MISSING,
    tester=MISSING,
    dataset=MISSING,
    model=MISSING,
    backbone=MISSING,
    time_encoder=MISSING,
    autoencoder=MISSING,
    optimizer=MISSING,
    scheduler=MISSING,
    run=MISSING,
    data_loader=pbuilds(
        DataLoader, builds_bases=(DataloaderConf,)
    ),  # Needs a partial because we need to set the dataset
)
store(Experiment, name="base_experiment")

# the experiment configs:
# - must be stored under the _global_ package
# - must inherit from `Experiment`
experiment_store = store(group="experiment", package="_global_")
experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "diffusion_model"},
            {"override /backbone": "unet_small"},
            {"override /dataset": "mnist"},
        ],
        backbone=dict(
            input_shape=(28, 28, 1), temporal_channels=512, output_paddings=(1, 0, 1, 1)
        ),
        run=dict(epochs=1000, viz_every=10),
        data_loader=dict(batch_size=128),
        bases=(Experiment,),
    ),
    name="diff_mnist",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "diffusion_model"},
            {"override /backbone": "unet_small"},
            {"override /dataset": "celeba"},
        ],
        backbone=dict(
            input_shape=(64, 64, 3), temporal_channels=512, output_paddings=(1, 1, 1, 1)
        ),
        run=dict(epochs=1000, viz_every=10),
        data_loader=dict(batch_size=64),
        bases=(Experiment,),
    ),
    name="diff_celeba",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "img_autoencoder"},
            {"override /trainer": "ae_trainer"},
            {"override /dataset": "mnist"},
        ],
        model=dict(input_shape=(28, 28, 1), output_paddings=(0, 1, 1)),
        run=dict(epochs=1000, viz_every=10),
        data_loader=dict(batch_size=128),
        bases=(Experiment,),
    ),
    name="ae_mnist",
)


experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "img_autoencoder"},
            {"override /trainer": "ae_trainer"},
            {"override /dataset": "celeba"},
        ],
        model=dict(input_shape=(64, 64, 3), output_paddings=(1, 1, 1)),
        run=dict(epochs=1000, viz_every=10),
        data_loader=dict(batch_size=64),
        bases=(Experiment,),
    ),
    name="ae_celeba",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "latent_diffusion_model"},
            {"override /trainer": "base"},
            {"override /dataset": "celeba"},
            {"override /backbone": "unet_mini"},
            {"override /autoencoder": "img_autoencoder"},
        ],
        autoencoder=dict(input_shape=(64, 64, 3)),
        backbone=dict(
            input_shape=(8, 8, 8),
            temporal_channels=256,
            output_paddings=(1, 1, 1, 1),
        ),  # (16, 16, 256) is the latent feature map shape
        run=dict(epochs=1000, viz_every=10),
        data_loader=dict(batch_size=64),
        bases=(Experiment,),
    ),
    name="ldm_celeba",
)

experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "latent_diffusion_model"},
            {"override /trainer": "base"},
            {"override /dataset": "mnist"},
            {"override /backbone": "unet_micro"},
            {"override /autoencoder": "img_autoencoder"},
        ],
        autoencoder=dict(input_shape=(28, 28, 1)),
        backbone=dict(
            input_shape=(4, 4, 8),
            temporal_channels=128,
            output_paddings=(1, 1, 1, 1),
        ),  # (16, 16, 256) is the latent feature map shape
        run=dict(epochs=1000, viz_every=10),
        data_loader=dict(batch_size=64),
        bases=(Experiment,),
    ),
    name="ldm_mnist",
)


experiment_store(
    make_config(
        hydra_defaults=[
            "_self_",
            {"override /model": "diffusion_model"},
            {"override /backbone": "mlp_unet"},
            {"override /dataset": "mnist"},
        ],
        backbone=dict(input_shape=(28, 28, 1), temporal_channels=256),
        run=dict(epochs=1000, viz_every=10),
        data_loader=dict(batch_size=128),
        bases=(Experiment,),
    ),
    name="diff_mnist_mlp",
)
