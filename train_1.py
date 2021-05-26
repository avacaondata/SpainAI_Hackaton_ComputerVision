import os

import numpy as np
import tensorflow as tf
from PIL import Image

from ISR.models import RRDN, Cut_VGG19, Discriminator
from ISR.train import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    lr_train_patch_size = 192
    layers_to_extract = [5, 9]
    scale = 4
    hr_train_patch_size = lr_train_patch_size * scale
    rrdn = RRDN(
        arch_params={"C": 4, "D": 3, "G": 64, "G0": 64, "T": 16, "x": scale},
        patch_size=lr_train_patch_size,
    )
    # rrdn.model.load_weights()
    f_ext = Cut_VGG19(
        patch_size=hr_train_patch_size, layers_to_extract=layers_to_extract
    )
    discr = Discriminator(patch_size=hr_train_patch_size, kernel_size=3)
    loss_weights = {
        "generator": 1,
        "feature_extractor": 0,  # 0.0833,
        "discriminator": 0,  # 0.01,
    }

    losses = {
        "generator": "mae",
        "feature_extractor": "mse",
        "discriminator": "binary_crossentropy",
    }

    log_dirs = {"logs": "./logs", "weights": "./weights_2403"}

    learning_rate = {
        "initial_value": 0.0001,
        "decay_factor": 0.5,
        "decay_frequency": 30,
    }

    flatness = {"min": 0.0, "max": 0.15, "increase": 0.01, "increase_frequency": 5}

    trainer = Trainer(
        generator=rrdn,
        discriminator=discr,
        feature_extractor=f_ext,
        lr_train_dir="train_data/input/",
        hr_train_dir="train_data/target/",
        lr_valid_dir="val_data/input/",
        hr_valid_dir="val_data/target/",
        loss_weights=loss_weights,
        learning_rate=learning_rate,
        flatness=flatness,
        dataname="spainai",
        log_dirs=log_dirs,
        # weights_generator="./weights_1003/rrdn-C4-D3-G64-G064-T10-x4/2021-03-11_1227/rrdn-C4-D3-G64-G064-T10-x4_best-val_generator_PSNR_Y_epoch099.hdf5",
        weights_discriminator=None,
        n_validation=100,
    )
    trainer.train(
        epochs=200,
        steps_per_epoch=200,
        batch_size=16,
        monitored_metrics={"val_generator_PSNR_Y": "max"},  # generator_
    )
