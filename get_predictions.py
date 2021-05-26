import glob
import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from tqdm import tqdm, trange

from files_predict import files_predict
from ISR.models import RRDN, Cut_VGG19, Discriminator
from ISR.train import Trainer
from ISR.utils.image_processing import (split_image_into_overlapping_patches,
                                        stich_together)
from skimage.util.shape import view_as_blocks

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

lr_train_patch_size = 60
layers_to_extract = [5, 9]
scale = 4
hr_train_patch_size = lr_train_patch_size * scale


def save_predictions(model, savedir):
    for file in tqdm(files_predict, desc="Iterating over files to predict"):
        img = Image.open(file)
        img = np.array(img)
        # patches = view_as_blocks(img, (lr_train_patch_size, lr_train_patch_size, 3))
        patches, patch_shape = split_image_into_overlapping_patches(
            img, lr_train_patch_size, 0
        )
        predicted_image = np.zeros((2400, 2400, 3))
        # for i in trange(patches.shape[0]):
        #    for j in trange(patches.shape[1]):
        #        pred = model.predict(patches[i, j, 0, :, :, :])
        #        ini_i = i * int(lr_train_patch_size * 4)
        #        end_i = (i + 1) * int(lr_train_patch_size * 4)
        #        ini_j = j * int(lr_train_patch_size * 4)
        #        end_j = (j + 1) * int(lr_train_patch_size * 4)
        #        predicted_image[ini_i:end_i, ini_j:end_j, :] = pred
        predicted_patches = []
        for patch in tqdm(patches):
            pred = model.predict(patch)
            predicted_patches.append(pred)
        predicted_patches = np.array(predicted_patches)
        predicted_patches = stich_together(
            predicted_patches,
            (int(patch_shape[0] * 4), int(patch_shape[1] * 4), 3),
            (2400, 2400, 3),
            padding_size=1,
        )
        predicted_image = predicted_patches.astype(np.uint8)
        predicted = Image.fromarray(predicted_image)
        predicted.save(
            f"{savedir}/candidate_{file.replace('image_600px_', '').replace('./reto_CV_2020_TestSet/TestSet/', '')}"
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--submission_dir",
        required=False,
        default="submission_1303",
        type=str,
        help="Name for submission directory",
    )
    args = parser.parse_args()
    rrdn = RRDN(
        arch_params={"C": 4, "D": 3, "G": 64, "G0": 64, "T": 10, "x": scale},
        patch_size=lr_train_patch_size,
    )
    rrdn.model.load_weights(
        "weights_1103/rrdn-C4-D3-G64-G064-T10-x4/2021-03-13_0312/rrdn-C4-D3-G64-G064-T10-x4_epoch199.hdf5"
    )
    os.makedirs(args.submission_dir, exist_ok=True)
    save_predictions(rrdn, args.submission_dir)
