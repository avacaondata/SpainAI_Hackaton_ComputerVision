import glob
import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils_images import split_image_into_overlapping_patches


def process_image_folder(folder, save_folder=None, lr_patch_size=28, hr_patch_size=112):
    input_folder = os.path.join(folder, "input")
    target_folder = os.path.join(folder, "target")
    images_names = [f for f in os.listdir(input_folder)]
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(os.path.join(save_folder, "lr"), exist_ok=True)
    os.makedirs(os.path.join(save_folder, "hr"), exist_ok=True)
    # images_lr = glob.glob(f"{input_folder}/*.png")
    # images_hr = glob.glob(f"{target_folder}/*.png")
    for image_name in tqdm(images_names, desc="Iterating over images for patching..."):
        lr_image = np.array(Image.open(f"{input_folder}/{image_name}"))
        hr_image = np.array(Image.open(f"{target_folder}/{image_name}"))
        patches_lr, _ = split_image_into_overlapping_patches(
            lr_image, lr_patch_size, padding_size=2
        )
        patches_hr, _ = split_image_into_overlapping_patches(
            hr_image, hr_patch_size, padding_size=8
        )
        for i, (patch_lr, patch_hr) in enumerate(zip(patches_lr, patches_hr)):
            imglr = Image.fromarray(patch_lr)
            imghr = Image.fromarray(patch_hr)
            imglr.save(
                os.path.join(
                    save_folder, "lr", f"{image_name.replace('.png', '')}_{i}.png"
                )
            )
            imghr.save(
                os.path.join(
                    save_folder, "hr", f"{image_name.replace('.png', '')}_{i}.png"
                )
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train_folder",
        type=str,
        default="train_data",
    )
    parser.add_argument(
        "--val_folder",
        type=str,
        default="val_data",
    )
    parser.add_argument("--save_folder", type=str, default="ESRGAN-PyTorch/dataset_srgan3")
    args = parser.parse_args()
    print("Processing train data...")
    # os.makedirs(args.save_folder, exist_ok=True)
    process_image_folder(args.train_folder, os.path.join(args.save_folder, "train"))
    process_image_folder(args.val_folder, os.path.join(args.save_folder, "test"))
