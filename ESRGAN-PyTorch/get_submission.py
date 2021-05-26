import glob
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

import esrgan_pytorch.models as models
from esrgan_pytorch.utils.transform import process_image
from external_utils import split_image_into_overlapping_patches, stich_together


class TemporalDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, img):
        img_array = np.array(img)
        patches, _ = split_image_into_overlapping_patches(img_array, 28, 2)
        self.image_patches = [
            Image.fromarray(patch) for patch in patches
        ]
        self.transforms = transforms.ToTensor()

    def __getitem__(self, index):
        return self.transforms(self.image_patches[index])

    def __len__(self):
        return len(self.image_patches)


def save_predictions(model, files, savedir, predict_folder):
    model.eval()
    for file in tqdm(files, desc="Iterating over files to predict"):
        img = Image.open(file)
        dataset = TemporalDataset(img)
        # predicted_image = np.zeros((2400, 2400, 3))
        predicted_patches = []
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=False, batch_size=64, drop_last=False
        )
        for batch in tqdm(dataloader, desc="Iterating over batches"):
            with torch.no_grad():
                predicted = model(batch)
            predicted_patches.extend(
                [
                    np.array(transforms.ToPILImage()(pred))
                    for pred in predicted
                ]
            )
        predicted_image = stich_together(
            np.array(predicted_patches),
            (2464, 2464, 3),
            (2400, 2400, 3),
            padding_size=8
        )
        predicted_image = predicted_image.astype(np.uint8)
        predicted = Image.fromarray(predicted_image)
        predicted.save(
            f"{savedir}/candidate_{file.replace('image_600px_', '').replace(predict_folder, '')}"
        )




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/home/alejandro.vaca/reto_gans/ESRGAN-PyTorch/weights/PSNR_epoch55.pth",
        type=str,
        help="Checkpoint dir",
    )
    parser.add_argument("--device", default="cpu", type=str, help="Device to use")
    parser.add_argument(
        "--target_dir",
        default="/home/alejandro.vaca/reto_gans/submission_2903",
        type=str,
        help="Directory to save generated images",
    )
    parser.add_argument(
        "--predict_folder",
        default="/home/alejandro.vaca/reto_gans/ESRGAN-PyTorch/test/reto_CV_2020_TestSet/TestSet/",
        type=str,
        help="The directory where images are."
    )
    args = parser.parse_args()
    os.makedirs(args.target_dir, exist_ok=True)
    generator = models.__dict__["esrgan16"]()
    checkpoint = torch.load(
            args.model_path, map_location=torch.device(args.device)
    )
    generator.load_state_dict(
        checkpoint["state_dict"]
    )
    files_predict = glob.glob(os.path.join(args.predict_folder, "*.png"))
    save_predictions(generator, files_predict, args.target_dir, args.predict_folder)
