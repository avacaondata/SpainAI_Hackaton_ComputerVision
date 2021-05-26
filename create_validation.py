import glob
import os
import shutil

from tqdm import tqdm

from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    train_names = glob.glob("dataset_srgan/lr/*.png")
    train_names = [f.replace("dataset_srgan/lr/", "") for f in train_names]
    tr, val = train_test_split(train_names, test_size=0.1, random_state=42)
    os.makedirs("val_data_srgan", exist_ok=True)
    os.makedirs("val_data_srgan/lr", exist_ok=True)
    os.makedirs("val_data_srgan/hr", exist_ok=True)
    for name in tqdm(val, desc="Saving val data..."):
        shutil.move(f"dataset_srgan/lr/{name}", f"val_data_srgan/lr/{name}")
        shutil.move(f"dataset_srgan/hr/{name}", f"val_data_srgan/hr/{name}")
