import glob
import os
import shutil

from tqdm import tqdm

from sklearn.model_selection import train_test_split
import multiprocessing as mp
from functools import partial

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def loop(images, source_dir, target_dir):
    for image in tqdm(images):
        #source = f"{source_dir}{image}"
        #target = f"{target_dir}{image}"
        shutil.copy(os.path.join(source_dir, "input", image), os.path.join(target_dir, "lr", image))
        shutil.copy(os.path.join(source_dir, "target", image), os.path.join(target_dir, "hr", image))


if __name__ == "__main__":
    train_names = glob.glob("train_data/input/*.png")
    train_names = [f.replace("train_data/input/", "") for f in train_names]
    tr, val = train_test_split(train_names, test_size=0.1, random_state=42)
    print(train_names)
    assert len(tr) + len(val) == len(train_names)
    assert all([text not in tr for text in val])
    #os.makedirs("val_data_srgan", exist_ok=True)
    #os.makedirs("val_data_srgan/lr", exist_ok=True)
    #os.makedirs("val_data_srgan/hr", exist_ok=True)
    os.makedirs("dataset_srgan3", exist_ok=True)
    os.makedirs("dataset_srgan3/train", exist_ok=True)
    os.makedirs("dataset_srgan3/train/lr", exist_ok=True)
    os.makedirs("dataset_srgan3/train/hr", exist_ok=True)
    os.makedirs("dataset_srgan3/test", exist_ok=True)
    os.makedirs("dataset_srgan3/test/lr", exist_ok=True)
    os.makedirs("dataset_srgan3/test/hr", exist_ok=True)
    cpus = mp.cpu_count()
    val_chunks = list(chunks(val, len(val) // cpus))
    train_chunks = list(chunks(tr, len(tr) // cpus))
    pool = mp.Pool(cpus)
    pool.map(partial(loop, source_dir="train_data", target_dir="dataset_srgan3/train"), train_chunks)
    pool.map(partial(loop, source_dir="train_data", target_dir="dataset_srgan3/test"), val_chunks)

    #for name in tqdm(val, desc="Saving val data..."):
    #    shutil.move(, f"val_data_srgan/lr/{name}")
    #    shutil.move(f"dataset_srgan/hr/{name}", f"val_data_srgan/hr/{name}")
