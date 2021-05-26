import multiprocessing as mp
import os
import shutil
from functools import partial

from tqdm import tqdm


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def images_loop(images_names, folder, target, lr=True):
    s = "600px" if lr else "2400px"
    for image_name in tqdm(images_names, desc="Iterating over images"):
        name = image_name.replace(f"image_{s}_", "")
        shutil.copy(f"{folder}/{s}/{image_name}", f"{target}{name}")


if __name__ == "__main__":
    folders_train = ["./raw_data/reto_CV_2020_TrainingSetPart1/TrainingSetPart1", "./raw_data/reto_CV_2020_TrainingSetPart2/TrainingSetPart2"]
    # folder_test = "./raw_datareto_CV_2020_TestSet/TestSet"
    os.makedirs("train_data", exist_ok=True)
    os.makedirs("train_data/input", exist_ok=True)
    os.makedirs("train_data/target", exist_ok=True)
    # os.makedirs("test_data", exist_ok=True)
    # os.makedirs("test_data/lr", exist_ok=True)
    # os.makedirs("test_data/hr", exist_ok=True)
    files_tr = []
    cpus = mp.cpu_count()
    pool = mp.Pool(cpus)
    for folder in tqdm(folders_train, desc="Iterating over folders"):
        files_lr = [file_ for file_ in os.listdir(f"{folder}/600px/")]
        # train_data/input/
        files_lr_batches = list(chunks(files_lr, len(files_lr) // cpus))
        files_hr = [f for f in os.listdir(f"{folder}/2400px/")]
        files_hr_batches = list(chunks(files_hr, len(files_hr) // cpus))
        pool.map(partial(images_loop, folder=folder, target="train_data/input/"), files_lr_batches)
        pool.map(partial(images_loop, folder=folder, target="train_data/target/", lr=False), files_hr_batches)
        #for file in tqdm(, desc="Iterating over images..."):
        #    name = file.replace("image_600px_", "")
        #    shutil.copy(f"{folder}/600px/{file}", f"train_data/input/{name}")
        #for file in tqdm(, desc="Itearing over images 2..."):
        #    name = file.replace("image_2400px_", "")
        #    shutil.copy(f"{folder}/2400px/{file}", f"train_data/target/{name}")


    #for file in tqdm(os.listdir(folder_test), desc="Iterating over images 3..."):
    #    name = file.replace("image_600px_", "")
    #    shutil.copy(f"{folder_test}/{file}", f"test_data/input/{name}")
    #    shutil.copy(f"{folder_test}/{file}", f"test_data/target/{name}")
