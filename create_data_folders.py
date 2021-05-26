import os
import shutil

from tqdm import tqdm

if __name__ == "__main__":
    folders_train = ["reto_CV_2020_TrainingSetPart1/TrainingSetPart1", "reto_CV_2020_TrainingSetPart2/TrainingSetPart2"]
    folder_test = "reto_CV_2020_TestSet/TestSet"
    os.makedirs("train_data", exist_ok=True)
    os.makedirs("train_data/input", exist_ok=True)
    os.makedirs("train_data/target", exist_ok=True)
    os.makedirs("test_data", exist_ok=True)
    os.makedirs("test_data/input", exist_ok=True)
    os.makedirs("test_data/target", exist_ok=True)
    files_tr = []
    for folder in tqdm(folders_train, desc="Iterating over folders"):
        for file in tqdm(os.listdir(f"{folder}/600px/"), desc="Iterating over images..."):
            name = file.replace("image_600px_", "")
            shutil.copy(f"{folder}/600px/{file}", f"train_data/input/{name}")
        for file in tqdm(os.listdir(f"{folder}/2400px/"), desc="Itearing over images 2..."):
            name = file.replace("image_2400px_", "")
            shutil.copy(f"{folder}/2400px/{file}", f"train_data/target/{name}")
    #for file in tqdm(os.listdir(folder_test), desc="Iterating over images 3..."):
    #    name = file.replace("image_600px_", "")
    #    shutil.copy(f"{folder_test}/{file}", f"test_data/input/{name}")
    #    shutil.copy(f"{folder_test}/{file}", f"test_data/target/{name}")
