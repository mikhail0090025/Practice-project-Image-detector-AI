import os
import shutil
import kagglehub
from data import rename_images_in_folder

def check_dataset_exists(target_path):
    return os.path.exists(target_path) and len(os.listdir(target_path)) > 0

def download_dataset(dataset_url, target_path):
    if check_dataset_exists(target_path):
        print(f"Dataset already exists in {target_path}. Skipping download.")
        return

    print(f"Downloading dataset from {dataset_url}...")
    cache_path = kagglehub.dataset_download(dataset_url)
    print("Path to cached dataset files:", cache_path)

    os.makedirs(target_path, exist_ok=True)

    for root, _, files in os.walk(cache_path):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, cache_path)
            target_file_path = os.path.join(target_path, rel_path)
            os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
            shutil.move(file_path, target_file_path)
            print(f"Moved {file} to {target_file_path}")

    print(f"Dataset moved to: {target_path}")
    shutil.rmtree(cache_path)
    print(f"Cleaned up cache directory: {cache_path}")

if __name__ == "__main__":
    target_path = "app/images"
    dataset_url = "cashbowman/ai-generated-images-vs-real-images"
    download_dataset(dataset_url, target_path)
    rename_images_in_folder(target_path)