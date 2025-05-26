import kagglehub        #type: ignore
import os
import shutil
import numpy as np
from PIL import Image

"""
==================================================================================
DATASET DOWNLOADING
==================================================================================
"""
def check_dataset_exists(target_path):
    return os.path.exists(target_path) and len(os.listdir(target_path)) > 0

def download_dataset(dataset_url, target_path):
    if check_dataset_exists(target_path):
        print(f"Dataset already exists in {target_path}. Skipping download.")
        return

    print(f"Downloading dataset from {dataset_url}...")
    cache_path = kagglehub.dataset_download(dataset_url)
    print("Path to cached dataset files:", cache_path)

    print("=====================================================================================")
    print("MOVING FILES")
    print("=====================================================================================")

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

    try:
        shutil.rmtree(cache_path)
        print(f"Cleaned up cache directory: {cache_path}")
    except Exception as e:
        print(f"Failed to clean up cache directory {cache_path}: {e}")

def rename_images_in_folder(folder_path):
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]
    files.sort()

    print(f"Found {len(files)} image files in {folder_path}")

    temp_files = []
    for idx, filename in enumerate(files):
        old_path = os.path.join(folder_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        temp_filename = f"__temp_{idx}{ext}"
        temp_path = os.path.join(folder_path, temp_filename)
        os.rename(old_path, temp_path)
        temp_files.append((temp_path, ext))

    for idx, (temp_path, ext) in enumerate(temp_files, 1):
        new_filename = f"{idx}{ext}"
        final_path = os.path.join(folder_path, new_filename)
        os.rename(temp_path, final_path)
        print(f"Renamed: {os.path.basename(temp_path)} -> {new_filename}")

"""
===============================================================================
WORK WITH DATASET
===============================================================================
"""
def load_image(dataset_path, image_index):
    """
    Loads image from dataset folder by index and returns numpy array and image mark

    Parameters:
    ----------
    dataset_path : str
        Path to dataset, example: .datasets/datasetX/ai or .datasets/datasetX/real
    image_index : int
        Image number index

    Returns:
    -----------
    tuple : (image_array, label)
        image_array
            Array of image with shape (128, 128, 3)
        label
            Category of image ([1, 0] for real, [0, 1] for ai)

    Exceptions:
    -----------
    FileNotFoundError
        говно не найдено в папке категории
    ValueError
        если форма залупки не соответствует с хуйней (128, 128, 3).
    """
    print(f"Full path: {os.path.abspath(dataset_path)}")
    if not os.path.exists(dataset_path):
        print(f"Path {dataset_path} does not exist")
        raise FileNotFoundError(f"Path {dataset_path} not found")

    category = os.path.basename(dataset_path).lower()
    if category not in ['real', 'ai']:
        print(f"Forled {dataset_path} must be real or ai")
        raise ValueError(f"Forled {dataset_path} must be real or ai")

    extensions = ['.png', '.jpg', '.jpeg']
    
    image_path = None
    for ext in extensions:
        potential_path = os.path.join(dataset_path, f"{image_index}{ext}")
        if os.path.exists(potential_path):
            image_path = potential_path
            break
    
    print(f"Image path: {image_path}, index: {image_index}")
    if image_path is None:
        raise FileNotFoundError(f"ЗАЛУПА С ИНДЕКСОМ ХУЙНИ {image_index} НЕ НАЙДЕНО В ЯЙЦАХ КОТА {category}")
    
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))
    
    image_array = np.array(image)
    
    if image_array.shape != (128, 128, 3):
        raise ValueError(f"ХУЕВАЯ ФОРМА МАССИВА КАКОГО ХУЯ: {image_array.shape}, ОЖИДАЕТСЯ ХУЙНЯ (128, 128, 3)")
    
    label = np.array([1, 0]) if category == 'real' else np.array([0, 1])
    return image_array, label

dataset_paths = []

def add_dataset_path(dataset_path, indicator):
    """
    Adds dataset to global list dataset_paths.

    Parameters:
    -----------
    dataset_path : str
        Path to dataset folder, example: .datasets/datasetX/ai
    indicator : int
        indicates if photo is ai generated or real, 1 - ai, 0 - real
    """
    dataset_paths.append((dataset_path, indicator))

def get_dataset_paths():
    """
    Returns all paths to dataset with their indicators

    Returns:
    -----------
    list
        List of registered dataset paths and their indicators, 1 - ai, 0 - real
    """
    return dataset_paths

def get_dataset_files(dataset_path):
    """
    Returns list of names of all files in directory (dataset)

    Parameters:
    -----------
    category_path : str
        Path to dataset folder, example: .datasets/datasetX/real or datasets/datasetX/ai

    Возвращает:
    -----------
    list
        List of all file names in directory (dataset)

    Exceptions:
    -----------
    ValueError
        Если ты pridurok и вызвал не существующий path
    """
    if not os.path.exists(dataset_path):
        raise ValueError(f"Path {dataset_path} not found")
    
    extensions = ['.png', '.jpg', '.jpeg']
    files = [os.path.splitext(f)[0] for f in os.listdir(dataset_path) 
             if os.path.isfile(os.path.join(dataset_path, f)) and 
             os.path.splitext(f)[1].lower() in extensions]
    
    return files

def main():
    """
    --------------------------
    - переименовать main в нужную залупку
    --------------------------
    - после скачивания папки датасетов ai/real с фотками завернуты в отдельную папку
    - нужно эти две папки вытащить в коренную т.е. .datasets/datasetX/ 
    - и переименовать в соответствующие
    - чтобы вышла структура 
    - .datasets
    -   datasetX
    -       ai
    -       real
    - это конечно можно автоматизировать с костылями
    - но я ебал
    ---------------------------
    - пример использования load_image
    - load_image("./datasets/dataset1", 1, "real")
    - load_image("./datasets/dataset1", 2, "real")
    ---------------------------
    """
    print("otsos")

    #target_directory = "./datasets/dataset1"
    #download_dataset("cashbowman/ai-generated-images-vs-real-images", target_directory)
    #rename_images_in_folder("./datasets/dataset1/real")
    #rename_images_in_folder("./datasets/dataset1/ai")
    #print(load_image("./datasets/dataset1", 1, "real"))
    #print(load_image("./datasets/dataset1", 1, "ai"))
    
    #target_directory = "./datasets/dataset2"
    #download_dataset("rafsunahmad/camera-photos-vs-ai-generated-photos-classifier", target_directory)
    #rename_images_in_folder("./datasets/dataset2/ai")
    #rename_images_in_folder("./datasets/dataset2/real")
    #print(load_image("./datasets/dataset2", 1, "real"))
    #print(load_image("./datasets/dataset2", 1, "ai"))

    """
    - ВАЖНО ДЛЯ ЛОГИКИ (скорее всего стоит засунуть в отдельную функцию с инициализацией)
    """
    add_dataset_path("./datasets/dataset1/ai", 1)
    add_dataset_path("./datasets/dataset1/real", 0)
    add_dataset_path("./datasets/dataset2/ai", 1)
    add_dataset_path("./datasets/dataset2/real", 0)

    """
    - пример использования get_dataset_paths & get_dataset_files
    - load_image(get_dataset_paths()[0][0], int(get_dataset_files(get_dataset_paths()[0][0])[0]))
    - вернет самую первую фотку по самому первому пути в dataset_paths
    """
    #print(get_dataset_paths())
    #print(get_dataset_files(get_dataset_paths()[0]))
    #print(load_image(get_dataset_paths()[0]), int(get_dataset_files(get_dataset_paths()[0])[0]))
    #print(load_image("./datasets/dataset1/real", 1))
    #print(load_image(get_dataset_paths()[0], int(get_dataset_files(get_dataset_paths()[0])[0])))

    #print(load_image(get_dataset_paths()[0][0], int(get_dataset_files(get_dataset_paths()[0][0])[0])))

main()