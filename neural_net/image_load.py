import kagglehub        #type: ignore
import os
import shutil
import numpy as np
from PIL import Image

import requests
from io import BytesIO

def load_image_by_file(image_path):
    """
    Loads image from image path, converts to RGB, resizes to 128 x 128 and returns image as numpy array.

    Parameters:
    -----------
    image_url : str
        image path

    Returns:
    -----------
    image_array 
        Array of image with shape (128, 128, 3)
    """
    if image_path is None:
        raise FileNotFoundError(f"{image_path} НЕ НАЙДЕНО В ЯЙЦАХ КОТА")
    
    image = Image.open(image_path).convert('RGB')
    image = image.resize((128, 128))

    image_array = np.array(image)

    if image_array.shape != (128, 128, 3):
        raise ValueError(f"ХУЕВАЯ ФОРМА МАССИВА КАКОГО ХУЯ: {image_array.shape}, ОЖИДАЕТСЯ ХУЙНЯ (128, 128, 3)")
    
    return image_array

def load_image_by_url(image_url):
    """
    Loads image from url, converts to RGB, resizes to 128 x 128 and returns image as numpy array.

    Parameters:
    -----------
    image_url : str
        image url

    Returns:
    -----------
    image_array 
        Array of image with shape (128, 128, 3)
    """
    if image_url is None:
        raise FileNotFoundError(f"{image_url} НЕ НАЙДЕНО В ЯЙЦАХ КОТА")
    
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
    except Exception as e:
        raise FileNotFoundError(f"ХУЕВО ЗАГРУЗИЛСЯ URL {image_url}: {str(e)}")

    image = image.resize((128, 128))
    image_array = np.array(image)

    if image_array.shape != (128, 128, 3):
        raise ValueError(f"ХУЕВАЯ ФОРМА МАССИВА КАКОГО ХУЯ: {image_array.shape}, ОЖИДАЕТСЯ ХУЙНЯ (128, 128, 3)")
    
    return image_array
