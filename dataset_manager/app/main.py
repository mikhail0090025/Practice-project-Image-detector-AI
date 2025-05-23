from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import os
from data import load_image, get_dataset_paths, get_dataset_files

app = FastAPI()

@app.get("/get_dataset_paths")
async def get_dataset_paths_endpoint():
    paths_with_indicators = get_dataset_paths()
    if not paths_with_indicators:
        return {"error": "Нет зарегистрированных датасетов"}
    return {"paths": paths_with_indicators}

@app.get("/get_dataset_files")
async def get_dataset_files_endpoint(dataset_path: str):
    try:
        files = get_dataset_files(dataset_path)
        return {"files": files}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/load_image")
async def load_image_endpoint(dataset_path: str, image_index: int):
    try:
        image_array, label = load_image(dataset_path, image_index)
        img = Image.fromarray(image_array)
        
        # Сохраняем временный файл
        temp_path = "temp_image.jpg"
        img.save(temp_path)
        
        # Возвращаем изображение и метку
        return FileResponse(temp_path, media_type="image/jpeg", headers={"X-Label": str(label)})
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # Удаляем временный файл