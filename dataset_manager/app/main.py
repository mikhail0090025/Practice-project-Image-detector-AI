from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import numpy as np
from PIL import Image
import os
from pydantic import BaseModel
from data import load_image, get_dataset_paths, get_dataset_files

app = FastAPI()

class ImageRequest(BaseModel):
    dataset_path: str
    image_index: int

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

@app.post("/load_image")
async def load_image_endpoint(request: ImageRequest):
    dataset_path = request.dataset_path
    image_index = request.image_index
    try:
        temp_path = "temp_image.jpg"
        image_array, label = load_image(dataset_path, image_index)
        return JSONResponse({"image": image_array.tolist(), "category": label.tolist()}, status_code=200)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Unexpected exception in load_image_endpoint - {e}")
        raise
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)  # Удаляем временный файл
            