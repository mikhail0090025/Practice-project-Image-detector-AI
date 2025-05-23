from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import numpy as np
from PIL import Image

app = FastAPI()
app.mount("/images", StaticFiles(directory="app/images"), name="images")

@app.get("/all_pathes")
async def get_all_pathes():
    images_dir = "app/images"
    if not os.path.exists(images_dir):
        return {"error": "Папка с изображениями не найдена"}
    
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    image_paths = [f"/images/{filename}" for filename in image_files]
    return {"paths": image_paths}

@app.get("/load_image/{index}")
async def load_image(index: int):
    images_dir = "app/images"
    if not os.path.exists(images_dir):
        raise HTTPException(status_code=404, detail="Папка с изображениями не найдена")
    
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    if index < 0 or index >= len(image_files):
        raise HTTPException(status_code=404, detail="Изображение с таким индексом не найдено")
    
    image_filename = image_files[index]
    image_path = os.path.join(images_dir, image_filename)
    
    image = Image.open(image_path).convert('RGB').resize((128, 128))
    image_array = np.array(image)
    if image_array.shape != (128, 128, 3):
        raise HTTPException(status_code=400, detail="Неверный формат изображения")
    
    return FileResponse(image_path)