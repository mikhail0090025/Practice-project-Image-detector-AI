from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import os
from dataset_manager.data import load_image, get_dataset_paths, get_dataset_files

app = FastAPI()

@app.post("/go_epochs")
async def go_epochs():
    paths_with_indicators = get_dataset_paths()
    if not paths_with_indicators:
        return {"error": "Нет зарегистрированных датасетов"}
    return {"paths": paths_with_indicators}