from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import neural_net_manager as nnm
import io
import numpy as np
from PIL import Image

app = FastAPI()

@app.get("/")
def root(request: Request):
    return "This is a root of neural net microservice"

@app.post("/go_epochs")
def go_epochs_endpoint(epochs_count: int):
    if epochs_count is None:
        return Response(f"Epochs count has to be a number, but given {epochs_count}", status_code=400)
    if type(epochs_count) is not int:
        return Response(f"Epochs count has to be a number, but given {epochs_count}", status_code=400)
    
    nnm.go_epochs(epochs_count)
    return f"{epochs_count} Epochs successfully passed"

@app.get("/get_graph")
async def get_graph():
    plt.figure(figsize=(10, 6))
    nnm.all_accuracies
    nnm.all_val_accuracies

    plt.plot(range(0, len(nnm.all_accuracies)), nnm.all_accuracies, label="Train Accuracy")
    plt.plot(range(0, len(nnm.all_val_accuracies)), nnm.all_val_accuracies, label="Validation Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=graph.png"}
    )

@app.get("/get_image")
async def get_image(index: int):
    """
    Возвращает изображение из датасета по индексу, размером 128x128.
    """
    try:
        data = np.load("dataset_cache.npz", mmap_mode='r')
        images = data['images']

        if index < 0 or index >= len(images):
            raise ValueError(f"Индекс {index} вне диапазона (0 до {len(images) - 1})")

        image = images[index]
        print(image.shape)  # (128, 128, 3)

        image = (image * 255).astype(np.uint8)

        image = Image.fromarray(image).convert('RGB')

        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)

        data.close()

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=image_{index}.png"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки изображения: {str(e)}")

print("File started")
nnm.main()