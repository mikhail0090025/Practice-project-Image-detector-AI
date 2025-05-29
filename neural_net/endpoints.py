from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import neural_net_manager as nnm
import io
import numpy as np
from PIL import Image

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.get("/get_graph_accuracy")
async def get_graph_accuracy_endpoint():
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

@app.get("/get_graph_loss")
async def get_graph_loss_endpoint():
    plt.figure(figsize=(10, 6))
    nnm.all_accuracies
    nnm.all_val_accuracies

    plt.plot(range(0, len(nnm.all_losses)), nnm.all_losses, label="Train Loss")
    plt.plot(range(0, len(nnm.all_val_losses)), nnm.all_val_losses, label="Validation Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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

from image_load import load_image_by_file, load_image_by_url

@app.get("/load_image_by_file")
async def load_image_by_file_endpoint(image_path: str):
    """
    Loads an image from a file path, resizes it to 128x128, and returns it as a PNG image.

    Parameters:
    -----------
    image_path : str
        Path to the image file (e.g., /load_image_by_file?image_path=/path/to/image.jpg)

    Returns:
    -----------
    StreamingResponse
        PNG image response
    """
    try:
        image_array = load_image_by_file(image_path)
        # Convert numpy array back to PIL Image for saving
        image = Image.fromarray(image_array)
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=image_from_file.png"}
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/predict_image_url")
async def predict_image_url_endpoint(image_url: str):
    """
    Loads an image from a URL, resizes it to 128x128, predicts its class using the model,
    and returns the prediction as JSON.

    Parameters:
    -----------
    image_url : str
        URL of the image (e.g., /predict_image_url?image_url=https://example.com/image.jpg)

    Returns:
    -----------
    JSONResponse
        Prediction result with probabilities and predicted class
    """
    try:
        image_array = load_image_by_url(image_url)
        print(f"Loaded image shape: {image_array.shape}, dtype: {image_array.dtype}")

        image_array = image_array.astype(np.float32) / 255.0

        image_array = np.expand_dims(image_array, axis=0)  # (1, 128, 128, 3)

        if nnm.main_model is None:
            raise ValueError("Model is not initialized. Call nnm.main() first.")

        prediction = nnm.main_model.predict(image_array, verbose=0)  # verbose=0 для тишины
        print(f"Prediction: {prediction}")

        predicted_class = "ai" if prediction[0][0] > prediction[0][1] else "real"
        probabilities = prediction[0].tolist()  # [prob_real, prob_ai]

        response_data = {
            "prediction": probabilities,
            "predicted_class": predicted_class,
            "answer": "Image generated by AI" if predicted_class == "ai" else "Image is real",
        }

        return JSONResponse(content=response_data, status_code=200)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.get("/another_details")
async def another_details_endpoint():
    # lr = nnm.main_model.optimizer.lr.value()
    train_accuracy = nnm.all_accuracies[-1] if nnm.all_accuracies else 0.5
    val_accuracy = nnm.all_val_accuracies[-1] if nnm.all_val_accuracies else 0.5
    total_epochs = nnm.total_epochs

    response_data = {
        "lr": 0.0001,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "total_epochs": total_epochs,
    }

    return JSONResponse(content=response_data, status_code=200)

print("File started")
nnm.main()