from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import neural_net_manager as nnm
import io

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

print("File started")
nnm.main()