from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import neural_net_manager as nnm
import io
import numpy as np
from PIL import Image

# Initialize the FastAPI application
app = FastAPI()

# Import and configure CORS middleware to allow cross-origin requests from any origin
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,  # Allow credentials (cookies, authorization headers, etc.)
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the root endpoint to provide a simple message
@app.get("/")
def root(request: Request):
    """Root endpoint that returns a welcome message for the neural net microservice."""
    return "This is a root of neural net microservice"

# Endpoint to trigger model training for a specified number of epochs
@app.post("/go_epochs")
def go_epochs_endpoint(epochs_count: int):
    """
    Triggers training of the neural network for the given number of epochs.
    
    Args:
        epochs_count (int): Number of epochs to train the model for.
        
    Returns:
        str: Confirmation message if successful.
        
    Raises:
        Response: Returns a 400 status code if the epochs_count is not a valid integer.
    """
    # Validate the input to ensure epochs_count is a valid integer
    if epochs_count is None:
        return Response(f"Epochs count has to be a number, but given {epochs_count}", status_code=400)
    if type(epochs_count) is not int:
        return Response(f"Epochs count has to be a number, but given {epochs_count}", status_code=400)
    
    # Call the training function from neural_net_manager
    nnm.go_epochs(epochs_count)
    return f"{epochs_count} Epochs successfully passed"

# Endpoint to generate and return a graph of training and validation accuracy
@app.get("/get_graph_accuracy")
async def get_graph_accuracy_endpoint():
    """
    Generates a plot of training and validation accuracy over epochs and returns it as a PNG image.
    
    Returns:
        StreamingResponse: A PNG image of the accuracy graph.
    """
    # Create a new figure for plotting with specified size
    plt.figure(figsize=(10, 6))
    
    # Access accuracy data (though these lines don't affect the plot, they might be for debugging)
    nnm.all_accuracies
    nnm.all_val_accuracies

    # Plot training and validation accuracy
    plt.plot(range(0, len(nnm.all_accuracies)), nnm.all_accuracies, label="Train Accuracy")
    plt.plot(range(0, len(nnm.all_val_accuracies)), nnm.all_val_accuracies, label="Validation Accuracy")
    
    # Set plot title and labels
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(min(min(nnm.all_val_accuracies), min(nnm.all_accuracies)), 1)  # Set y-axis limits to range from 0 to 1
    plt.legend()  # Add a legend to the plot

    # Save the plot to a BytesIO buffer as a PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the plot to free memory
    buf.seek(0)  # Rewind the buffer to the beginning

    # Return the image as a streaming response
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=graph.png"}
    )

# Endpoint to generate and return a graph of training and validation loss
@app.get("/get_graph_loss")
async def get_graph_loss_endpoint():
    """
    Generates a plot of training and validation loss over epochs and returns it as a PNG image.
    
    Returns:
        StreamingResponse: A PNG image of the loss graph.
    """
    # Create a new figure for plotting with specified size
    plt.figure(figsize=(10, 6))
    
    # Access accuracy data (though these lines are unnecessary for the loss plot)
    nnm.all_accuracies
    nnm.all_val_accuracies

    # Plot training and validation loss
    plt.plot(range(0, len(nnm.all_losses)), nnm.all_losses, label="Train Loss")
    plt.plot(range(0, len(nnm.all_val_losses)), nnm.all_val_losses, label="Validation Loss")
    
    # Set plot title and labels
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()  # Add a legend to the plot

    # Save the plot to a BytesIO buffer as a PNG image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()  # Close the plot to free memory
    buf.seek(0)  # Rewind the buffer to the beginning

    # Return the image as a streaming response
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": "inline; filename=graph.png"}
    )

# Endpoint to retrieve an image from the dataset by index
@app.get("/get_image")
async def get_image(index: int):
    """
    Returns an image from the dataset by index, resized to 128x128.

    Parameters:
    -----------
    index : int
        Index of the image in the dataset

    Returns:
    -----------
    StreamingResponse
        PNG image response

    Raises:
    -----------
    HTTPException: If the image cannot be loaded or the index is out of range
    """
    try:
        # Load the dataset in read-only mode to save memory
        data = np.load("dataset_cache.npz", mmap_mode='r')
        images = data['images']

        # Validate the index to ensure it is within the dataset range
        if index < 0 or index >= len(images):
            raise ValueError(f"Index {index} is out of range (0 to {len(images) - 1})")

        # Retrieve the image at the specified index
        image = images[index]
        print(image.shape)  # Expected shape: (128, 128, 3)

        # Convert the image array to uint8 format for PIL compatibility
        image = (image * 255).astype(np.uint8)

        # Convert numpy array to PIL Image and ensure RGB format
        image = Image.fromarray(image).convert('RGB')

        # Save the image to a BytesIO buffer as PNG
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)  # Rewind the buffer to the beginning

        # Close the dataset to free resources
        data.close()

        # Return the image as a streaming response
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=image_{index}.png"}
        )
    except Exception as e:
        # Raise an HTTP exception if an error occurs during image loading
        raise HTTPException(status_code=500, detail=f"Error loading image: {str(e)}")

# Import image loading utilities
from image_load import load_image_by_file, load_image_by_url

# Endpoint to load an image from a file path and return it as PNG
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

    Raises:
    -----------
    HTTPException: If the file is not found, the shape is invalid, or a server error occurs
    """
    try:
        # Load the image using the utility function
        image_array = load_image_by_file(image_path)
        
        # Convert numpy array back to PIL Image for saving
        image = Image.fromarray(image_array)
        
        # Save the image to a BytesIO buffer as PNG
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)  # Rewind the buffer to the beginning

        # Return the image as a streaming response
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=image_from_file.png"}
        )
    except FileNotFoundError as e:
        # Handle file not found error
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        # Handle invalid image shape error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle any other server errors
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Endpoint to predict the class of an image from a URL
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

    Raises:
    -----------
    HTTPException: If the URL is invalid, the shape is incorrect, or a server error occurs
    """
    try:
        # Load the image from the URL using the utility function
        image_array = load_image_by_url(image_url)
        print(f"Loaded image shape: {image_array.shape}, dtype: {image_array.dtype}")

        # Normalize the image array for model prediction
        image_array = image_array.astype(np.float32) / 255.0

        # Add a batch dimension to the image array for prediction
        image_array = np.expand_dims(image_array, axis=0)  # Shape: (1, 128, 128, 3)

        # Check if the model is initialized
        if nnm.main_model is None:
            raise ValueError("Model is not initialized. Call nnm.main() first.")

        # Predict the class using the neural network model
        prediction = nnm.main_model.predict(image_array, verbose=0)  # Silent prediction
        print(f"Prediction: {prediction}")

        # Determine the predicted class based on probabilities
        predicted_class = "ai" if prediction[0][0] > prediction[0][1] else "real"
        probabilities = prediction[0].tolist()  # Convert probabilities to list for JSON response

        # Prepare the response data
        response_data = {
            "prediction": probabilities,  # [prob_ai, prob_real]
            "predicted_class": predicted_class,
            "answer": "Image generated by AI" if predicted_class == "ai" else "Image is real",
        }

        # Return the prediction as a JSON response
        return JSONResponse(content=response_data, status_code=200)

    except FileNotFoundError as e:
        # Handle URL not found error
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        # Handle invalid image shape or uninitialized model error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle any other server errors
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# Endpoint to get model training details
@app.get("/another_details")
async def another_details_endpoint():
    """
    Returns training details including learning rate, accuracy, and total epochs trained.

    Returns:
        JSONResponse: A JSON object containing training details.
    """
    # Retrieve the latest training and validation accuracy, default to 0.5 if not available
    train_accuracy = nnm.all_accuracies[-1] if nnm.all_accuracies else 0.5
    val_accuracy = nnm.all_val_accuracies[-1] if nnm.all_val_accuracies else 0.5
    
    # Get the total number of epochs trained
    total_epochs = nnm.total_epochs

    # Prepare the response data with training details
    response_data = {
        "lr": 0.0001,  # Hardcoded since lr retrieval is commented out
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "total_epochs": total_epochs,
    }

    # Return the details as a JSON response
    return JSONResponse(content=response_data, status_code=200)

# Print a message to indicate the file has started and initialize the neural net manager
print("File started")
nnm.main()