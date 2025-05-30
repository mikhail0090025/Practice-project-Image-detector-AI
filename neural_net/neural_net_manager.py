import tensorflow as tf
import keras
import main_variables as MV
import numpy as np
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
from PIL import Image
import io
import gc
from tensorflow.keras import mixed_precision

# Enable mixed precision for better performance (currently commented out)
# mixed_precision.set_global_policy('mixed_float16')

# Initialize variables to track training metrics and total epochs
total_epochs = 0
all_losses = []
all_val_losses = []
all_accuracies = []
all_val_accuracies = []

# Callback to trigger garbage collection after each epoch and batch
class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        print(f"Garbage collection triggered at the end of epoch {epoch + 1}")

    def on_batch_end(self, batch, logs=None):
        gc.collect()

# Save training metrics to a JSON file
def save_metrics():
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies
    metrics = {
        'all_losses': all_losses,
        'all_val_losses': all_val_losses,
        'all_accuracies': all_accuracies,
        'all_val_accuracies': all_val_accuracies
    }
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f)

# Load previously saved training metrics from a JSON file
def load_metrics():
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
            all_losses = metrics.get('all_losses', [])
            all_val_losses = metrics.get('all_val_losses', [])
            all_accuracies = metrics.get('all_accuracies', [])
            all_val_accuracies = metrics.get('all_val_accuracies', [])

# Load or download dataset of images and labels
def get_dataset():
    save_path = "dataset_cache.npz"
    if os.path.exists(save_path):
        print(f"Saved dataset was found. loading...")
        data = np.load(save_path)
        images = data['images']
        outputs = data['outputs']
        print("Saved dataset was loaded")
        return images, outputs
    
    images = []
    outputs = []
    dataset_manager_port = 5000
    url_dataset_manager = f"http://dataset_manager:{dataset_manager_port}/load_image"
    url_dataset_manager_pathes = f"http://dataset_manager:{dataset_manager_port}/get_dataset_paths"

    # Fetch dataset paths from the dataset manager
    pathes_response = requests.get(url_dataset_manager_pathes, timeout=60)
    pathes_response.raise_for_status()
    pathes_json = pathes_response.json()
    print("pathes_json")
    print(pathes_json)

    # Download images and assign labels (real or AI-generated)
    for path_info in pathes_json['paths']:
        dataset_path = path_info[0]
        indicator = path_info[1]
        for image_index in range(1, 1000):
            response = requests.post(
                url_dataset_manager,
                json={"dataset_path": dataset_path, "image_index": image_index},
                headers={"Accept": "application/json"},
                timeout=60
            )
            if response.status_code == 404:
                print(f"All images from {dataset_path} were loaded")
                break

            response.raise_for_status()
            images.append(response.json()['image'])
            outputs.append([0, 1] if indicator == 0 else [1, 0])  # [0, 1] - Real image, [1, 0] - AI-generated image

    images = np.array(images, dtype=np.float16) / 255.0
    outputs = np.array(outputs, dtype=np.float16)
    print(images)
    print(outputs)
    print(images.shape)
    print(outputs.shape)
    print("Dataset is downloaded")

    # Save the dataset to a file for future use
    np.savez(save_path, images=images, outputs=outputs)
    print(f"Dataset saved to file: {save_path}")
    gc.collect()

    return images, outputs

# Set random seed for reproducibility
np.random.seed(52)

path_to_model = 'saved_model.keras'

# Load or create a new CNN model
def get_model():
    try:
        loaded_model = tf.keras.models.load_model(path_to_model)
        print("Saved model found. Loading...")
        return loaded_model
    except ValueError as e:
        print(f"Model file not found at {path_to_model}. Create new. \n {e}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Define a new CNN model with multiple convolutional and dense layers
    model = keras.Sequential([
        keras.layers.Conv2D(20, (5, 5), activation="elu", input_shape=MV.inputs_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(40, (5, 5), activation="elu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Dropout(0.2),
        keras.layers.Conv2D(80, (5, 5), activation="elu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(2048, activation="elu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(256, activation="elu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation="elu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation="elu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(2, activation='softmax')
    ])

    # Compile the model with Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=MV.initial_lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

from sklearn.model_selection import train_test_split

# Prepare training and validation data generators
def prepare_dataset(images, outputs):
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, outputs, test_size=0.2, random_state=52
    )

    print(f"Train images shape: {train_images.shape}, Val images shape: {val_images.shape}")
    print(f"Train labels shape: {train_labels.shape}, Val labels shape: {val_labels.shape}")

    if val_images.size == 0 or val_labels.size == 0:
        raise ValueError("Validation data are empty!")
    
    train_datagen = None
    if MV.augmentation:
        train_datagen = ImageDataGenerator(
            rotation_range=0.001,
            width_shift_range=0.005,
            height_shift_range=0.005,
            brightness_range=[0.995, 1.005],
            horizontal_flip=True,
            zoom_range=0.005,
            fill_mode='nearest'
        )
    else:
        train_datagen = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
        )
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        train_images,
        train_labels,
        batch_size=32,
        shuffle=True
    )
    val_generator = val_datagen.flow(
        val_images,
        val_labels,
        batch_size=32,
        shuffle=False
    )

    # Define callbacks for learning rate scheduling and model checkpointing
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-9
    )
    SaveCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        path_to_model,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    return train_generator, val_generator, lr_scheduler, SaveCheckpoint

# Display dataset statistics (real vs AI-generated images)
def show_data():
    outputs = np.load("dataset_cache.npz", mmap_mode='r')['outputs']
    real_count = np.sum(outputs[:, 0] == 1)
    generated_count = np.sum(outputs[:, 1] == 1)
    total = len(outputs)
    print(f"Real: {real_count} ({real_count/total*100:.1f}%), AI: {generated_count} ({generated_count/total*100:.1f}%)")
    outputs = None
    real_count = None
    generated_count = None
    total = None
    gc.collect()

# Initialize the model, dataset, and data generators
def main():
    global main_model, images, outputs, train_generator, val_generator, lr_scheduler, SaveCheckpoint
    main_model = get_model()
    images, outputs = get_dataset()
    train_generator, val_generator, lr_scheduler, SaveCheckpoint = prepare_dataset(images, outputs)
    show_data()

# Train the model for a specified number of epochs
def go_epochs(epochs_count):
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies, total_epochs
    global main_model, images, outputs, train_generator, val_generator, lr_scheduler, SaveCheckpoint
    gc_callback = GarbageCollectionCallback()
    load_metrics()
    gc.collect()
    history = main_model.fit(
        train_generator,
        epochs=epochs_count,
        validation_data=val_generator,
        callbacks=[SaveCheckpoint, lr_scheduler, gc_callback],
        verbose=1
    )
    total_epochs += epochs_count
    all_losses.extend(history.history['loss'])
    all_val_losses.extend(history.history['val_loss'])
    all_accuracies.extend(history.history['accuracy'])
    all_val_accuracies.extend(history.history['val_accuracy'])
    print("Updated metrics:", all_losses, all_val_losses, all_accuracies, all_val_accuracies)
    save_metrics()
    gc.collect()

# main()