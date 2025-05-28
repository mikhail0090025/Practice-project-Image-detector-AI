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

mixed_precision.set_global_policy('mixed_float16')

all_losses = []
all_val_losses = []
all_accuracies = []
all_val_accuracies = []

class GarbageCollectionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        print(f"Garbage collection triggered at the end of epoch {epoch + 1}")

    def on_batch_end(self, batch, logs=None):
        gc.collect()
        print(f"Garbage collection triggered at the end of batch {batch + 1}")

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

# Load statistic
def load_metrics():
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies
    if os.path.exists('metrics.json'):
        with open('metrics.json', 'r') as f:
            metrics = json.load(f)
            all_losses = metrics.get('all_losses', [])
            all_val_losses = metrics.get('all_val_losses', [])
            all_accuracies = metrics.get('all_accuracies', [])
            all_val_accuracies = metrics.get('all_val_accuracies', [])

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
    # url_dataset_manager = f"http://localhost:{dataset_manager_port}/load_image"
    # url_dataset_manager_pathes = f"http://localhost:{dataset_manager_port}/get_dataset_paths"

    pathes_response = requests.get(url_dataset_manager_pathes, timeout=60)
    pathes_response.raise_for_status()
    pathes_json = pathes_response.json()
    print("pathes_json")
    print(pathes_json)

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
            outputs.append([0, 1] if indicator == 0 else [1, 0])
            # [0, 1] - Real image
            # [1, 0] - AI Generated image

    images = np.array(images, dtype=np.float16) / 255.0
    outputs = np.array(outputs, dtype=np.float16)
    print(images)
    print(outputs)
    print(images.shape)
    print(outputs.shape)
    print("Dataset is downloaded")

    np.savez(save_path, images=images, outputs=outputs)
    print(f"Dataset saved to file: {save_path}")
    gc.collect()

    return images, outputs

np.random.seed(52)

path_to_model = 'saved_model.keras'

def get_model():

    # Load model if exists
    try:
        loaded_model = tf.keras.models.load_model(path_to_model)
        print("Saved model found. Loading...")
        return loaded_model
    except ValueError as e:
        print(f"Model file not found at {path_to_model}. Create new. \n {e}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Create new model if not exists
    model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=MV.inputs_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2), strides=(2, 2)),
            # keras.layers.GlobalAveragePooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dropout(0.3),

            keras.layers.Dense(2048, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),

            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.4),

            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(2, activation='softmax')
    ])

    # Other settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

from sklearn.model_selection import train_test_split

def prepare_dataset(images, outputs):
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, outputs, test_size=0.2, random_state=42
    )

    print(f"Train images shape: {train_images.shape}, Val images shape: {val_images.shape}")
    print(f"Train labels shape: {train_labels.shape}, Val labels shape: {val_labels.shape}")

    if val_images.size == 0 or val_labels.size == 0:
        raise ValueError("Validation data are empty!")

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.9, 1.1],
        horizontal_flip=False,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(
        train_images,
        train_labels,
        batch_size=128,
        shuffle=True
    )
    val_generator = val_datagen.flow(
        val_images,
        val_labels,
        batch_size=128,
        shuffle=False
    )

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy', factor=0.5, patience=3, min_lr=1e-6
    )
    SaveCheckpoint = tf.keras.callbacks.ModelCheckpoint(
        path_to_model,
        monitor='accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch'
    )
    return train_generator, val_generator, lr_scheduler, SaveCheckpoint

def main():
    global main_model, images, outputs, train_generator, val_generator, lr_scheduler, SaveCheckpoint
    main_model = get_model()
    images, outputs = get_dataset()
    train_generator, val_generator, lr_scheduler, SaveCheckpoint = prepare_dataset(images, outputs)
    gc.collect()

def go_epochs(epochs_count):
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies
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
    all_losses.extend(history.history['loss'])
    all_val_losses.extend(history.history['val_loss'])
    all_accuracies.extend(history.history['accuracy'])
    all_val_accuracies.extend(history.history['val_accuracy'])
    print("Updated metrics:", all_losses, all_val_losses, all_accuracies, all_val_accuracies)
    save_metrics()
    gc.collect()

# main()