import tensorflow as tf
import keras
import main_variables as MV
import numpy as np
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os

all_losses = []
all_val_losses = []
all_accuracies = []
all_val_accuracies = []

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
        print(f.__dir__())

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
    images = []
    outputs = []

    dataset_manager_port = 5001
    url_dataset_manager = f"http://dataset_manager:{dataset_manager_port}/load_image"
    url_dataset_manager_pathes = f"http://dataset_manager:{dataset_manager_port}/all_pathes"

    pathes_response = requests.get(url_dataset_manager_pathes)
    pathes_response.raise_for_status()
    pathes_json = pathes_response.json()

    dataset_response = requests.get(url_dataset_manager)
    dataset_response.raise_for_status()
    dataset_json = dataset_response.json()

    print(dataset_json)
    images.append(dataset_json.image)
    outputs.append(dataset_json.output)
    return images, outputs

np.random.seed(52)

path_to_model = 'model.h5'

def get_model():

    # Load model if exists
    try:
        loaded_model = tf.keras.models.load_model(path_to_model)
        print("Saved model found. Loading...")
        return loaded_model
    except FileNotFoundError:
        print(f"Model file not found at {path_to_model}. Create new")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    # Create new model if not exists
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=MV.inputs_shape),
        keras.layers.MaxPooling2D((2,2), (2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2), (2,2)),
        keras.layers.Conv2D(128, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2), (2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2, activation='softmax'),
    ])

    # Other settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    return model

def prepare_dataset(images, outputs):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        brightness_range=[0.9, 1.1],
        horizontal_flip=False,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=0.2
    )
    val_datagen = ImageDataGenerator(validation_split=0.2)
    train_generator = datagen.flow(
        images,
        outputs,
        batch_size=64,
        subset='training',
        shuffle=True
    )
    val_generator = val_datagen.flow(
        images,
        outputs,
        batch_size=64,
        subset='validation',
        shuffle=False
    )

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(
        monitor='accuracy', factor=0.5, patience=3, min_lr=1e-6
    )
    SaveCheckpoint = tf.keras.callbacks.ModelCheckpoint(path_to_model,
        monitor='accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch')
    return train_generator, val_generator, lr_scheduler, SaveCheckpoint

# Main variables
main_model = get_model()
images, outputs = get_dataset()
train_generator, val_generator, lr_scheduler, SaveCheckpoint = prepare_dataset(images, outputs)

def go_epochs(epochs_count):
    global all_losses, all_val_losses, all_accuracies, all_val_accuracies, main_model
    load_metrics()
    history = main_model.fit(
        train_generator,
        epochs=epochs_count,
        validation_data=val_generator,
        callbacks=[SaveCheckpoint, lr_scheduler]
    )
    all_losses.extend(history.history['loss'])
    all_val_losses.extend(history.history['val_loss'])
    all_accuracies.extend(history.history['accuracy'])
    all_val_accuracies.extend(history.history['val_accuracy'])
    print("Updated metrics:", all_losses, all_val_losses, all_accuracies, all_val_accuracies)
    save_metrics()
