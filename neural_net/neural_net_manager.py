import tensorflow as tf
import keras
import main_variables as MV
import numpy as np

dataset_manager_port = 5001
url_dataset_manager = f"http://localhost:{dataset_manager_port}/"

np.random.seed(42)

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

# Main variables
main_model = get_model()
x_train = []
y_train = []
x_test = []
y_test = []