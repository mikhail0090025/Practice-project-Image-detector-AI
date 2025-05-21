import tensorflow as tf
import keras
import main_variables as MV

path_to_model = 'model.h5'

def get_model():
    try:
        return tf.keras.models.load_model(path_to_model)
    except FileNotFoundError:
        print(f"Model file not found at {path_to_model}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

get_model()