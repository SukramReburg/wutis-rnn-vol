import tensorflow as tf

from tensorflow import keras
from keras import layers, models, optimizers
from keras.optimizers import Adam



class ModelConfig:
    def __init__(self, input_shape, learning_rate): # TODO: add more parameters for model config
        self.input_shape = input_shape
        self.learning_rate = learning_rate

def create_model(model_config: ModelConfig):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Input(shape=model_config.input_shape[1:]))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Reshape for RNN layers
    model.add(layers.Reshape((-1, model.output_shape[-1])))

    # Recurrent layers
    model.add(layers.GRU(128, return_sequences=True))
    model.add(layers.GRU(128))

    # Fully connected layer
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    model.compile(
        optimizer=Adam(model_config.learning_rate),
        loss='mean_squared_error',
        metrics=['mae']  # Mean Absolute Error
    )
    model.summary()

    return model
