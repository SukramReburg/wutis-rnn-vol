import os
import numpy as np
import matplotlib.pyplot as plt
from models import create_model, ModelConfig
# from .models import create_model, ModelConfig
import yaml
import tensorflow as tf
from tensorflow import keras


def train_model(model_name, 
                epochs, 
                batch_size, 
                save_path, 
                learning_rate, 
                verbose=True) -> 'tf.keras.Sequential':

    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_path = os.path.join(base_dir, paths['processed'])

    X_train = np.load(os.path.join(processed_path, "X_train.npy"))
    y_train = np.load(os.path.join(processed_path, "y_train.npy"))
    X_val = np.load(os.path.join(processed_path, "X_val.npy"))
    y_val = np.load(os.path.join(processed_path, "y_val.npy"))

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")

    model_config = ModelConfig(input_shape=X_train.shape, learning_rate=learning_rate)
    model = create_model(model_config)

    print("Training model...")

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    # If verbose mode is enabled, generate training plots
    if verbose:

        with open('config/model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        paths = config['paths']
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        analysis_path = os.path.join(base_dir, paths['plots'])
        os.makedirs(analysis_path, exist_ok=True)

        # Plot Mean Absolute Error (MAE)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(analysis_path, f"{model_name}_training_mae_plot.png"))
        plt.close()

        # Plot Loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(os.path.join(analysis_path, f"{model_name}_training_loss_plot.png"))
        plt.close()

    # Save the trained model
    model.save(os.path.join(save_path, f"{model_name}_final_model.keras"))
    print("Model saved")
    return model

if __name__ == "__main__":
    with open('config/model_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config['name']
    path = config['paths']['models']
    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, path)
    os.makedirs(save_path, exist_ok=True)

    model = train_model(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        learning_rate=learning_rate,
        verbose=True
    )
