import os
import pickle
import yaml
import numpy as np
import optuna
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def load_best_model(study_path: str, study_name: str):
    """Load the best model stored in an Optuna study."""
    storage = f"sqlite:///{study_path}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    best_trial = study.best_trials[0]
    model_path = best_trial.user_attrs.get("model_path")
    if model_path is None:
        raise ValueError("Best trial does not contain a model path")
    model = keras.models.load_model(model_path)
    return model


def load_data(processed_dir: str):
    """Load test arrays and scalers."""
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))

    scaler_path = os.path.join(processed_dir, "y_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            y_scaler = pickle.load(f)
    else:
        y_scaler = None

    return X_test, y_test, y_scaler


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mda = np.mean((np.sign(y_true) == np.sign(y_pred)).astype(float))
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    return mae, mse, mda, smape


def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Predicted vs Actual Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest the best trained model")
    parser.add_argument("--plot", action="store_true", help="Plot predictions")
    args = parser.parse_args()

    # Load configs
    with open("config/model_config.yaml", "r") as f:
        model_cfg = yaml.safe_load(f)
    with open("config/data_config.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir)
    processed_dir = os.path.join(base_dir, data_cfg["paths"]["processed"])
    study_path = os.path.join(base_dir, "studies", "tuning_crnn.sqlite3")
    study_name = model_cfg["optuna_param"]["study_name"]

    model = load_best_model(study_path, study_name)
    X_test, y_test, y_scaler = load_data(processed_dir)

    y_pred = model.predict(X_test).flatten()

    if y_scaler is not None:
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    mae, mse, mda, smape = calculate_metrics(y_test, y_pred)
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MDA: {mda:.4f}")
    print(f"SMAPE: {smape:.2f}%")

    if args.plot:
        plot_predictions(y_test, y_pred)
