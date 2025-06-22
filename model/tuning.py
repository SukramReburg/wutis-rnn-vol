import optuna
import numpy as np
from train import train_model
# from .train import train_model
import os
import yaml
from pathlib import Path


def calculate_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    mda = np.mean((np.sign(y_true) == np.sign(y_pred)).astype(int))
    mse = np.mean((y_true - y_pred) ** 2)
    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    return mae, mda, mse, smape

def objective(trial):
    # Hyperparameter suggestions
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Load model configuration
    with open('config/model_config.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    name = model_config["name"]
    epochs = model_config["epochs"]
    model_save_dir = model_config['paths']['models']

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_path = os.path.join(base_dir, model_save_dir)
    os.makedirs(save_path, exist_ok=True)

    model_name = trial.suggest_categorical("model_name", [name])

    # Load data configuration
    with open('config/data_config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    processed_path = os.path.join(base_dir, data_config['paths']['processed'])

    # Load data
    X_val = np.load(os.path.join(processed_path, "X_val.npy"))
    y_val = np.load(os.path.join(processed_path, "y_val.npy"))

    # Train model
    model = train_model(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        learning_rate=learning_rate,
        verbose=False
    )

    # Get predictions
    y_pred = model.predict(X_val).flatten()

    # Compute metrics
    mae, mda, mse, smape = calculate_metrics(y_val.flatten(), y_pred)

    # Log additional metrics for monitoring
    trial.set_user_attr("mae", mae)
    trial.set_user_attr("mda", mda)
    trial.set_user_attr("mse", mse)
    trial.set_user_attr("smape", smape)

    # Save the model if it's the best so far
    trial_number = trial.number
    model_save_path = os.path.join(save_path, f"best_model_trial_{trial_number}.h5")
    model.save(model_save_path)
    trial.set_user_attr("model_path", model_save_path)

    return mae, mse, 1 - mda

if __name__ == "__main__":

    with open('config/model_config.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    optuna_config = model_config['optuna_param']

    PROJECT_ROOT = Path.cwd()  
    # make sure the folder exists
    optuna_dir = PROJECT_ROOT / "studies"
    optuna_dir.mkdir(exist_ok=True)

    db_path = optuna_dir / "tuning_crnn.sqlite3"
    storage_name = f"sqlite:///{db_path}"

    # Multi-objective optimization with Optuna
    study = optuna.create_study(
        directions=["minimize", "minimize", "minimize"],
        study_name=optuna_config['study_name'],
        storage=storage_name,
        load_if_exists=True
    )
    
    study.optimize(objective, n_trials=15)

    # Analyze results
    print("Best trial:")
    best_trial = study.best_trials[0]
    print(f"  Values: {best_trial.values}")
    print(f"  Params: {best_trial.params}")
    for key, value in best_trial.user_attrs.items():
        print(f"{key}: {value}")

    # Save the best model path
    best_model_path = best_trial.user_attrs.get("model_path")
    if best_model_path:
        print(f"Best model saved at: {best_model_path}")
