import os
import pickle
import yaml
import numpy as np
import optuna
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from tuning import calculate_metrics
from trader.trading_model import TradingModel

def save_metrics_to_yaml(metrics: dict, model_name: str, output_dir: str, filename: str = "model_metrics.yaml"):
    # Define path
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)

    # Load existing file if available
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_data = yaml.safe_load(f) or {}
    else:
        existing_data = {}

    # Update or add metrics for this model
    existing_data[model_name] = {k: float(f"{v:.6f}") for k, v in metrics.items()}

    # Write updated YAML
    with open(output_file, "w") as f:
        yaml.dump(existing_data, f, default_flow_style=False)

    print(f"Saved metrics for model '{model_name}' to {output_file}")


def find_project_root(marker=".git"):
    current_dir = os.path.abspath(os.path.dirname(__file__))
    while True:
        if marker in os.listdir(current_dir):
            return current_dir
        parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        if parent_dir == current_dir:
            raise FileNotFoundError(f"Project root marker '{marker}' not found.")
        current_dir = parent_dir

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
    print('Test Dataset shape:',X_test.shape, y_test.shape)

    scaler_path = os.path.join(processed_dir, "scalers", "y_scaler.pkl")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            y_scaler = pickle.load(f)
    else:
        y_scaler = None

    return X_test, y_test, y_scaler


#def calculate_metrics(y_true, y_pred):
#    mae = mean_absolute_error(y_true, y_pred)
#    mse = mean_squared_error(y_true, y_pred)
#    mda = np.mean((np.sign(y_true) == np.sign(y_pred)).astype(float))
#    smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
#    return mae, mse, mda, smape


def plot_predictions(y_true, y_pred, analysis_path, model_name):
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Predicted vs Actual Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(analysis_path, f"{model_name}_training_mae_plot.png"))
    plt.show()
    plt.close()

# Backtesting with TradingModel
class BacktestTradingModel(TradingModel):
    def __init__(self):
        super().__init__()
        self.trades = []  # List to store trades

    def buy(self, last, symbol, qty):
        self.trades.append({"action": "buy", "price": last, "symbol": symbol, "qty": qty})

    def sell(self, last, symbol, qty):
        self.trades.append({"action": "sell", "price": last, "symbol": symbol, "qty": qty})


def backtest_trading_model(X_test, y_test, model, n_deviations=1.5, initial_capital=10000):
    """
    Perform backtesting using the TradingModel and calculate capital flow.
    :param X_test: Test features (numpy array)
    :param y_test: Test labels (numpy array)
    :param model: Instance of BacktestTradingModel
    :param n_deviations: Number of deviations to determine trade direction
    :param initial_capital: Starting capital for backtesting
    """
    capital = initial_capital
    capital_flow = [capital]

    for i in range(len(X_test)):
        window = X_test[i]
        last_value = y_test[i - 1] if i > 0 else y_test[i]
        model.trade(window, last_value, n_deviations)

        # Update capital based on trades
        for trade in model.trades:
            if trade["action"] == "buy":
                capital -= trade["price"] * trade["qty"]
            elif trade["action"] == "sell":
                capital += trade["price"] * trade["qty"]

        capital_flow.append(capital)

    # Plot capital flow
    plt.figure(figsize=(12, 6))
    plt.plot(capital_flow, label="Capital Flow")
    plt.title("Capital Flow During Backtesting")
    plt.xlabel("Time Steps")
    plt.ylabel("Capital")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()

    return model.trades, capital_flow


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

    paths = model_cfg['paths']
    paths_data = data_cfg['paths']

    base_dir = find_project_root(marker=".git")
    print(f"Base directory: {base_dir}")
    processed_dir   = os.path.join(base_dir, data_cfg["paths"]["processed"])
    study_path      = os.path.join(base_dir, "studies")
    plot_path       = os.path.join(base_dir, paths_data['plots'])
    analysis_path   = os.path.join(base_dir, paths_data['analysis'])

    model_name = model_cfg['name']
    study_name = model_cfg["optuna_param"]["study_name"]
    study_name = f"{study_name}.sqlite3"


    model = keras.models.load_model(os.path.join(base_dir, model_cfg["paths"]["models"], "crnn_final_model.keras"))

    X_test, y_test, y_scaler = load_data(processed_dir)

    y_pred = model.predict(X_test).flatten()
    
    print(y_pred.shape, y_test.shape)
    if y_scaler is not None:
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    mae, mse, mda, smape = calculate_metrics(y_test, y_pred)
    
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"MDA: {mda:.4f}")
    print(f"SMAPE: {smape:.2f}%")

    metrics = {
    "MAE": mae,
    "MSE": mse,
    "MDA": mda,
    "SMAPE": smape
    }
    save_metrics_to_yaml(metrics, model_name=model_name, output_dir=analysis_path)

    if args.plot:
        plot_predictions(y_test, y_pred, analysis_path=plot_path, model_name=model_name)

    
    np.savetxt(os.path.join(base_dir, paths_data['processed'], "y_pred.csv"), y_pred)
    np.savetxt(os.path.join(base_dir, paths_data['processed'], "y_test.csv"), y_test) 
    
    # Initialize the trading model
    trading_model = BacktestTradingModel()

    # Perform backtesting
    trades = backtest_trading_model(X_test, y_test, trading_model)

    # Save trades to a CSV file
    trades_path = os.path.join(base_dir, paths_data['processed'], "trades.csv")
    np.savetxt(trades_path, trades, delimiter=",")
    print(f"Trades saved to {trades_path}")

