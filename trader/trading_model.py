import numpy as np
from tensorflow import keras
from keras.models import load_model
from alpaca.trading.client import TradingClient
import joblib
import os 
import yaml
import warnings


class TradingModel:
    def __init__(self):
        # Load the .keras model
        with open('config/model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        model_path = os.path.join(config['paths']['models'], config['name'] + '_final_model.keras')

        with open("config/data_config.yaml", "r") as f:
            config = yaml.safe_load(f) 

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        processed_dir = os.path.join(base_dir, config["paths"]["processed"])

        scaler_path = os.path.join(processed_dir, "scalers", "y_scaler.joblib")
        if os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = joblib.load(f)
        else:
            self.scaler = None

        self.model = load_model(model_path)
        self.last_predicted_value = None

    def predict_next_value(self, window):
        """
        Predict the next value based on the input window.
        :param window: A numpy array of shape (60, 31)
        :return: Predicted next value
        """
        window = np.expand_dims(window, axis=0)  # Add batch dimension
        prediction = self.model.predict(window).flatten()
        prediction = self.scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()[0] if self.scaler else prediction

        return prediction

    def trade(self, window, current_values ,threshold_percent=0.5):
        """
        Place a trade based on the predicted value.
        :param window: A numpy array of shape (60, 31)
        :param last_value: The last value in the series
        :param n_deviations: Number of deviations to determine trade direction
        """
        predicted_value = self.predict_next_value(window)
        if self.last_predicted_value is None:
            self.last_predicted_value = predicted_value
            return None
        pred_percent_change = (predicted_value - self.last_predicted_value) / self.last_predicted_value

        self.last_predicted_value = predicted_value.copy()
        print(f"Percent Change: {pred_percent_change}")
        current_percent_change = (current_values[1] - current_values[0]) / current_values[0]

        with open('config/data_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        ticker = config['tickers'][-1]  # Assuming we are trading the last ticker

        qty = 1 
        if self.last_predicted_value is not None:
            if pred_percent_change > threshold_percent and current_percent_change < 0:
                return self.buy(ticker, qty=pred_percent_change/threshold_percent)

            elif pred_percent_change < -threshold_percent and current_percent_change > 0:
                return self.sell(ticker, qty=pred_percent_change/threshold_percent)
                
            else:
                return None
        else:
            return None
        

    def buy(self,last_value, symbol, qty):
        """
        Placeholder method for buying. Should be overridden in subclasses.
        """
        raise NotImplementedError("The 'buy' method must be implemented in a subclass.")

    def sell(self,last_value, symbol, qty):
        """
        Placeholder method for selling. Should be overridden in subclasses.
        """
        raise NotImplementedError("The 'sell' method must be implemented in a subclass.")


class AlpacaTradingClient(TradingModel):
    def __init__(self, config_path='config/config.yaml'):
        super().__init__()
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        api_key = config['alpaca_api']['api_key']
        secret_key = config['alpaca_api']['secret_key']
        base_url = config['alpaca_api']['base_url']

        # Initialize Alpaca API
        self.trading_client = TradingClient(api_key=api_key, secret_key=secret_key, paper=True, url_override=base_url)

    def buy(self, last_value, symbol, qty):
        """Submit a buy order through Alpaca.

        If the order submission fails a warning is emitted and ``None`` is
        returned.
        """

        abs_qty = abs(qty)

        try:
            return self.trading_client.submit_order(
                symbol=symbol,
                qty=abs_qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
        except Exception as e:  # pragma: no cover - network dependent
            warnings.warn(f"Failed to execute buy order for {symbol}: {e}")
            return None

    def sell(self, last_value, symbol, qty):
        """Submit a sell order through Alpaca.

        Any negative ``qty`` values are converted to a positive number as the
        Alpaca API expects the quantity to always be positive.  If the order
        submission fails, a warning is emitted and ``None`` is returned.
        """

        abs_qty = abs(qty)

        try:
            return self.trading_client.submit_order(
                symbol=symbol,
                qty=abs_qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
        except Exception as e:  # pragma: no cover - network dependent
            warnings.warn(f"Failed to execute sell order for {symbol}: {e}")
            return None

# Example usage:
# if __name__ == "__main__":
#     model = AlpacaTradingClient()    
#     window = np.random.rand(60, 31)  # Example input window
#     last_value = 100.0  # Example last value
#     n_deviations = 2  # Example number of deviations
#     model.trade(window, last_value, n_deviations) 