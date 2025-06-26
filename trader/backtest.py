import backtrader as bt
from trading_model import TradingModel
import pandas as pd
import yaml
import os 
import numpy as np
import joblib

class TestTradingModel(TradingModel):
    def __init__(self):
        super().__init__()

    def buy(self, symbol, qty):
        return {"dir": 1, "qty": qty}
    
    def sell(self, symbol, qty):
        return {"dir": -1, "qty": qty}

class BacktestModel(bt.Strategy):
    def __init__(self, trading_model, X_test, y_test, scaler, threshold_percent=1.0):
        self.trading_model = trading_model
        self.X_test = X_test
        self.y_test = y_test
        self.scaler = scaler 
        self.window_size = 60
        self.bar_counter = 0
        self.threshold_percent = threshold_percent
        self.idx = 0  # to index into X_test

    def next(self):
        if len(self) < self.window_size:
            return

        if self.idx < len(self.X_test):
            window = self.X_test[self.idx]
            last_value = self.y_test[self.idx - 2:self.idx] if self.idx > 1 else None   # only to get the last scaled value without scaling X_test
            if last_value is not None:
                trade = self.trading_model.trade(window, last_value, self.threshold_percent)
                print(trade)
                if trade:
                    direction = trade['dir']
                    qty = trade['qty']

                    if direction == 1:
                        self.buy(size=qty)
                    elif direction == -1:
                        self.sell(size=qty)

            self.idx += 1


def load_data(processed_dir: str):
    """Load test arrays and scalers."""
    X_test = np.load(os.path.join(processed_dir, "X_test.npy"))
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    print('Test Dataset shape:',X_test.shape, y_test.shape)

    scaler_path = os.path.join(processed_dir, "scalers", "y_scaler.joblib")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = joblib.load(f)
    else:
        scaler = None
    print("Scaler loaded:", scaler)

    return X_test, y_test, scaler

if __name__ == "__main__":

    with open("config/data_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, config["paths"]["processed"])
    raw_dir = os.path.join(base_dir, config["paths"]["raw"])

    vixy_path = os.path.join(raw_dir, "VIXY.csv")
    
    X_test, y_test, y_scaler = load_data(processed_dir)

    if y_scaler is not None:
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Load data
    df = pd.read_csv(vixy_path).loc[y_test.shape[0]-1:]
    # Convert 'timestamp' to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Set as the index
    df.set_index('timestamp', inplace=True)

    data = bt.feeds.PandasData(
        dataname=df,
        timeframe=bt.TimeFrame.Minutes,
        compression=1  # 1-minute frequency
    )

    # Initialize TradingModel
    trading_model = TestTradingModel()

    # Initialize backtrader
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(BacktestModel, trading_model=trading_model, X_test=X_test, y_test=y_test, scaler=y_scaler, threshold_percent=0.025)
    #Run backtest
    cerebro.run()
    cerebro.plot(style='candlestick')