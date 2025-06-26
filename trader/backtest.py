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
        self.trade_history = []
        self.order_history = []  # Track every executed order
        self.open_trades = []  # Track all open trades (both long and short)

    def notify_order(self, order):
        if order.status != order.Completed:
            return

        dt = bt.num2date(order.executed.dt)
        price = order.executed.price
        size = order.executed.size

        self.order_history.append({
            "datetime": dt,
            "price": price,
            "size": size,
            "direction": "buy" if order.isbuy() else "sell",
        })

        abs_size = abs(size)

        if order.isbuy():
            remaining = abs_size
            closed = []
            for pos in self.open_trades:
                if pos["dir"] != -1:
                    continue
                close_qty = min(remaining, pos["size"])
                self.trade_history.append({
                    "entry_datetime": pos["entry_datetime"],
                    "entry_price": pos["entry_price"],
                    "exit_datetime": dt,
                    "exit_price": price,
                    "size": close_qty,
                    "dir": -1,
                    "pnl": (pos["entry_price"] - price) * close_qty,
                })
                pos["size"] -= close_qty
                remaining -= close_qty
                if pos["size"] == 0:
                    closed.append(pos)
                if remaining == 0:
                    break
            for pos in closed:
                self.open_trades.remove(pos)

            if remaining > 0:
                self.open_trades.append({
                    "order_ref": order.ref,
                    "entry_datetime": dt,
                    "entry_price": price,
                    "size": remaining,
                    "dir": 1,
                })

        elif order.issell():
            remaining = abs_size
            closed = []
            for pos in self.open_trades:
                if pos["dir"] != 1:
                    continue
                close_qty = min(remaining, pos["size"])
                self.trade_history.append({
                    "entry_datetime": pos["entry_datetime"],
                    "entry_price": pos["entry_price"],
                    "exit_datetime": dt,
                    "exit_price": price,
                    "size": close_qty,
                    "dir": 1,
                    "pnl": (price - pos["entry_price"]) * close_qty,
                })
                pos["size"] -= close_qty
                remaining -= close_qty
                if pos["size"] == 0:
                    closed.append(pos)
                if remaining == 0:
                    break
            for pos in closed:
                self.open_trades.remove(pos)

            if remaining > 0:
                self.open_trades.append({
                    "order_ref": order.ref,
                    "entry_datetime": dt,
                    "entry_price": price,
                    "size": remaining,
                    "dir": -1,
                })


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
    X_test = np.squeeze(X_test, axis=-1)
    y_test = np.load(os.path.join(processed_dir, "y_test.npy"))
    print('Test Dataset shape:',X_test.shape, y_test.shape)

    scaler_path = os.path.join(processed_dir, "scalers", "y_scaler.joblib")
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = joblib.load(f)
            
    else:
        scaler = None
    print("Scaler loaded:", scaler)

    feature_scalers_path = os.path.join(processed_dir, "scalers", "feature_scalers.joblib")
    if os.path.exists(feature_scalers_path):
        with open(feature_scalers_path, "rb") as f:
            feature_scalers = joblib.load(f)
    else:
        feature_scalers = None

    return X_test, y_test, scaler, feature_scalers

def reverse_sliding_window(X_test, sequence_length):
    """
    Reconstruct the original data array from overlapping sliding windows used in LSTM preprocessing.

    Parameters:
    - X_test: np.ndarray, shape (num_windows, sequence_length, num_features)
    - sequence_length: int, length of each sequence window

    Returns:
    - reconstructed: np.ndarray, shape (original_length, num_features)
    """
    num_windows, _, num_features = X_test.shape
    original_length = num_windows + sequence_length - 1

    reconstructed = np.zeros((original_length, num_features))
    counts = np.zeros((original_length, 1))

    for i in range(num_windows):
        reconstructed[i:i+sequence_length] += X_test[i]
        counts[i:i+sequence_length] += 1

    # Avoid division by zero
    counts[counts == 0] = 1
    reconstructed /= counts

    return reconstructed

if __name__ == "__main__":

    with open("config/data_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, config["paths"]["processed"])
    raw_dir = os.path.join(base_dir, config["paths"]["raw"])
    result_dir = os.path.join(base_dir, config["paths"]["results"])

    X_test, y_test, y_scaler,feature_scalers = load_data(processed_dir)
    # print("X_test shape:", X_test.shape)

    if y_scaler is not None:
        y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    if feature_scalers is not None:
        for i, scaler in enumerate(feature_scalers):
            X_test[:,:, i] = scaler.inverse_transform(X_test[:,:, i])

    # Just for testing purposes, we will use the last 7 minutes of data
    n_bars = 2*2*60
    window_size = X_test.shape[1]
    n_windows = n_bars - window_size + 1
    X_subset = X_test[-n_windows:]
    y_subset = y_test[-n_windows:]

    # data = reverse_sliding_window(X_test, X_test.shape[1])[:, 15:22]
    data = reverse_sliding_window(X_subset, window_size)[:, 15:22]

    columns = ['open','high','low','close','volume','trade_count','vwap']
    data = pd.DataFrame(data, columns=columns)
    print("Data shape:", data.shape)
    print("Data:\n", data.head())

    print("X_test shape:", X_subset.shape)

    df = data.copy()    
    df['datetime'] = pd.date_range(start='2024-11-01', periods=len(df), freq='1min')
    df.set_index('datetime', inplace=True)

    data = bt.feeds.PandasData(dataname=df, datetime=None, openinterest=None, fromdate=None, todate=None)   

    # Initialize TradingModel
    trading_model = TestTradingModel()

    # Initialize backtrader
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(BacktestModel, trading_model=trading_model, X_test=X_subset, y_test=y_subset, scaler=y_scaler, threshold_percent=0.04)
    # cerebro.addstrategy(BacktestModel, trading_model=trading_model, X_test=X_test, y_test=y_test, scaler=y_scaler, threshold_percent=0.04)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    #Run backtest
    result = cerebro.run()
    print("Backtest completed.")
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    print("Number of trades executed:", len(cerebro.broker.orders))
    print("Final cash:", cerebro.broker.getcash())
    print("Results:", result)

    # Extract results from analyzers
    analysis = result[0].analyzers
    trade_stats = analysis.trades.get_analysis()
    drawdown = analysis.drawdown.get_analysis()
    sharpe = analysis.sharpe.get_analysis()
    returns = analysis.returns.get_analysis()

    total_net_profit = trade_stats.get('pnl', {}).get('net', {}).get('total', 0.0)
    total_wins = trade_stats.get('won', {}).get('total', 0)
    total_closed = trade_stats.get('total', {}).get('closed', 0)
    win_ratio = (total_wins / total_closed) if total_closed else 0.0
    max_dd_percent = drawdown.get('max', {}).get('drawdown', 0.0)
    sharpe_ratio = sharpe.get('sharperatio', 0.0)
    if sharpe_ratio is None:
        sharpe_ratio = 0.0

    print("Total Net Profit:", total_net_profit)
    print("Win Ratio:", win_ratio)
    print("Max Drawdown %:", max_dd_percent)
    print("Sharpe Ratio:", sharpe_ratio)

    # Optionally store metrics for later comparison
    metrics = {
        'total_net_profit': float(total_net_profit),
        'win_ratio': float(win_ratio),
        'max_drawdown_percent': float(max_dd_percent),
        'sharpe_ratio': float(sharpe_ratio)
    }

    metrics_path = os.path.join(result_dir, 'backtest_metrics.yaml')
    with open(metrics_path, 'w') as f:
        yaml.safe_dump(metrics, f)
    
    # Save trade history and order history to CSV
    trade_df = pd.DataFrame(result[0].trade_history)
    order_df = pd.DataFrame(result[0].order_history)

    print("Trade History DataFrame:\n", trade_df.head())
    if not trade_df.empty:
        trade_path = os.path.join(result_dir, 'trades.csv')
        trade_df.to_csv(trade_path, index=False)

    if not order_df.empty:
        order_path = os.path.join(result_dir, 'orders.csv')
        order_df.to_csv(order_path, index=False)

    # Plot results
    cerebro.plot(style='candlestick')