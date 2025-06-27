import os
import pandas as pd
import matplotlib.pyplot as plt


def load_trades(csv_path: str) -> pd.DataFrame:
    """Load trades from a CSV file."""
    return pd.read_csv(
        csv_path,
        parse_dates=["entry_datetime", "exit_datetime"],
    )

def load_orders(csv_path: str) -> pd.DataFrame:
    """Load orders from a CSV file."""
    df = pd.read_csv(csv_path)
    print(df.head())  # Debugging: print the first few rows of the DataFrame
    df.columns = df.columns.str.strip()  # Ensure no whitespace issues
    if "datetime" not in df.columns:
        raise KeyError(f"Missing 'datetime' column. Found columns: {df.columns.tolist()}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


# def plot_cumulative_pnl(df: pd.DataFrame, out_path: str) -> None:
#     """Plot cumulative PnL over time."""
#     df = df.sort_values("exit_datetime")
#     df["cumulative_pnl"] = df["pnl"].cumsum()
# 
#     plt.figure(figsize=(10, 6))
#     plt.plot(df["exit_datetime"], df["cumulative_pnl"], marker="o")
#     plt.title("Cumulative PnL Over Time")
#     plt.xlabel("Exit Time")
#     plt.ylabel("Cumulative PnL")
#     plt.grid(True)
#     plt.gcf().autofmt_xdate()
#     plt.tight_layout()
# 
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.savefig(out_path)
#     plt.close()


def plot_pnl_distribution(trades: pd.DataFrame, save_path: str) -> None:
    # Clip extreme values to focus on central mass
    clipped_pnl = trades["pnl"].clip(lower=trades["pnl"].quantile(0.01),
                                     upper=trades["pnl"].quantile(0.99))
    
    plt.figure(figsize=(10, 5))
    plt.hist(clipped_pnl, bins=100, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Trade PnL (1st–99th Percentile)")
    plt.xlabel("PnL")
    plt.ylabel("Number of Trades")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

#def plot_capital_flow(
#    df: pd.DataFrame, out_path: str, initial_capital: float = 10_000
#) -> None:
#    """Plot account value over time based on trade PnL."""
#    df = df.sort_values("exit_datetime")
#    df["capital"] = initial_capital + df["pnl"].cumsum()
#
#    plt.figure(figsize=(10, 6))
#    plt.plot(df["exit_datetime"], df["capital"], marker="o")
#    plt.title("Capital Flow Over Time")
#    plt.xlabel("Exit Time")
#    plt.ylabel("Account Value")
#    plt.grid(True)
#    plt.gcf().autofmt_xdate()
#    plt.tight_layout()
#
#    os.makedirs(os.path.dirname(out_path), exist_ok=True)
#    plt.savefig(out_path)
#    plt.close()


def plot_model_predictions(
    pred_path: str, actual_path: str, out_path: str
) -> None:
    """Plot model predictions against actual values if data exists."""
    if not os.path.exists(pred_path) or not os.path.exists(actual_path):
        print("Prediction or actual CSV not found, skipping model prediction plot.")
        return

    y_pred = pd.read_csv(pred_path, header=None).squeeze()
    y_true = pd.read_csv(actual_path, header=None).squeeze()

    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Model Predictions vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_cumulative_pnl(trades: pd.DataFrame, save_path: str) -> None:
    trades["cumulative_pnl"] = trades["pnl"].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(trades["exit_datetime"], trades["cumulative_pnl"], marker='o')
    plt.title("Cumulative PnL Over Time")
    plt.xlabel("Exit Time")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_pnl_vs_duration(df: pd.DataFrame, out_path: str) -> None:
    """Plot trade PnL versus trade duration."""
    df = df.sort_values("exit_datetime")
    df["duration_minutes"] = (
        df["exit_datetime"] - df["entry_datetime"]
    ).dt.total_seconds() / 60

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df["duration_minutes"],
        df["pnl"],
        c=df["pnl"].cumsum(),
        cmap="coolwarm",
        edgecolor="black",
    )
    plt.title("Trade PnL vs Duration")
    plt.xlabel("Duration (minutes)")
    plt.ylabel("PnL")
    plt.grid(True)
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cumulative PnL")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_capital_flow(orders: pd.DataFrame, save_path: str) -> None:
    orders["net_position"] = orders["size"].cumsum()
    plt.figure(figsize=(10, 5))
    plt.plot(orders["datetime"], orders["net_position"], color="blue")
    plt.title("Capital Flow: Net Position Over Time")
    plt.xlabel("Time")
    plt.ylabel("Net Position")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_pnl_distribution(trades: pd.DataFrame, save_path: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(trades["pnl"], bins=30, edgecolor='black', alpha=0.7)
    plt.title("Distribution of Trade PnL")
    plt.xlabel("PnL")
    plt.ylabel("Number of Trades")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_avg_win_vs_loss(trades: pd.DataFrame, save_path: str) -> None:
    wins = trades[trades["pnl"] > 0]
    losses = trades[trades["pnl"] < 0]

    avg_win = wins["pnl"].mean() if not wins.empty else 0.0
    avg_loss = abs(losses["pnl"].mean()) if not losses.empty else 0.0

    plt.figure(figsize=(6, 5))
    bars = plt.bar(["Average Win", "Average Loss"], [avg_win, avg_loss], color=["green", "red"])
    plt.title("Average Win vs Average Loss")
    plt.ylabel("PnL (Absolute)")
    plt.grid(True, axis='y')

    # Add value annotations
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    csv_path_trades = os.path.join("analysis", "results", "trades.csv")
    csv_path_orders = os.path.join("analysis", "results", "orders.csv")

    trades = load_trades(csv_path_trades)
    orders = load_orders(csv_path_orders)

    plots_dir = os.path.join("analysis", "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_cumulative_pnl(trades, os.path.join(plots_dir, "cumulative_pnl.png"))
    plot_pnl_vs_duration(trades, os.path.join(plots_dir, "pnl_vs_duration.png"))
    plot_capital_flow(orders, os.path.join(plots_dir, "capital_flow.png"))
    plot_pnl_distribution(trades, os.path.join(plots_dir, "pnl_distribution.png"))
    plot_avg_win_vs_loss(trades, os.path.join(plots_dir, "avg_win_vs_loss.png"))

if __name__ == "__main__":
    main()


def main() -> None:
    csv_path = os.path.join("analysis", "results", "trades.csv")
    trades = load_trades(csv_path)
    orders = load_orders(csv_path.replace("trades", "orders"))

    plots_dir = os.path.join("analysis", "plots")
    plot_cumulative_pnl(trades, os.path.join(plots_dir, "cumulative_pnl.png"))
    plot_pnl_vs_duration(trades, os.path.join(plots_dir, "pnl_vs_duration.png"))
    plot_capital_flow(orders, os.path.join(plots_dir, "capital_flow.png"))

    pred_path = os.path.join("data", "processed", "y_pred.csv")
    actual_path = os.path.join("data", "processed", "y_test.csv")
    plot_model_predictions(
        pred_path,
        actual_path,
        os.path.join(plots_dir, "model_predictions.png"),
    )


if __name__ == "__main__":
    main()