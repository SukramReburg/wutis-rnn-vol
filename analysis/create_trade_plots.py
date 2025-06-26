import os
import pandas as pd
import matplotlib.pyplot as plt


def load_trades(csv_path: str) -> pd.DataFrame:
    """Load trades from a CSV file."""
    return pd.read_csv(
        csv_path,
        parse_dates=["entry_datetime", "exit_datetime"],
    )


def plot_cumulative_pnl(df: pd.DataFrame, out_path: str) -> None:
    """Plot cumulative PnL over time."""
    df = df.sort_values("exit_datetime")
    df["cumulative_pnl"] = df["pnl"].cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(df["exit_datetime"], df["cumulative_pnl"], marker="o")
    plt.title("Cumulative PnL Over Time")
    plt.xlabel("Exit Time")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.gcf().autofmt_xdate()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
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


def main() -> None:
    csv_path = os.path.join("analysis", "results", "trades.csv")
    trades = load_trades(csv_path)
    plots_dir = os.path.join("analysis", "plots")
    plot_cumulative_pnl(trades, os.path.join(plots_dir, "cumulative_pnl.png"))
    plot_pnl_vs_duration(trades, os.path.join(plots_dir, "pnl_vs_duration.png"))


if __name__ == "__main__":
    main()
