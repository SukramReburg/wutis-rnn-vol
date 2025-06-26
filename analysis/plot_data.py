import os
import pandas as pd
import seaborn as sns
import yaml 
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_corr():
    """
    Load time-series data from CSV files in a specified folder,
    compute the correlation matrix, and plot a heatmap.
    """     


    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_folder = config['paths']['raw']
    output_path = config['paths']['plots']

    os.makedirs(output_path, exist_ok=True)

    # Load all CSV files in the folder and concatenate them
    data_frames = []
    tickers = config['tickers']

    for ticker in tickers:
        file_name = f"{ticker}.csv"
        file_path = os.path.join(data_folder, file_name)
        df = pd.read_csv(file_path).drop(columns=['timestamp','symbol'], errors='ignore')  # Drop timestamp if it exists

        # Compute the correlation matrix
        correlation_matrix = df.corr()

        # Plot the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
        plt.title(f"Correlation Heatmap of {ticker} Data")
        plt.tight_layout()

        # Save the figure
        path = os.path.join(output_path, f'{ticker}_correlation_heatmap.png')
        plt.savefig(path)
        plt.close()

        print(f"Correlation heatmap of {ticker} saved to {output_path}")

def plot_spx_dist(): 
    """
    Load SPX data from a CSV file, compute the distribution of returns,
    and plot a histogram.
    """
    with open('config/data_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    data_folder = config['paths']['raw']
    output_path = config['paths']['plots']

    os.makedirs(output_path, exist_ok=True)

    spx_file = os.path.join(data_folder, "SPY.csv")
    df_spx = pd.read_csv(spx_file)
    
    # Calculate daily returns
    df_spx['returns'] = df_spx['close'].pct_change().dropna()

    returns = df_spx['returns']
    p01, p99 = np.percentile(returns, [1, 99])
    returns_clipped = returns.clip(p01, p99)

    # df_sample = df_spx.sample(n=10000, random_state=42)
    
    # Plot the distribution of returns
    plt.figure(figsize=(10, 6))

    sns.histplot(returns_clipped, bins=100, kde=True, stat='density')
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(returns.min(), returns.max(), 2000)
    plt.plot(x, norm.pdf(x, mu, sigma), label='Normal Distribution', color='red')

    
    # sns.histplot(df_sample['returns'], bins=50, kde=True)
    plt.xlim(-0.02, 0.02)

    plt.title('SPY Returns Distribution')
    plt.xlabel("Returns")
    plt.ylabel("Frequency")

    plt.show()


if __name__ == "__main__":
    sns.set(style="whitegrid")
    # plot_corr()
    plot_spx_dist()

