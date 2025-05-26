import os
import pandas as pd
import seaborn as sns
import yaml 
import matplotlib.pyplot as plt

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

    
if __name__ == "__main__":
    sns.set(style="whitegrid")
    plot_corr()

