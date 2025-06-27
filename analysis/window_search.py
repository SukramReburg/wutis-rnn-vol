import sys, os

import pandas as pd
import yaml
from IPython.display import display
from pathlib import Path
import numpy as np
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour
import plotly.io as pio


import matplotlib.pyplot as plt
from trading_window.optim_slots import (
    DataSplitter,
    MarketEnvironment,
    RSIStrategy,
    QLearningAgent,
    train_q_learning_and_extract_dates 
)
from trading_window.finding_slots import TradingWindowOptimizer, DEFAULT_ALPHA, DEFAULT_BETA

if __name__ == "__main__":
    
    PROJECT_ROOT = Path.cwd()  
    # make sure the folder exists
    optuna_dir = PROJECT_ROOT / "studies"
    optuna_dir.mkdir(exist_ok=True)

    db_path = optuna_dir / "window_search.sqlite3"
    storage_name = f"sqlite:///{db_path}"

    study_name = "SPY_trading_window_optimization_70_18_12"

    config_path = 'config/data_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths = config['paths']
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, paths['raw'], f"SPY.csv")

    # Read the CSV file
    df = pd.read_csv(path, parse_dates=['timestamp'])

    # …then, assuming you’ve already loaded your full-minute DataFrame `df`:
    df['date'] = pd.to_datetime(df['timestamp']).dt.date

    # from hyperparameter_optimizer import TradingWindowOptimizer
    optimizer = TradingWindowOptimizer(
        df,
        MarketEnvironment,
        RSIStrategy,
        DataSplitter.split_by_date,
        alpha = 0.7, # Volatility
        beta = 0.18, # Volume
        gamma = 0.12, # Profit
        min_window_size=15,  # Minimum window size
        max_window_size=720  # Maximum window size 
    )
    study = optimizer.optimize(n_trials=100, 
                            study_name = study_name, 
                            storage_name = storage_name)

    print(study.best_params)


    study = optuna.load_study(
    study_name=study_name, storage=storage_name
    )

    output_path = config['paths']['plots']
    os.makedirs(output_path, exist_ok=True)

    # 1) Optimization history
    fig1 = plot_optimization_history(study)
    path1 = os.path.join(output_path, 'opt_history.png')
    fig1.write_image(path1)  

    # 2) Hyperparameter importances
    fig2 = plot_param_importances(study)
    path2 = os.path.join(output_path, 'opt_importances.png')
    fig2.write_image(path2)

    # 3) Contour plot of parameters
    fig3 = plot_contour(study, params=["n_days", "window_size"])
    path3 = os.path.join(output_path, 'opt_contour.png')
    fig3.write_image(path3)
# 
    # one_day = DataSplitter.split_by_date(df)[0]
    # best_n = 107
    # best_w = 24
# 
    # bars_per_day = len(one_day)
    # slots = list(range(0, bars_per_day, best_w))
# 
    # # 4) Stage 2: Q‐learning to pick EXACTLY best_n days + slots
    # env   = MarketEnvironment(df, RSIStrategy())
    # n_states = len(env.calendar_days) * len(slots)
    # agent = QLearningAgent(n_states=n_states, alpha=0.1, gamma=0.99, epsilon=1.0)

    # chosen_calendar = train_q_learning_and_extract_dates( # does not work because of default RSI strategy not in the dataset
    #     env         = env,
    #     agent       = agent,
    #     episodes    = 500,
    #     n_days      = best_n,
    #     window_size = best_w,
    #     slots       = slots
    # )

    # print("Learned (MM-DD → start_minute):")
    # for day, start in chosen_calendar:
    #     print(f"  {day} @ minute {start}")
