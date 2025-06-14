# hyperparameter_optimizer.py
from typing import Callable
import optuna
import random
import pandas as pd
from trading_window.optim_slots import (
    DataSplitter,
    MarketEnvironment,
    RSIStrategy,
    QLearningAgent
)

DEFAULT_ALPHA = 1.0
DEFAULT_BETA  = 1.0

class TradingWindowOptimizer:
    """
    Encapsulates a Bayesian search over (n_days, window_size) for trading-window scoring.
    You can import this class and call `.optimize()` from a notebook.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        env_cls: Callable,
        strategy_cls: Callable,
        split_fn: Callable,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        min_window_size: int = 15,
        max_window_size: int = 720
    ):
        self.df = df.reset_index(drop=True).copy()
        self.env_cls = env_cls
        self.strategy_cls = strategy_cls
        self.split_fn = split_fn  # expected to return List[pd.DataFrame]
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size

    def _objective(self, trial: optuna.Trial) -> float:
        # suggest number of days (1--252) and window size in minutes (1--720)
        n_days = trial.suggest_int("n_days", 1, 252)
        window_size = trial.suggest_int("window_size", self.min_window_size, self.max_window_size)

        # split data by date into daily DataFrames
        days = self.split_fn(self.df)
        # ensure it's a list
        if not isinstance(days, list):
            try:
                days = list(days)
            except:
                return 0.0

        if len(days) < n_days:
            raise optuna.TrialPruned()

        # pick the first n_days (or sample randomly)
        selected_days = random.sample(days, k=n_days)
        scores = []
        for day_df in selected_days:
            env = self.env_cls(day_df, self.strategy_cls())
            # we evaluate all possible start-minute slots for that day
            max_start = len(day_df) - window_size + 1
            if max_start < 1:
                continue
            slots = list(range(0, max_start))
            day_scores = [env.evaluate_time_window(s, window_size, self.alpha, self.beta, self.gamma,
                                                   self.min_window_size, self.max_window_size) for s in slots]
            if day_scores:
                scores.append(max(day_scores))

        return float(pd.Series(scores).mean()) if scores else 0.0

    def optimize(self, n_trials: int = 50, 
                 study_name: str = None,
                 storage_name: str = None,) -> optuna.Study:
        """
        Run the Optuna study, searching for the best (n_days, window_size).
    
        Parameters:
        -----------
        n_trials : int
            How many trials to run.
        study_name : str | None
            Name for the study (so you can load it later). If None, an unnamed in-memory study.
        storage_name : str | None
            URL to an RDB storage, e.g. "sqlite:///example.db". If None, runs in memory.
    
        Returns:
        --------
        study : optuna.Study
            The completed study; .best_params and .best_value are set.
        """
        # Create or load an RDB-backed study if storage_name provided:
        if storage_name:
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_name,
                direction="maximize",
                load_if_exists=True,
            )
        else:
            # pure in-memory study
            study = optuna.create_study(
                study_name=study_name,
                direction="maximize",
            )
    
        study.optimize(self._objective, n_trials=n_trials)
        return study