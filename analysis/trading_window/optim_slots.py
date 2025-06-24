# trading_algorithm.py
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from datetime import datetime


class DataSplitter:
    @staticmethod
    def split_by_date(data: pd.DataFrame, date_col: str = 'date') -> list[pd.DataFrame]:
        """
        Splits a DataFrame into a list of daily DataFrames (grouped by date_col).
        """
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        groups = [grp.reset_index(drop=True) for _, grp in df.groupby(date_col, sort=True)]
        return groups
    
    @staticmethod
    def split_by_mmdd(data: pd.DataFrame, date_col='date') -> list[pd.DataFrame]:
        df = data.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df['MMDD'] = df[date_col].dt.strftime('%m-%d')
        groups = [grp.reset_index(drop=True) for _, grp in df.groupby('MMDD', sort=True)]
        return groups



class TradingAlgorithm:
    def decide(self, state: Dict) -> str:
        """ Given current market state, return 'buy', 'sell' or 'hold' """
        raise NotImplementedError

class RSIStrategy(TradingAlgorithm):
    def __init__(self, lower_threshold: float = 30, upper_threshold: float = 70):
        self.lower, self.upper = lower_threshold, upper_threshold

    def decide(self, state: Dict) -> str:
        rsi = state.get('RSI', 50)
        if rsi < self.lower:
            return 'buy'
        elif rsi > self.upper:
            return 'sell'
        else:
            return 'hold'


class MarketEnvironment:
    """
    Handles both:
      1) stepping/bar-by-bar through the (daily) DataFrame for trading Q‐learning 
         (step() / reset())
      2) slicing one entire day's worth of bars, for optimizing “time‐window vs. alpha/beta”
         via evaluate_time_window() + reset_daily().

    Key changes compared to before:
      • __init__() now initializes self.t, self.cash, self.inventory
      • reset() re‐sets t, cash, inventory in one call
      • reset_daily() now *sets* self.data = that single-day DataFrame
      • No more “missing attribute” complaints.
    """

    def __init__(self, data: pd.DataFrame, strategy: TradingAlgorithm):
        # Copy & reset index
        self.full_data = data.copy().reset_index(drop=True)

        # Ensure we have a 'date' column (just the date part of timestamp)
        if 'date' not in self.full_data.columns:
            self.full_data['date'] = pd.to_datetime(self.full_data['timestamp']).dt.date

        self.strategy = strategy
        self.unique_dates = sorted(self.full_data['date'].unique())


        # Compute log returns if not already present
        if 'log_return' not in self.full_data.columns:
            p = self.full_data['open']
            self.full_data['log_return'] = np.log(p / p.shift(1)).fillna(0)

        # Precompute min/max for volatility + volume (for evaluate_time_window)
        all_abs = self.full_data["log_return"].abs().dropna().values
        self._vola_min = all_abs.min()    # usually 0.0
        self._vola_max = all_abs.max()    # highest single‐bar abs‐return in entire set

        # lr_abs = self.full_data['log_return'].abs().dropna()
        # self._vola_min = float(lr_abs.min()) if not lr_abs.empty else 0.0
        # self._vola_max = float(lr_abs.max()) if not lr_abs.empty else 1.0

        if 'volume' in self.full_data.columns:
            vol_data = self.full_data['volume'].dropna()
            self._volb_min = float(vol_data.min()) if not vol_data.empty else 0.0
            self._volb_max = float(vol_data.max()) if not vol_data.empty else 1.0
        else:
            self._volb_min, self._volb_max = 0.0, 1.0

        # If all volumes were identical
        if self._volb_max == self._volb_min:
            self._volb_min, self._volb_max = 0.0, 1.0

        # INTERNAL pointers / states for step() / reset()
        self.data = self.full_data  # will switch to single‐day in reset_daily()
        self.t = 0                  # current bar index in self.data
        self.cash = 10000.0         # starting cash
        self.inventory = 0          # number of shares held

        # For reset_daily()
        self._daily_groups = None
        self._day_pointer = 0

        self.full_data["MMDD"] = self.full_data["timestamp"].dt.strftime("%m-%d")
        self.calendar_days = sorted(self.full_data["MMDD"].unique())
        self.mapping: Dict[str, List[pd.DataFrame]] = {}
        for mm in self.calendar_days:
            group = self.full_data[self.full_data["MMDD"] == mm]
            # group by actual date to separate years
            by_date = [g.reset_index(drop=True)
                       for _, g in group.groupby(group["date"], sort=False)]
            self.mapping[mm] = by_date

    def simulate_strategy_pl(self, day_df: pd.DataFrame, start: int, window: int) -> float:
        """
        Simulate `strategy` over rows [start:start+window] in day_df.
        Returns net P/L: +sum(log_return) on buys, -sum(log_return) on sells.
        """
        end = min(start + window, len(day_df))
        pnl = 0.0
        for i in range(start, end):
            state = {"RSI": day_df.at[i, "RSI"]}
            act = self.strategy.decide(state)
            lr = day_df.at[i, "log_return"]
            if act == "buy":
                pnl += lr
            elif act == "sell":
                pnl -= lr
        return pnl
    #def reset(self):
    #    """
    #    Entire‐environment reset for a new Q‐learning episode.
    #    Returns the initial state (dictionary).
    #    """
    #    self.data = self.full_data  # ensure we are back to the full DataFrame
    #    self.t = 0
    #    self.cash = 10000.0
    #    self.inventory = 0
    #    return self._state()
    #
    #def reset_all_data(self):
    #    self.data = self.full_data.copy()
    #    self.t = 0
    #    self.cash = 10000.0
    #    self.inventory = 0
#
    #def _state(self) -> dict:
    #    """
    #    Current‐bar state dictionary. Must match what your agent expects.
    #    """
    #    row = self.data.iloc[self.t]
    #    return {
    #        'price': float(row['open']),
    #        'log_return': float(row['log_return']),
    #        'volume': float(row.get('volume', 0.0)),
    #        'RSI': float(row.get('RSI', 50.0))
    #    }
#
    #def step(self, action: str) -> tuple[dict, float, bool]:
    #    """
    #    Execute one bar (“minute”/“5‐minute”/etc.) of trading.
    #    action ∈ {'buy','sell','hold'} or you may map your RL actions → these.
    #    Returns (next_state, reward, done_bool).
    #    """
    #    state = self._state()
    #    price = state['price']
    #    reward = 0.0
#
    #    if action == 'sell' and self.inventory > 0:
    #        self.inventory -= 1
    #        self.cash += price
    #        reward = price  # you get the proceeds as “reward”
#
    #    elif action == 'buy' and self.cash >= price:
    #        self.inventory += 1
    #        self.cash -= price
    #        reward = -price  # assume negative “reward” = spending cash
#
    #    # Otherwise 'hold' or invalid buy/sell → reward stays 0.0
#
    #    self.t += 1
    #    done = (self.t >= len(self.data))
    #    if not done:
    #        next_state = self._state()
    #    else:
    #        next_state = {}
    #    return next_state, float(reward), done

    def evaluate_time_window(self, start_minute: int, window_size: int,
                             alpha: float = 1.0, beta: float = 1.0, gamma: float = 1.0,
                             min_window_size: int = 15, max_window_size: int = 720) -> float:
        """
        Given a *full-day* self.data (use reset_daily() first!), look at bars [start_minute : start_minute+window_size),
        compute mean(|log_return|) and mean(volume), normalize each, then return alpha*vol_norm + beta*volu_norm.

        If out‐of‐bounds → return 0.0
        """
        end_minute = start_minute + window_size
        if start_minute < 0 or end_minute > len(self.data):
            return 0.0

        window = self.data.iloc[start_minute:end_minute]
        if window.empty:
            return 0.0

        # 1) mean absolute return over that slice
        abs_rets = window['log_return'].abs().dropna()

        raw_vol = np.sqrt((abs_rets**2).sum())
        vols = raw_vol / np.sqrt(window_size)
        #vols = float(abs_rets.mean()) if not abs_rets.empty else 0.0

        # 2) mean volume over that slice (if exists)
        vols_vol = float(window['volume'].mean()) if 'volume' in window.columns else 0.0

        # Normalize each
        if self._vola_max == self._vola_min:
            vols_norm = 0.0
        else:
            vols_norm = (vols - self._vola_min) / (self._vola_max - self._vola_min)

        if self._volb_max == self._volb_min:
            volu_norm = 0.5
        else:
            volu_norm = (vols_vol - self._volb_min) / (self._volb_max - self._volb_min)

        window_size_norm = (window_size - min_window_size) / (max_window_size - min_window_size) 

        return float(alpha * vols_norm + beta * volu_norm + gamma * window_size_norm)

    #def reset_daily(self) -> pd.DataFrame:
    #    """
    #    Return a *single‐day* DataFrame slice (grouped by date). Also sets self.data
    #    to that day’s bars so that step() and evaluate_time_window() operate on it.
    #    """
    #    if self._daily_groups is None:
    #        self._daily_groups = DataSplitter.split_by_date(self.full_data, date_col='date')
    #        self._day_pointer = 0
#
    #    day_df = self._daily_groups[self._day_pointer].copy().reset_index(drop=True)
    #    self._day_pointer += 1
    #    if self._day_pointer >= len(self._daily_groups):
    #        self._day_pointer = 0
#
    #    self.data = day_df  # switch “active” data to this one day’s bars
    #    self.t = 0          # start at minute‐0 of that day
    #    return day_df


class QLearningAgent:
    def __init__(self, n_states: int, alpha: float, gamma: float, epsilon: float):
        self.actions = ["trade", "skip"]
        self.Q = np.zeros((n_states, len(self.actions)), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, s_idx: int) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.actions))
        return int(np.argmax(self.Q[s_idx]))

    def update(self, s: int, a: int, r: float, s2: int, done: bool):
        target = r + (0.0 if done else self.gamma * np.max(self.Q[s2]))
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])



def train_q_learning_and_extract_dates(
    env: MarketEnvironment,
    agent: QLearningAgent,
    episodes: int,
    n_days: int,
    window_size: int,
    slots: List[int]
):
    calendar_days = env.calendar_days
    num_slots     = len(slots)
    n_states      = len(calendar_days) * num_slots

    for ep in range(episodes):
        # env.reset_all_data()
        state_indices = list(range(n_states))
        random.shuffle(state_indices)

        for s in state_indices:
            day_idx  = s // num_slots
            slot_idx = s % num_slots
            mm       = calendar_days[day_idx]
            day_df   = random.choice(env.mapping[mm])
            start    = slots[slot_idx]

            # simulate P/L
            pl = env.simulate_strategy_pl(day_df, start, window_size)

            # ε-greedy
            if np.random.rand() < agent.epsilon:
                a = np.random.randint(2)
            else:
                a = int(np.argmax(agent.Q[s]))

            # trade=reward, skip=0
            reward = pl if a == 0 else 0.0

            # Q update
            done = False
            target = reward + agent.gamma * np.max(agent.Q[s])  # self-loop
            agent.Q[s, a] += agent.alpha * (target - agent.Q[s, a])

        agent.epsilon *= 0.995

    # after training: pick top-n_days states by max‐Q
    ranked = sorted(
        [(agent.Q[s].max(), s) for s in range(n_states)],
        key=lambda x: x[0],
        reverse=True
    )[:n_days]

    best_calendar = []
    for _, s in ranked:
        d = s // num_slots
        sl= s % num_slots
        best_calendar.append((calendar_days[d], slots[sl]))

    return best_calendar
