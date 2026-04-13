import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class MultiAssetTradingEnv(gym.Env):
    """
    Simulation-grade Multi-Asset Trading Environment.
    Supports 5 reward variants and realistic simulation of slippage/impact.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, ticker_dfs, initial_balance=100000, transaction_fee=0.001, 
                 slippage_pct=0.0005, window_size=10, reward_variant='A'):
        super().__init__()
        
        self.ticker_dfs = ticker_dfs
        self.tickers = list(ticker_dfs.keys())
        self.n_assets = len(self.tickers)
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.slippage_pct = slippage_pct
        self.window_size = window_size
        self.reward_variant = reward_variant
        
        # Determine episode length (min length across assets)
        self.max_steps = min([len(df) for df in ticker_dfs.values()]) - 1
        
        # Action space: Portfolio weights for each asset. 
        # Range [0, 1] for long-only, or [-1, 1] for shorting. 
        # We will normalize these to sum to 1 inside step().
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32) # +1 for Cash

        # Observation space: 
        # window_size * (OHLCV + Indicators) for each asset + Portfolio State
        # Portfolio State: Balance, Weights, Current PnL
        sample_df = next(iter(ticker_dfs.values()))
        self.n_features = len(sample_df.columns)
        
        obs_shape = (self.window_size * self.n_assets * self.n_features) + (self.n_assets + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        self.reset()

    def _get_obs(self):
        obs_list = []
        for ticker in self.tickers:
            df = self.ticker_dfs[ticker]
            # Get raw window
            window = df.iloc[self.current_step - self.window_size + 1 : self.current_step + 1].values
            
            # Simple Z-score like normalization per window
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0) + 1e-6
            norm_window = (window - mean) / std
            
            obs_list.append(norm_window.flatten())
        
        # Normalize portfolio state
        safe_net_worth = max(self.net_worth, 1.0)
        portfolio_state = np.concatenate([self.weights, [self.balance / safe_net_worth]])
        
        obs = np.concatenate(obs_list + [portfolio_state])
        # Final safety clip and NaN check
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size - 1
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.weights = np.zeros(self.n_assets)
        self.asset_holdings = np.zeros(self.n_assets)
        self.max_net_worth = self.initial_balance
        self.returns_history = []
        
        return self._get_obs(), {}

    def step(self, action):
        # Stable softmax
        exp_action = np.exp(action - np.max(action))
        weights = exp_action / np.sum(exp_action)
        target_asset_weights = weights[:-1]
        target_cash_weight = weights[-1]
        
        # Get current prices
        current_prices = np.array([self.ticker_dfs[t]['Close'].iloc[self.current_step] for t in self.tickers])
        next_prices = np.array([self.ticker_dfs[t]['Close'].iloc[self.current_step + 1] for t in self.tickers])
        
        # Rebalance Logic
        old_net_worth = self.net_worth
        
        # 1. Calculate how much we need to trade
        target_values = target_asset_weights * self.net_worth
        current_holding_values = self.asset_holdings * current_prices
        trade_amounts = target_values - current_holding_values
        
        # 2. Deduct costs (Transaction fee + Slippage)
        costs = np.sum(np.abs(trade_amounts) * (self.transaction_fee + self.slippage_pct))
        self.balance -= costs
        
        # 3. Update holdings
        self.asset_holdings = target_values / current_prices
        self.weights = target_asset_weights
        
        # 4. Advance price and calculate new net worth
        new_holding_values = self.asset_holdings * next_prices
        self.net_worth = np.sum(new_holding_values) + (self.balance * (1.0)) # Cash return is 0
        
        # Calculate log return for reward
        step_return = (self.net_worth - old_net_worth) / old_net_worth
        self.returns_history.append(step_return)
        
        # 5. Reward Calculation
        reward = self._calculate_reward()
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        if self.net_worth <= self.initial_balance * 0.1: # Bankrupt
            terminated = True
            reward -= 10
            
        return self._get_obs(), float(reward), terminated, truncated, {"net_worth": self.net_worth}

    def _calculate_reward(self):
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array(self.returns_history)
        
        if self.reward_variant == 'A': # Simple Return
            return returns[-1]
        
        elif self.reward_variant == 'B': # Sharpe-inspired
            mean_ret = np.mean(returns[-20:])
            std_ret = np.std(returns[-20:]) + 1e-6
            return mean_ret / std_ret
        
        elif self.reward_variant == 'C': # Differential Sharpe (simplified)
            # Placeholder for proper Moody formula
            return returns[-1] / (np.std(returns) + 1e-6)
            
        elif self.reward_variant == 'D': # Sortino-penalized
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-6
            return np.mean(returns[-20:]) / downside_std
            
        elif self.reward_variant == 'E': # Max Drawdown penalty
            self.max_net_worth = max(self.max_net_worth, self.net_worth)
            mdd = (self.max_net_worth - self.net_worth) / self.max_net_worth
            return returns[-1] - (0.5 * mdd)
        
        return returns[-1]
