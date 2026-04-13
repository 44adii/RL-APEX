import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO, SAC
from utils.data_pipeline import DataPipeline
from envs.trading_env import MultiAssetTradingEnv

class Evaluator:
    def __init__(self, tickers, test_dfs):
        self.tickers = tickers
        self.test_dfs = test_dfs
        self.env = MultiAssetTradingEnv(test_dfs, window_size=10)

    def calculate_metrics(self, net_worth_history):
        returns = pd.Series(net_worth_history).pct_change().dropna()
        
        # Annualization factors (assuming 252 trading days)
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / (ann_vol + 1e-6)
        
        # Drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        max_drawdown = drawdown.min()
        
        return {
            "Total Return": (net_worth_history[-1] / net_worth_history[0]) - 1,
            "Ann. Return": ann_return,
            "Ann. Vol": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_drawdown
        }

    def run_agent(self, model_path):
        model = PPO.load(model_path)
        obs, _ = self.env.reset()
        history = [self.env.net_worth]
        
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            history.append(info['net_worth'])
            done = terminated or truncated
            
        return history

    def run_buy_and_hold(self):
        # Equal weight buy and hold
        obs, _ = self.env.reset()
        n_assets = len(self.tickers)
        # Constant action: Equal weights once, then hold (by passing same weight)
        action = np.ones(n_assets + 1) / (n_assets + 1)
        
        history = [self.env.net_worth]
        done = False
        while not done:
            obs, reward, terminated, truncated, info = self.env.step(action)
            history.append(info['net_worth'])
            done = terminated or truncated
        return history

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'GLD', 'SPY', 'BTC-USD']
    pipeline = DataPipeline(tickers, '2015-01-01', '2024-12-31')
    raw_dfs = pipeline.download_data()
    _, _, test_dfs = pipeline.split_data(raw_dfs)
    
    evaluator = Evaluator(tickers, test_dfs)
    
    print("\n--- Evaluating Baselines ---")
    bh_history = evaluator.run_buy_and_hold()
    bh_metrics = evaluator.calculate_metrics(bh_history)
    print(f"Buy & Hold Metrics: {bh_metrics}")
    
    # Example agent loading (requires a trained model)
    # agent_history = evaluator.run_agent("models/ppo_A")
    # agent_metrics = evaluator.calculate_metrics(agent_history)
    # print(f"Agent Metrics: {agent_metrics}")
