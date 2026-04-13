import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from utils.data_pipeline import DataPipeline
from envs.trading_env import MultiAssetTradingEnv
import os

# --- PREMIUM STYLING ---
st.set_page_config(layout="wide", page_title="APEX RL Trading Dashboard", page_icon="📈")

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e3440,#2e3440);
        color: white;
    }
    h1, h2, h3 {
        color: #00ffcc !important;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #00ffcc;
        color: #0e1117;
        font-weight: bold;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00cca3;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ APEX RL: Portfolio Simulation Engine")
st.markdown("---")

# Sidebar Configuration
st.sidebar.header("🕹️ Simulation Control")
tickers = ['AAPL', 'MSFT', 'JPM', 'XOM', 'GLD', 'SPY', 'BTC-USD']
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))

# Model Discovery
model_files = [f for f in os.listdir("models") if f.endswith(".zip")] if os.path.exists("models") else []
selected_model = st.sidebar.selectbox("Select Agent Policy", model_files) if model_files else None

cost_pct = st.sidebar.slider("Transaction Cost (%)", 0.0, 1.0, 0.1) / 100
slip_pct = st.sidebar.slider("Slippage (%)", 0.0, 1.0, 0.05) / 100

if st.sidebar.button("🚀 Execute Simulation Replay"):
    with st.spinner("Initializing neural engine and market data..."):
        # 1. Pipeline Execution
        pipeline = DataPipeline(tickers, str(start_date), str(end_date))
        dfs = pipeline.download_data()
        
        # 2. Environment Initialization
        env = MultiAssetTradingEnv(dfs, window_size=10, transaction_fee=cost_pct, slippage_pct=slip_pct)
        
        # 3. Simulation Loop
        net_worth_history = []
        weights_history = []
        action_history = []
        benchmark_history = []
        
        obs, _ = env.reset()
        portfolio_val = env.net_worth
        
        # Baseline: Equal Weight
        initial_prices = np.array([dfs[t]['Close'].iloc[0] for t in tickers])
        
        model = PPO.load(f"models/{selected_model.replace('.zip', '')}") if selected_model else None
        
        done = False
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = np.ones(len(tickers) + 1) / (len(tickers) + 1)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            net_worth_history.append(info['net_worth'])
            weights_history.append(env.weights.copy())
            
            # Simple Benchmark (Mean Return)
            benchmark_val = np.mean([dfs[t]['Close'].iloc[env.current_step] / dfs[t]['Close'].iloc[0] for t in tickers]) * 100000
            benchmark_history.append(benchmark_val)
            
            done = terminated or truncated
            
        # --- DASHBOARD RENDERING ---
        
        # Row 1: Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        total_ret = ((net_worth_history[-1] / net_worth_history[0]) - 1) * 100
        bench_ret = ((benchmark_history[-1] / benchmark_history[0]) - 1) * 100
        
        m1.metric("Agent Alpha", f"{total_ret - bench_ret:+.2f}%", delta=f"{total_ret - bench_ret:.2f}%")
        m2.metric("Final Net Worth", f"${net_worth_history[-1]:,.0f}")
        m3.metric("Max Drawdown", f"{min(pd.Series(net_worth_history).pct_change().cumsum()):.2%}")
        m4.metric("Market Rel. Return", f"{bench_ret:.2f}%")

        # Row 2: Performance Comparison
        st.subheader("📊 Portfolio Alpha Trajectory")
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=net_worth_history, name='RL Agent (PPO)', line=dict(color='#00ffcc', width=3)))
        fig.add_trace(go.Scatter(y=benchmark_history, name='Equal Weight Baseline', line=dict(color='rgba(255,255,255,0.3)', dash='dash')))
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=0, b=0),
                          legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True)

        # Row 3: Allocation Dynamics
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("🧩 Adaptive Allocation Stream")
            weight_df = pd.DataFrame(weights_history, columns=tickers)
            fig_w = go.Figure()
            for t in tickers:
                fig_w.add_trace(go.Scatter(y=weight_df[t], name=t, stackgroup='one', line=dict(width=0)))
            fig_w.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_w, use_container_width=True)
            
        with c2:
            st.subheader("🎯 Current Strategy Mix")
            latest_weights = weights_history[-1]
            fig_pie = go.Figure(data=[go.Pie(labels=tickers, values=latest_weights, hole=.4)])
            fig_pie.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.success("Simulation session finalized. Insights derived from learned policy.")
else:
    st.info("👈 Configure parameters and press 'Execute Simulation Replay' to begin.")
