import yfinance as yf
import pandas as pd
import numpy as np
import os

def calculate_indicators(df):
    """
    Manually calculate technical indicators to avoid dependency issues.
    """
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['std20'] = df['Close'].rolling(window=20).std()
    df['UpperBB'] = df['MA20'] + (df['std20'] * 2)
    df['LowerBB'] = df['MA20'] - (df['std20'] * 2)
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Fill NAs
    df.fillna(method='bfill', inplace=True)
    return df

class DataPipeline:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        
    def download_data(self):
        print(f"Downloading data for {self.tickers}...")
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date)
        
        # Handle multi-index columns if multiple tickers
        processed_dfs = {}
        
        if len(self.tickers) > 1:
            for ticker in self.tickers:
                # Extract columns for this ticker
                ticker_df = pd.DataFrame(index=data.index)
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    ticker_df[col] = data[col][ticker]
                
                ticker_df = calculate_indicators(ticker_df)
                processed_dfs[ticker] = ticker_df
        else:
            ticker = self.tickers[0]
            ticker_df = data.copy()
            ticker_df = calculate_indicators(ticker_df)
            processed_dfs[ticker] = ticker_df
            
        return processed_dfs

    def split_data(self, processed_dfs, train_pct=0.7, val_pct=0.15):
        train_data = {}
        val_data = {}
        test_data = {}
        
        for ticker, df in processed_dfs.items():
            n = len(df)
            train_end = int(n * train_pct)
            val_end = int(n * (train_pct + val_pct))
            
            train_data[ticker] = df.iloc[:train_end]
            val_data[ticker] = df.iloc[train_end:val_end]
            test_data[ticker] = df.iloc[val_end:]
            
        return train_data, val_data, test_data

if __name__ == "__main__":
    tickers = ['AAPL', 'MSFT', 'BTC-USD']
    pipeline = DataPipeline(tickers, '2015-01-01', '2024-12-31')
    dfs = pipeline.download_data()
    train, val, test = pipeline.split_data(dfs)
    print(f"Data split: Train={len(train['AAPL'])}, Val={len(val['AAPL'])}, Test={len(test['AAPL'])}")
