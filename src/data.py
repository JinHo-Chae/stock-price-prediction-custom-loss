import numpy as np
import pandas as pd
import yfinance as yf
from .scaling import z_score_normalize

def download_stock(code: str, start: str, end: str | None = None):
    return yf.download(code, start=start, end=end)

def make_df(open_v, high_v, low_v, close_v):
    return pd.DataFrame({"Open": open_v, "High": high_v, "Low": low_v, "Close": close_v})

def normalize_train(df):
    open_v  = df["Open"].to_numpy()
    high_v  = df["High"].to_numpy()
    low_v   = df["Low"].to_numpy()
    close_v = df["Close"].to_numpy()
    vol_v   = df["Volume"].to_numpy()

    open_n,  open_m,  open_s  = z_score_normalize(open_v)
    high_n,  high_m,  high_s  = z_score_normalize(high_v)
    low_n,   low_m,   low_s   = z_score_normalize(low_v)
    close_n, close_m, close_s = z_score_normalize(close_v)
    vol_n,   vol_m,   vol_s   = z_score_normalize(vol_v)

    stats = {
        "open":   [open_m,  open_s],
        "high":   [high_m,  high_s],
        "low":    [low_m,   low_s],
        "close":  [close_m, close_s],
        "volume": [vol_m,   vol_s],
    }

    return (
        open_n.reshape(-1,),
        high_n.reshape(-1,),
        low_n.reshape(-1,),
        close_n.reshape(-1,),
        vol_n.reshape(-1,),
        stats
    )

def normalize_test(df, stats):
    open_m,  open_s  = stats["open"]
    high_m,  high_s  = stats["high"]
    low_m,   low_s   = stats["low"]
    close_m, close_s = stats["close"]
    vol_m,   vol_s   = stats["volume"]

    open_n  = (df["Open"].to_numpy()   - open_m)  / open_s
    high_n  = (df["High"].to_numpy()   - high_m)  / high_s
    low_n   = (df["Low"].to_numpy()    - low_m)   / low_s
    close_n = (df["Close"].to_numpy()  - close_m) / close_s
    vol_n   = (df["Volume"].to_numpy() - vol_m)   / vol_s

    open_n  = open_n.reshape(-1,)
    high_n  = high_n.reshape(-1,)
    low_n   = low_n.reshape(-1,)
    close_n = close_n.reshape(-1,)
    vol_n   = vol_n.reshape(-1,)

    return make_df(open_n, high_n, low_n, close_n), vol_n

def create_dataset(df, volume, window=10, horizon=4):
    X, y, X2 = [], [], []
    price_data = df[["Open","High","Low","Close"]].values
    vol_data = volume.reshape(-1, 1)

    for i in range(len(price_data) - window - horizon):
        X.append(price_data[i:i+window])
        y.append(price_data[i+window:i+window+horizon])
        X2.append(vol_data[i:i+window])

    return np.array(X), np.array(y), np.array(X2)

def create_dataset_test(df, volume, window=10, horizon=4):
    X, y, X2 = [], [], []
    price_data = df[["Open","High","Low","Close"]].values
    vol_data = volume.reshape(-1, 1)

    for i in range(len(price_data) - window):
        X.append(price_data[i:i+window])
        X2.append(vol_data[i:i+window])

    for i in range(len(price_data) - window - horizon):
        y.append(price_data[i+window:i+window+horizon])

    return np.array(X), np.array(y), np.array(X2)
