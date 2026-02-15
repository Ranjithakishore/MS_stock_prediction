"""Simple technical feature engineering for beginners."""
import pandas as pd


def add_sma(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return df["close"].rolling(window=window).mean()


def add_ema(df: pd.DataFrame, span: int = 14) -> pd.Series:
    return df["close"].ewm(span=span, adjust=False).mean()


def add_bollinger(df: pd.DataFrame, window: int = 20, num_std: int = 2) -> pd.DataFrame:
    ma = df["close"].rolling(window=window).mean()
    std = df["close"].rolling(window=window).std()
    df[f"bb_upper_{window}"] = ma + num_std * std
    df[f"bb_lower_{window}"] = ma - num_std * std
    return df


def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(window=window).mean()
    ma_down = down.rolling(window=window).mean()
    rs = ma_up / (ma_down + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sma_14"] = add_sma(df, 14)
    df["ema_14"] = add_ema(df, 14)
    df = add_bollinger(df, 20, 2)
    df["rsi_14"] = add_rsi(df, 14)
    return df
