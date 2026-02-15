"""Preprocessing: fill, scale, and create LSTM sequences."""
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fill_and_dropna(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.interpolate().ffill().bfill()
    return df.dropna()


def scale_features(df: pd.DataFrame, feature_cols: list) -> Tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[feature_cols])
    return data, scaler


def create_sequences(data: np.ndarray, seq_len: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len : i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)
