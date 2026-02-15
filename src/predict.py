"""Simple rolling prediction for the next N days using saved model."""
import os
import pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import tensorflow as tf

from .features import add_all_features
from .preprocess import fill_and_dropna, scale_features


def load_msft(csv_path: str = "data/MicrosoftStock.csv") -> pd.DataFrame:
    """Load MSFT data from local CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def forecast_next_n(model_path: str, n: int = 30, seq_len: int = 60):
    df = load_msft()
    df = add_all_features(df)
    df = fill_and_dropna(df)

    feature_cols = [
        "close",
        "sma_14",
        "ema_14",
        "bb_upper_20",
        "bb_lower_20",
        "rsi_14",
        "volume",
    ]

    data, scaler = scale_features(df, feature_cols)

    model = tf.keras.models.load_model(model_path)
    
    # Load scaler for inverse transform
    with open("models/scaler.pkl", "rb") as f:
        saved_scaler = pickle.load(f)

    last_seq = data[-seq_len:]
    preds_scaled = []
    seq = last_seq.copy()
    for _ in range(n):
        x = seq.reshape((1, seq_len, seq.shape[1]))
        p = model.predict(x)[0, 0]
        preds_scaled.append(p)
        # shift and append predicted close (first feature)
        next_row = np.zeros(seq.shape[1])
        next_row[0] = p
        seq = np.vstack([seq[1:], next_row])

    # Inverse transform to get actual prices
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    # Create a dummy array with same shape as original features for inverse_transform
    dummy = np.zeros((len(preds_scaled), len(feature_cols)))
    dummy[:, 0] = preds_scaled.flatten()
    preds_actual = saved_scaler.inverse_transform(dummy)[:, 0]
    
    # Generate dates (business days starting from last date in df)
    last_date = df.index[-1]
    dates = []
    current = last_date
    for _ in range(n):
        current += timedelta(days=1)
        # Skip weekends (Saturday=5, Sunday=6)
        while current.weekday() >= 5:
            current += timedelta(days=1)
        dates.append(current)
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        "date": dates,
        "predicted_close": preds_actual,
    })
    
    return result_df


if __name__ == "__main__":
    print("Forecasting next 7 days (example)...")
    preds = forecast_next_n("models/msft_lstm", n=7)
    print(preds)
