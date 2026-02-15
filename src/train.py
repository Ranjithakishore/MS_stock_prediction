"""Train pipeline: load data, engineer features, train LSTM, evaluate."""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .features import add_all_features
from .model import build_lstm
from .preprocess import create_sequences, fill_and_dropna, scale_features


def load_msft(csv_path: str = "data/MicrosoftStock.csv") -> pd.DataFrame:
    """Load MSFT data from local CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df[["open", "high", "low", "close", "volume"]]
    return df


def train_main(ticker: str = "MSFT", epochs: int = 5):
    print("Loading data...")
    df = load_msft()
    print("Adding features...")
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

    seq_len = 60
    X, y = create_sequences(data, seq_len)

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(X_train)}, test samples: {len(X_test)}")

    model = build_lstm(input_shape=X_train.shape[1:])
    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1)

    preds = model.predict(X_test).flatten()

    # Metrics (note predictions/targets are scaled to first feature range)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.6f}, RMSE: {rmse:.6f}, R2: {r2:.4f}")

    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    model.save("models/msft_lstm.keras")
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Model saved to models/msft_lstm")
    print("Scaler saved to models/scaler.pkl")


if __name__ == "__main__":
    train_main()
