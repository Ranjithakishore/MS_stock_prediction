"""Build and return a simple LSTM model using TensorFlow Keras."""
from typing import Any

import tensorflow as tf


def build_lstm(input_shape: Any) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model
