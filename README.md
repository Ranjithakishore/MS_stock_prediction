# Microsoft Stock Price Prediction (Beginner)

This beginner-friendly project shows how to build a simple time-series forecasting pipeline to predict Microsoft (`MSFT`) stock prices using Python and TensorFlow (LSTM).

Quick start

1. Create and activate a Python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run the training script (downloads data automatically):

```powershell
python -m src.train
```

Project layout

- `src/` — core modules and scripts
- `requirements.txt` — Python dependencies
- `README.md` — this file

What the code does

- Downloads `MSFT` historical data using `yfinance`.
- Adds simple technical features (SMA, EMA, Bollinger Bands, RSI).
- Preprocesses and scales data, builds sequences for LSTM.
- Trains a compact LSTM model and prints MAE, RMSE, R2.
- Provides a small `predict.py` helper to forecast the next N days.

If you want, I can run the training locally in this workspace next. Would you like that?
