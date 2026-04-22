import os

POSTGRES_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "stock_forecast"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

AVAILABLE_STOCKS = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp.",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corp.",
    "JPM": "JPMorgan Chase & Co.",
    "V": "Visa Inc.",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart Inc.",
    "PG": "Procter & Gamble Co.",
    "DIS": "Walt Disney Co.",
    "NFLX": "Netflix Inc.",
    "INTC": "Intel Corp.",
}

FORECAST_HORIZON_MIN = 1
FORECAST_HORIZON_MAX = 30
FORECAST_HORIZON_DEFAULT = 14

HISTORY_YEARS = 5

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1,
}

LSTM_PARAMS = {
    "units_1": 64,
    "units_2": 32,
    "units_3": 16,
    "dropout": 0.15,
    "epochs": 50,
    "batch_size": 64,
    "window_size": 40,
    "learning_rate": 0.0005,
}

FEATURE_LAGS = [1, 2, 3, 5, 10, 20]
SMA_PERIODS = [5, 10, 20, 50]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ATR_PERIOD = 14

TEST_SIZE = 0.2

SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
