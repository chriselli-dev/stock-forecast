import hashlib
import pandas as pd
import numpy as np
import talib
import yfinance as yf
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    FEATURE_LAGS, SMA_PERIODS, EMA_PERIODS, RSI_PERIOD,
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    BOLLINGER_PERIOD, BOLLINGER_STD, ATR_PERIOD,
    HISTORY_YEARS, AVAILABLE_STOCKS,
)
from database.db_manager import DBManager


class CacheManager:
    def __init__(self, db: DBManager):
        self.db = db

    @staticmethod
    def compute_hash(ticker: str, start: str, end: str) -> str:
        raw = f"{ticker}|{start}|{end}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def is_cached(self, ticker: str, start: str, end: str) -> bool:
        h = self.compute_hash(ticker, start, end)
        return self.db.check_cache(h)

    def mark_cached(self, ticker: str, start: str, end: str):
        h = self.compute_hash(ticker, start, end)
        self.db.save_cache_entry(ticker, start, end, h)


class DataLoader:
    def __init__(self, db: DBManager):
        self.db = db
        self.cache = CacheManager(db)

    def load(self, ticker: str, period_years: int = None) -> pd.DataFrame:
        if period_years is None:
            period_years = HISTORY_YEARS
        end_date = datetime.today()
        start_date = end_date - timedelta(days=period_years * 365)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        stock_name = AVAILABLE_STOCKS.get(ticker, ticker)
        self.db.upsert_stock(ticker, stock_name)

        if self.cache.is_cached(ticker, start_str, end_str):
            df = self.db.load_historical_data(ticker, start_str, end_str)
            if not df.empty and len(df) > 50:
                return df

        df = self._fetch_from_api(ticker, start_str, end_str)
        if df is not None and not df.empty:
            self.db.save_historical_data(ticker, df)
            self.cache.mark_cached(ticker, start_str, end_str)
        return df

    @staticmethod
    def _fetch_from_api(ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if data.empty:
                return pd.DataFrame()
            data = data.reset_index()
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
            cols_map = {}
            for c in data.columns:
                cl = str(c).lower().strip()
                if cl == "date":
                    cols_map[c] = "Date"
                elif cl == "open":
                    cols_map[c] = "Open"
                elif cl == "high":
                    cols_map[c] = "High"
                elif cl == "low":
                    cols_map[c] = "Low"
                elif cl == "close":
                    cols_map[c] = "Close"
                elif cl == "volume":
                    cols_map[c] = "Volume"
            data = data.rename(columns=cols_map)
            required = ["Date", "Open", "High", "Low", "Close", "Volume"]
            for col in required:
                if col not in data.columns:
                    return pd.DataFrame()
            data = data[required].copy()
            data["Date"] = pd.to_datetime(data["Date"])
            for col in ["Open", "High", "Low", "Close"]:
                data[col] = pd.to_numeric(data[col], errors="coerce")
            data["Volume"] = pd.to_numeric(data["Volume"], errors="coerce").fillna(0).astype(int)
            return data
        except Exception:
            return pd.DataFrame()


class DataPreprocessor:
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.copy()
        df = df.drop_duplicates(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df = df.dropna(subset=["Close"])
        df = df[df["Close"] > 0]
        return df.reset_index(drop=True)

    @staticmethod
    def validate(df: pd.DataFrame, min_rows: int = 100) -> bool:
        return df is not None and not df.empty and len(df) >= min_rows


class FeatureEngineer:
    @staticmethod
    def add_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        close = df["Close"].values.astype(float)
        high = df["High"].values.astype(float)
        low = df["Low"].values.astype(float)
        volume = df["Volume"].values.astype(float)

        for period in SMA_PERIODS:
            df[f"SMA_{period}"] = talib.SMA(close, timeperiod=period)
            sma = df[f"SMA_{period}"].values
            df[f"Close_to_SMA_{period}"] = np.where(sma != 0, close / sma, np.nan)

        for period in EMA_PERIODS:
            df[f"EMA_{period}"] = talib.EMA(close, timeperiod=period)

        macd, macd_signal, macd_hist = talib.MACD(
            close, fastperiod=MACD_FAST, slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL
        )
        df["MACD"] = macd
        df["MACD_Signal"] = macd_signal
        df["MACD_Hist"] = macd_hist

        df["RSI"] = talib.RSI(close, timeperiod=RSI_PERIOD)

        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close, timeperiod=BOLLINGER_PERIOD, nbdevup=BOLLINGER_STD, nbdevdn=BOLLINGER_STD
        )
        df["BB_Middle"] = bb_middle
        df["BB_Upper"] = bb_upper
        df["BB_Lower"] = bb_lower
        bb_width = bb_upper - bb_lower
        df["BB_Position"] = np.where(bb_width != 0, (close - bb_lower) / bb_width, np.nan)

        df["ATR"] = talib.ATR(high, low, close, timeperiod=ATR_PERIOD)
        df["ATR_Pct"] = np.where(close != 0, df["ATR"].values / close, np.nan)

        df["OBV"] = talib.OBV(close, volume)
        df["OBV_SMA_10"] = talib.SMA(df["OBV"].values.astype(float), timeperiod=10)

        df["MOM_5"] = talib.MOM(close, timeperiod=5)
        df["MOM_10"] = talib.MOM(close, timeperiod=10)
        df["ROC_5"] = talib.ROC(close, timeperiod=5)
        df["ROC_10"] = talib.ROC(close, timeperiod=10)

        df["WILLR"] = talib.WILLR(high, low, close, timeperiod=14)
        df["CCI"] = talib.CCI(high, low, close, timeperiod=20)
        df["ADX"] = talib.ADX(high, low, close, timeperiod=14)

        close_series = df["Close"]
        volume_series = df["Volume"]

        for lag in FEATURE_LAGS:
            df[f"Close_Lag_{lag}"] = close_series.shift(lag)
            df[f"Return_{lag}d"] = close_series.pct_change(periods=lag)

        df["Pct_Change"] = close_series.pct_change()
        df["Volatility_5"] = close_series.pct_change().rolling(5).std()
        df["Volatility_10"] = close_series.pct_change().rolling(10).std()
        df["Volatility_20"] = close_series.pct_change().rolling(20).std()
        df["Volume_SMA_10"] = talib.SMA(volume, timeperiod=10)
        df["Volume_SMA_20"] = talib.SMA(volume, timeperiod=20)
        vol_sma20 = df["Volume_SMA_20"].values
        df["Volume_Ratio"] = np.where(vol_sma20 != 0, volume / vol_sma20, np.nan)
        df["High_Low_Range"] = high - low
        df["High_Low_Pct"] = np.where(close != 0, (high - low) / close, np.nan)

        df["Day_of_Week"] = pd.to_datetime(df["Date"]).dt.dayofweek
        df["Month"] = pd.to_datetime(df["Date"]).dt.month

        df = df.dropna().reset_index(drop=True)
        return df
