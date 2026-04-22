import json
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.extras
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import POSTGRES_CONFIG


class DBManager:
    def __init__(self):
        self._init_tables()

    @contextmanager
    def _get_connection(self):
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_tables(self):
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stocks (
                    ticker VARCHAR(10) PRIMARY KEY,
                    name VARCHAR(200) NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS historical_data (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
                    date DATE NOT NULL,
                    open DOUBLE PRECISION,
                    high DOUBLE PRECISION,
                    low DOUBLE PRECISION,
                    close DOUBLE PRECISION,
                    volume BIGINT,
                    UNIQUE(ticker, date)
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
                    model_type VARCHAR(20) NOT NULL CHECK (model_type IN ('RandomForest', 'LSTM')),
                    file_path VARCHAR(500) NOT NULL,
                    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rmse NUMERIC(10,4),
                    mae NUMERIC(10,4),
                    mape NUMERIC(10,4),
                    parameters JSONB
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS request_cache (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10) NOT NULL REFERENCES stocks(ticker) ON DELETE CASCADE,
                    period_start DATE NOT NULL,
                    period_end DATE NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    request_hash VARCHAR(64) UNIQUE
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_historical_ticker_date
                ON historical_data (ticker, date)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_model_ticker_type
                ON model_metadata (ticker, model_type)
            """)

    def upsert_stock(self, ticker: str, name: str):
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO stocks (ticker, name) VALUES (%s, %s) "
                "ON CONFLICT (ticker) DO UPDATE SET name = EXCLUDED.name",
                (ticker, name),
            )

    def save_historical_data(self, ticker: str, df: pd.DataFrame):
        with self._get_connection() as conn:
            cur = conn.cursor()
            rows = []
            for _, row in df.iterrows():
                date_str = row["Date"].strftime("%Y-%m-%d") if hasattr(row["Date"], "strftime") else str(row["Date"])
                rows.append((
                    ticker, date_str,
                    float(row["Open"]), float(row["High"]),
                    float(row["Low"]), float(row["Close"]),
                    int(row["Volume"]),
                ))
            psycopg2.extras.execute_batch(
                cur,
                """INSERT INTO historical_data (ticker, date, open, high, low, close, volume)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (ticker, date) DO UPDATE SET
                   open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                   close=EXCLUDED.close, volume=EXCLUDED.volume""",
                rows,
                page_size=500,
            )

    def save_model_metadata(self, ticker, model_type, file_path, rmse, mae, mape, parameters):
        with self._get_connection() as conn:
            cur = conn.cursor()
            params_json = json.dumps(parameters) if isinstance(parameters, dict) else parameters
            cur.execute(
                """INSERT INTO model_metadata (ticker, model_type, file_path, rmse, mae, mape, parameters)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (ticker, model_type, file_path, rmse, mae, mape, params_json),
            )

    def get_latest_model(self, ticker, model_type):
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT id, ticker, model_type, file_path, trained_at, rmse, mae, mape, parameters
                   FROM model_metadata WHERE ticker=%s AND model_type=%s
                   ORDER BY trained_at DESC LIMIT 1""",
                (ticker, model_type),
            )
            row = cur.fetchone()
            if row:
                return {
                    "id": row[0], "ticker": row[1], "model_type": row[2],
                    "file_path": row[3], "trained_at": row[4],
                    "rmse": float(row[5]) if row[5] else None,
                    "mae": float(row[6]) if row[6] else None,
                    "mape": float(row[7]) if row[7] else None,
                    "parameters": row[8],
                }
        return None

    def save_cache_entry(self, ticker, period_start, period_end, request_hash):
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO request_cache (ticker, period_start, period_end, request_hash)
                   VALUES (%s, %s, %s, %s) ON CONFLICT (request_hash) DO NOTHING""",
                (ticker, str(period_start), str(period_end), request_hash),
            )

    def check_cache(self, request_hash):
        with self._get_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id FROM request_cache WHERE request_hash=%s", (request_hash,))
            return cur.fetchone() is not None

    def load_historical_data(self, ticker, start_date, end_date):
        with self._get_connection() as conn:
            query = """SELECT date, open, high, low, close, volume
                       FROM historical_data
                       WHERE ticker=%s AND date BETWEEN %s AND %s
                       ORDER BY date"""
            df = pd.read_sql_query(query, conn, params=(ticker, str(start_date), str(end_date)))
            if not df.empty:
                df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
                df["Date"] = pd.to_datetime(df["Date"])
            return df
