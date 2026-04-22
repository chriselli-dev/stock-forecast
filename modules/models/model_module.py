import os
import json
import pickle
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import RF_PARAMS, LSTM_PARAMS, SAVED_MODELS_DIR, TEST_SIZE, FORECAST_HORIZON_MAX


class BaseModel(ABC):
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def get_params(self) -> dict:
        pass


class RandomForestModel(BaseModel):
    """Модель Random Forest для прогнозирования фондового рынка.

    Вместо предсказания абсолютных цен модель обучается на дневных
    доходностях (относительных изменениях цены), что обеспечивает
    масштабную инвариантность и делает модель направленно-ориентированной:
    знак предсказанной доходности определяет прогнозируемое направление
    движения цены, а величина — силу движения.
    """

    def __init__(self, ticker: str, params: dict = None):
        super().__init__(ticker)
        self.params = params or RF_PARAMS.copy()
        self.eval_model = None
        self.forecast_models = {}
        self.feature_names = None
        self.max_horizon = FORECAST_HORIZON_MAX

    def train(self, X_train, y_train):
        """Обучение eval-модели на дневных доходностях.

        Целевая переменная — дневная доходность: return[t] = (Close[t+1] -
        Close[t]) / Close[t]. Модель обучается на признаках features[t] и
        предсказывает доходность, что позволяет определить как направление
        движения цены (по знаку), так и его ожидаемую величину.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_np = X_train.values
        else:
            X_np = np.array(X_train)

        closes = np.array(y_train, dtype=float)

        # Вычисление дневных доходностей: return[t] = (Close[t+1] - Close[t]) / Close[t]
        returns = np.diff(closes) / np.where(closes[:-1] == 0, 1.0, closes[:-1])

        # Выравнивание: features[0..N-2] → returns[0..N-2]
        X_aligned = X_np[:-1]

        self.eval_model = RandomForestRegressor(**self.params)
        self.eval_model.fit(X_aligned, returns)
        self.model = self.eval_model
        self.is_trained = True

    def train_multistep(self, df: pd.DataFrame, feature_cols: list, target_col: str = "Close"):
        """Прямое многошаговое прогнозирование на основе кумулятивных доходностей.

        Для каждого горизонта h обучается отдельная модель, предсказывающая
        кумулятивную доходность за h дней: (Close[t+h] - Close[t]) / Close[t].
        При прогнозировании предсказанная доходность конвертируется в цену
        через умножение на базовую цену.
        """
        self.feature_names = feature_cols
        X = df[feature_cols].values
        close = df[target_col].values.astype(float)

        for h in range(1, self.max_horizon + 1):
            valid = len(X) - h
            if valid < 50:
                break

            # Кумулятивная доходность за h дней
            base_prices = close[:valid]
            future_prices = close[h : h + valid]
            cum_returns = (future_prices - base_prices) / np.where(
                base_prices == 0, 1.0, base_prices
            )

            model = RandomForestRegressor(**self.params)
            model.fit(X[:valid], cum_returns)
            self.forecast_models[h] = model

    def predict(self, X, close_prices=None):
        """Предсказание доходностей с конвертацией в цены.

        Модель предсказывает дневную доходность для каждой строки X.
        Если передан массив close_prices (базовые цены), результат
        конвертируется в абсолютные цены: price = close * (1 + return).
        """
        if isinstance(X, pd.DataFrame):
            X_np = X.values
        else:
            X_np = np.array(X)

        predicted_returns = self.eval_model.predict(X_np)

        if close_prices is not None:
            bases = np.array(close_prices, dtype=float)
            return bases * (1.0 + predicted_returns)

        return predicted_returns

    def predict_horizon(self, X_last_row, horizon: int, last_close: float = None) -> np.ndarray:
        """Многошаговый прогноз на основе кумулятивных доходностей.

        Каждая модель из forecast_models предсказывает кумулятивную
        доходность за соответствующий горизонт h. Прогнозная цена
        вычисляется как: price = last_close * (1 + cumulative_return).
        """
        if isinstance(X_last_row, pd.DataFrame):
            X_last_row = X_last_row.values
        if X_last_row.ndim == 1:
            X_last_row = X_last_row.reshape(1, -1)

        if last_close is None or last_close == 0:
            last_close = 100.0  # fallback

        if not self.forecast_models:
            pred_return = float(self.eval_model.predict(X_last_row)[0])
            return np.full(horizon, last_close * (1.0 + pred_return))

        predictions = []
        for h in range(1, horizon + 1):
            if h in self.forecast_models:
                cum_return = float(self.forecast_models[h].predict(X_last_row)[0])
            else:
                available = [k for k in self.forecast_models if k <= h]
                max_h = max(available) if available else max(self.forecast_models)
                cum_return = float(self.forecast_models[max_h].predict(X_last_row)[0])

            pred_price = last_close * (1.0 + cum_return)
            predictions.append(pred_price)

        return np.array(predictions)

    def save(self, path: str):
        data = {
            "eval_model": self.eval_model,
            "forecast_models": self.forecast_models,
            "params": self.params,
            "feature_names": self.feature_names,
            "max_horizon": self.max_horizon,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.eval_model = data["eval_model"]
        self.forecast_models = data["forecast_models"]
        self.params = data["params"]
        self.feature_names = data.get("feature_names")
        self.max_horizon = data.get("max_horizon", FORECAST_HORIZON_MAX)
        self.model = self.eval_model
        self.is_trained = True

    def get_params(self) -> dict:
        return self.params.copy()

    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_trained or self.feature_names is None or self.eval_model is None:
            return pd.DataFrame()
        imp = self.eval_model.feature_importances_
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": imp}
        ).sort_values("importance", ascending=False).reset_index(drop=True)


class LSTMModel(BaseModel):
    def __init__(self, ticker: str, params: dict = None):
        super().__init__(ticker)
        self.params = params or LSTM_PARAMS.copy()
        self.scaler_X = StandardScaler()
        self.window_size = self.params["window_size"]
        self.feature_names = None

    def _build_model(self, input_shape):
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM as KerasLSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        model = Sequential([
            KerasLSTM(self.params["units_1"], return_sequences=True, input_shape=input_shape),
            Dropout(self.params["dropout"]),
            KerasLSTM(self.params["units_2"], return_sequences=True),
            Dropout(self.params["dropout"]),
            KerasLSTM(self.params["units_3"], return_sequences=False),
            Dropout(self.params["dropout"]),
            Dense(32, activation="relu"),
            Dense(1),
        ])
        model.compile(
            optimizer=Adam(learning_rate=self.params["learning_rate"]),
            loss="huber",
        )
        return model

    def train(self, X_train, y_train):
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            X_train = X_train.values
        X_train = np.array(X_train, dtype=float)
        closes = np.array(y_train, dtype=float)
        N = len(X_train)
        W = self.window_size

        returns = np.diff(closes) / np.where(closes[:-1] == 0, 1.0, closes[:-1])

        self.scaler_X.fit(X_train)
        X_scaled = self.scaler_X.transform(X_train)

        Xs, ys = [], []
        for i in range(N - W):
            t = W - 1 + i
            if t >= len(returns):
                break
            Xs.append(X_scaled[i : i + W])
            ys.append(returns[t])

        if len(Xs) < 20:
            raise ValueError("Недостаточно данных для обучения LSTM")

        X_seq = np.array(Xs)
        y_seq = np.array(ys)

        self.model = self._build_model((W, X_seq.shape[2]))
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]
        self.model.fit(
            X_seq, y_seq,
            epochs=self.params["epochs"],
            batch_size=self.params["batch_size"],
            validation_split=0.15,
            callbacks=callbacks,
            verbose=0,
        )
        self.is_trained = True

    def train_multistep(self, df, feature_cols, target_col="Close"):
        """Заглушка для совместимости с интерфейсом RandomForestModel.

        LSTM не использует стратегию Direct Multi-Step Forecasting:
        многошаговый прогноз строится рекурсивно в predict_horizon на
        основе единственной обученной сети, поэтому отдельные модели
        для каждого горизонта не нужны. Метод оставлен пустым, чтобы
        ModelTrainer мог единообразно вызывать его для любой модели,
        наследующей BaseModel.
        """
        pass

    def predict(self, X, close_prices=None):
        """Прогноз цен на тестовой выборке (однодневный прогноз).

        Для каждой позиции t скользящее окно из W предшествующих строк
        признаков подаётся в обученную нейросеть, которая возвращает
        предсказанную доходность на следующий день. Прогнозная цена
        вычисляется как closes[t] * (1 + pred_return) — то есть базой
        служит фактическая цена в момент t, а не ранее предсказанная.

        Такой режим используется для оценки качества модели на тестовой
        выборке: метрики RMSE/MAE/MAPE измеряют точность однодневного
        прогноза. Это согласуется с eval-моделью Random Forest, которая
        также предсказывает однодневную доходность, и позволяет корректно
        сравнивать обе модели в одинаковых условиях. Многошаговый прогноз
        на произвольный горизонт реализован отдельно в predict_horizon.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = np.array(X, dtype=float)

        if close_prices is None:
            return np.array([])

        closes = np.array(close_prices, dtype=float)
        M = len(X)
        W = self.window_size

        if M <= W:
            return np.array([])

        X_scaled = self.scaler_X.transform(X)

        predictions = []
        for i in range(M - W):
            t = W - 1 + i
            if t >= len(closes) - 1:
                break
            seq = X_scaled[i : i + W].reshape(1, W, -1)
            pred_return = float(self.model.predict(seq, verbose=0)[0, 0])
            # Ограничение доходности диапазоном [-5%, +5%] отсекает
            # нереалистичные выбросы и согласовано с predict_horizon.
            pred_return = np.clip(pred_return, -0.05, 0.05)
            base_price = closes[t]
            pred_price = base_price * (1.0 + pred_return)
            predictions.append(pred_price)

        return np.array(predictions) if predictions else np.array([])

    def predict_horizon(self, X_full, horizon: int, last_close: float) -> np.ndarray:
        """Рекурсивное многошаговое прогнозирование с использованием LSTM."""
        if isinstance(X_full, pd.DataFrame):
            X_full = X_full.values
        X_full = np.array(X_full, dtype=float)
        W = self.window_size

        if len(X_full) < W:
            return np.full(horizon, last_close)

        X_scaled = self.scaler_X.transform(X_full)

        current_window = X_scaled[-W:].copy()
        current_price = float(last_close)
        predictions = []

        for step in range(horizon):
            seq = current_window.reshape(1, W, -1)
            pred_return = float(self.model.predict(seq, verbose=0)[0, 0])

            if not np.isfinite(pred_return):
                pred_return = 0.0
            pred_return = np.clip(pred_return, -0.05, 0.05)

            new_price = current_price * (1.0 + pred_return)
            predictions.append(new_price)

            new_row = current_window[-1:].copy()
            current_window = np.vstack([current_window[1:], new_row])

            current_price = new_price

        return np.array(predictions)

    def save(self, path: str):
        """Сохраняет модель LSTM в единый файл формата HDF5."""
        import h5py

        h5_path = path if path.endswith(".h5") else path + ".h5"

        self.model.save(h5_path)

        with h5py.File(h5_path, "a") as f:
            if "app_metadata" in f:
                del f["app_metadata"]
            grp = f.create_group("app_metadata")
            grp.create_dataset("scaler_mean", data=self.scaler_X.mean_)
            grp.create_dataset("scaler_scale", data=self.scaler_X.scale_)
            grp.attrs["window_size"] = self.window_size
            grp.attrs["params"] = json.dumps(self.params)
            if self.feature_names:
                dt = h5py.string_dtype()
                grp.create_dataset("feature_names", data=self.feature_names, dtype=dt)

    def load(self, path: str):
        """Загружает модель LSTM из единого HDF5-файла."""
        import h5py
        from tensorflow.keras.models import load_model as keras_load

        h5_path = path if path.endswith(".h5") else path + ".h5"

        self.model = keras_load(h5_path, compile=False)

        with h5py.File(h5_path, "r") as f:
            grp = f["app_metadata"]
            scaler_mean = grp["scaler_mean"][:]
            scaler_scale = grp["scaler_scale"][:]
            self.window_size = int(grp.attrs["window_size"])
            self.params = json.loads(grp.attrs["params"])
            if "feature_names" in grp:
                raw = grp["feature_names"][:]
                self.feature_names = [
                    s.decode("utf-8") if isinstance(s, bytes) else str(s)
                    for s in raw
                ]
            else:
                self.feature_names = None

        self.scaler_X = StandardScaler()
        self.scaler_X.mean_ = scaler_mean
        self.scaler_X.scale_ = scaler_scale
        self.scaler_X.var_ = scaler_scale ** 2
        self.scaler_X.n_features_in_ = len(scaler_mean)

        self.is_trained = True

    def get_params(self) -> dict:
        return self.params.copy()


class ModelTrainer:
    def __init__(self, db_manager):
        self.db = db_manager

    def train_model(self, model: BaseModel, df: pd.DataFrame, feature_cols: list, target_col: str = "Close"):
        from modules.visualization.metrics_calculator import MetricsCalculator
        from config import AVAILABLE_STOCKS

        split_idx = int(len(df) * (1 - TEST_SIZE))
        X = df[feature_cols]
        y = df[target_col]

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        model.train(X_train, y_train)

        if isinstance(model, RandomForestModel):
            model.train_multistep(df.iloc[:split_idx], feature_cols, target_col)
            base_prices = y_test.values[:-1]
            preds = model.predict(X_test.iloc[:-1], close_prices=base_prices)
            y_actual = y_test.values[1:]
        else:
            preds = model.predict(X_test, close_prices=y_test.values)
            y_actual = y_test.values[model.window_size:]

        min_len = min(len(y_actual), len(preds))
        if min_len == 0:
            metrics = {"RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0}
            return model, metrics, np.array([]), np.array([])

        y_actual = y_actual[:min_len]
        preds = preds[:min_len]

        metrics = MetricsCalculator.compute(y_actual, preds)

        model_type = "RandomForest" if isinstance(model, RandomForestModel) else "LSTM"
        filename = f"{model.ticker}_{model_type}"
        filepath = os.path.join(SAVED_MODELS_DIR, filename)
        model.save(filepath)

        stock_name = AVAILABLE_STOCKS.get(model.ticker, model.ticker)
        self.db.upsert_stock(model.ticker, stock_name)
        self.db.save_model_metadata(
            ticker=model.ticker,
            model_type=model_type,
            file_path=filepath,
            rmse=metrics["RMSE"],
            mae=metrics["MAE"],
            mape=metrics["MAPE"],
            parameters=model.get_params(),
        )
        return model, metrics, y_actual, preds


class Predictor:
    def __init__(self, db_manager):
        self.db = db_manager

    def load_model(self, ticker: str, model_type: str) -> BaseModel:
        meta = self.db.get_latest_model(ticker, model_type)
        if meta is None:
            return None
        if model_type == "RandomForest":
            model = RandomForestModel(ticker)
        else:
            model = LSTMModel(ticker)
        model.load(meta["file_path"])
        return model

    @staticmethod
    def forecast_rf(model: RandomForestModel, df: pd.DataFrame, feature_cols: list, horizon: int):
        last_row = df[feature_cols].iloc[-1:]
        last_close = float(df["Close"].iloc[-1])
        return model.predict_horizon(last_row, horizon, last_close)

    @staticmethod
    def forecast_lstm(model: LSTMModel, df: pd.DataFrame, feature_cols: list, horizon: int):
        last_close = float(df["Close"].iloc[-1])
        return model.predict_horizon(df[feature_cols], horizon, last_close)