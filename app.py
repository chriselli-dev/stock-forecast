import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
from datetime import timedelta

from config import (
    AVAILABLE_STOCKS, FORECAST_HORIZON_MIN, FORECAST_HORIZON_MAX,
    FORECAST_HORIZON_DEFAULT, HISTORY_YEARS,
)
from modules.data_module import DataLoader, DataPreprocessor, FeatureEngineer
from modules.models.model_module import (
    RandomForestModel, LSTMModel, ModelTrainer, Predictor,
)
from modules.visualization.chart_generator import ChartGenerator, TableFormatter
from modules.visualization.metrics_calculator import MetricsCalculator

st.set_page_config(
    page_title="Прогнозирование фондового рынка",
    page_icon="📈",
    layout="wide",
)

FEATURE_COLS_EXCLUDE = ["Date", "Open", "High", "Low", "Volume"]

MODEL_DISPLAY = {"RandomForest": "Random Forest", "LSTM": "LSTM"}


@st.cache_resource
def get_db():
    try:
        from database.db_manager import DBManager
        return DBManager()
    except Exception as e:
        return str(e)


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE and c != "Close"]


def generate_forecast_dates(last_date: pd.Timestamp, horizon: int) -> list:
    dates = []
    current = last_date
    count = 0
    while count < horizon:
        current = current + timedelta(days=1)
        if current.weekday() < 5:
            dates.append(current)
            count += 1
    return dates


def train_and_predict(db, ticker, model_type, df_featured, feature_cols, horizon):
    trainer = ModelTrainer(db)
    if model_type == "RandomForest":
        model = RandomForestModel(ticker)
    else:
        model = LSTMModel(ticker)

    model, metrics, y_actual, y_predicted = trainer.train_model(
        model, df_featured, feature_cols, "Close"
    )

    if model_type == "RandomForest":
        forecast = Predictor.forecast_rf(model, df_featured, feature_cols, horizon)
    else:
        last_close = float(df_featured["Close"].iloc[-1])
        forecast = model.predict_horizon(df_featured[feature_cols], horizon, last_close)

    return model, metrics, y_actual, y_predicted, forecast


def _render_test_chart(df_featured, y_actual, y_predicted, ticker, model_type):
    if len(y_actual) == 0 or len(y_predicted) == 0:
        st.warning(
            "Недостаточно данных для оценки на тестовой выборке. "
            "Увеличьте период истории до 3–5 лет для получения метрик."
        )
        return

    split_idx = int(len(df_featured) * 0.8)
    test_dates = df_featured["Date"].iloc[split_idx:].values
    offset = len(test_dates) - len(y_actual)
    td = test_dates[offset:] if offset > 0 else test_dates
    ml = min(len(td), len(y_actual), len(y_predicted))

    if ml == 0:
        st.warning("Недостаточно данных для построения графика тестовой выборки.")
        return

    fig = ChartGenerator.plot_test_predictions(
        td[:ml], y_actual[:ml], y_predicted[:ml], ticker, model_type,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_metrics(metrics):
    c1, c2, c3 = st.columns(3)
    rmse_val = metrics["RMSE"]
    mae_val = metrics["MAE"]
    mape_val = metrics["MAPE"]

    if rmse_val == 0 and mae_val == 0 and mape_val == 0:
        c1.metric("RMSE", "—", help=MetricsCalculator.describe_metric("RMSE"))
        c2.metric("MAE", "—", help=MetricsCalculator.describe_metric("MAE"))
        c3.metric("MAPE", "—", help=MetricsCalculator.describe_metric("MAPE"))
        st.caption("Метрики недоступны: недостаточно тестовых данных. Увеличьте период истории.")
    else:
        c1.metric("RMSE", f"{rmse_val:.4f}", help=MetricsCalculator.describe_metric("RMSE"))
        c2.metric("MAE", f"{mae_val:.4f}", help=MetricsCalculator.describe_metric("MAE"))
        c3.metric("MAPE", f"{mape_val:.4f}%", help=MetricsCalculator.describe_metric("MAPE"))


def main():
    db = get_db()
    if isinstance(db, str):
        st.error(
            "Не удалось подключиться к PostgreSQL. Убедитесь, что сервер запущен "
            "и параметры подключения заданы верно (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD).\n\n"
            f"Ошибка: `{db}`"
        )
        return

    st.sidebar.title("📊 Панель управления")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Режим работы",
        ["📈 Историческая динамика", "🔮 Прогноз", "⚖️ Сравнение моделей"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Выбор инструмента")

    stock_options = [f"{t} — {n}" for t, n in AVAILABLE_STOCKS.items()]
    selected = st.sidebar.selectbox(
        "Акция",
        stock_options,
        index=0,
        help="Выберите финансовый инструмент из списка",
    )
    ticker = selected.split(" — ")[0]
    stock_name = AVAILABLE_STOCKS[ticker]

    history_years = st.sidebar.slider(
        "Период истории (лет)",
        min_value=1, max_value=5, value=HISTORY_YEARS,
        help="Количество лет исторических данных для загрузки и анализа. "
             "Для прогнозирования рекомендуется не менее 3 лет.",
    )

    horizon = FORECAST_HORIZON_DEFAULT
    model_choice = None

    if mode != "📈 Историческая динамика":
        st.sidebar.markdown("---")
        st.sidebar.subheader("Параметры прогноза")

        if mode == "🔮 Прогноз":
            model_choice = st.sidebar.selectbox(
                "Модель прогнозирования",
                ["Random Forest", "LSTM"],
                help="Random Forest — быстрая и устойчивая модель на основе ансамбля деревьев решений.\n\n"
                     "LSTM — рекуррентная нейросеть, учитывающая долгосрочные зависимости во временных рядах.",
            )

        horizon = st.sidebar.slider(
            "Горизонт прогноза (торговых дней)",
            min_value=FORECAST_HORIZON_MIN,
            max_value=FORECAST_HORIZON_MAX,
            value=FORECAST_HORIZON_DEFAULT,
            help="Количество рабочих дней, на которые строится прогноз (от 1 до 30)",
        )

    run_button = st.sidebar.button(
        "🚀 Запустить",
        use_container_width=True,
        type="primary",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Система прогнозирования фондового рынка с использованием "
        "методов машинного обучения. Разработана в рамках ВКР."
    )

    st.title("Система прогнозирования фондового рынка")
    st.markdown(f"**Инструмент:** {ticker} — {stock_name}")

    if not run_button:
        st.info("Выберите параметры на боковой панели и нажмите «Запустить» для начала работы.")
        return

    with st.spinner("Загрузка исторических данных..."):
        try:
            loader = DataLoader(db)
            df_raw = loader.load(ticker, period_years=history_years)
        except Exception as e:
            st.error(f"Ошибка при загрузке данных: {e}")
            return

    if df_raw is None or df_raw.empty:
        st.error(
            f"Не удалось загрузить данные для {ticker}. "
            "Проверьте подключение к интернету и попробуйте снова."
        )
        return

    df_clean = DataPreprocessor.clean(df_raw)

    if not DataPreprocessor.validate(df_clean):
        st.error(
            f"Недостаточно данных для анализа {ticker}. "
            "Необходимо минимум 100 торговых дней. Попробуйте увеличить период."
        )
        return

    if mode == "📈 Историческая динамика":
        _show_history(df_clean, ticker)
    elif mode == "🔮 Прогноз":
        _show_forecast(db, df_clean, ticker, model_choice, horizon)
    elif mode == "⚖️ Сравнение моделей":
        _show_comparison(db, df_clean, ticker, horizon)


def _show_history(df: pd.DataFrame, ticker: str):
    st.subheader("Историческая динамика")

    fig = ChartGenerator.plot_candlestick(df, ticker)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    change = float(last["Close"] - prev["Close"])
    change_pct = (change / float(prev["Close"])) * 100 if float(prev["Close"]) != 0 else 0

    col1.metric("Последняя цена", f"${last['Close']:.2f}", f"{change:+.2f} ({change_pct:+.1f}%)")
    col2.metric("Максимум за период", f"${df['High'].max():.2f}")
    col3.metric("Минимум за период", f"${df['Low'].min():.2f}")
    col4.metric("Средний объём", f"{df['Volume'].mean():,.0f}")

    with st.expander("Данные за последние 20 торговых дней"):
        display_df = df.tail(20)[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%d.%m.%Y")
        for col in ["Open", "High", "Low", "Close"]:
            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}")
        display_df["Volume"] = display_df["Volume"].apply(lambda x: f"{int(x):,}")
        display_df.columns = ["Дата", "Открытие", "Максимум", "Минимум", "Закрытие", "Объём"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


def _show_forecast(db, df_clean, ticker, model_choice, horizon):
    model_type = "RandomForest" if model_choice == "Random Forest" else "LSTM"
    display_name = MODEL_DISPLAY[model_type]

    st.subheader(f"Прогноз — {display_name}")

    df_featured = FeatureEngineer.add_features(df_clean)
    feature_cols = get_feature_cols(df_featured)

    if not DataPreprocessor.validate(df_featured, min_rows=100):
        st.error("После расчёта индикаторов осталось недостаточно данных. Увеличьте период истории.")
        return

    with st.spinner(f"Обучение модели {display_name}... Это может занять до минуты."):
        try:
            model, metrics, y_actual, y_predicted, forecast = train_and_predict(
                db, ticker, model_type, df_featured, feature_cols, horizon
            )
        except Exception as e:
            st.error(f"Ошибка при обучении модели: {e}")
            return

    forecast_dates = generate_forecast_dates(df_clean["Date"].iloc[-1], horizon)

    fig = ChartGenerator.plot_history_and_forecast(
        df_clean, forecast_dates, forecast, ticker, model_type
    )
    st.plotly_chart(fig, use_container_width=True)

    _render_metrics(metrics)

    st.markdown("#### Прогнозные значения")
    forecast_df = TableFormatter.forecast_table(forecast_dates, forecast, display_name)
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

    with st.expander("Результаты на тестовой выборке"):
        _render_test_chart(df_featured, y_actual, y_predicted, ticker, model_type)


def _show_comparison(db, df_clean, ticker, horizon):
    st.subheader("Сравнение моделей")

    df_featured = FeatureEngineer.add_features(df_clean)
    feature_cols = get_feature_cols(df_featured)

    if not DataPreprocessor.validate(df_featured, min_rows=100):
        st.error("После расчёта индикаторов осталось недостаточно данных. Увеличьте период истории.")
        return

    progress = st.progress(0, text="Обучение модели Random Forest...")
    try:
        rf_model, rf_metrics, rf_actual, rf_pred, rf_forecast = train_and_predict(
            db, ticker, "RandomForest", df_featured, feature_cols, horizon
        )
    except Exception as e:
        st.error(f"Ошибка при обучении Random Forest: {e}")
        return

    progress.progress(50, text="Обучение модели LSTM...")
    try:
        lstm_model, lstm_metrics, lstm_actual, lstm_pred, lstm_forecast = train_and_predict(
            db, ticker, "LSTM", df_featured, feature_cols, horizon
        )
    except Exception as e:
        st.error(f"Ошибка при обучении LSTM: {e}")
        return

    progress.progress(100, text="Готово!")
    progress.empty()

    forecast_dates = generate_forecast_dates(df_clean["Date"].iloc[-1], horizon)

    fig = ChartGenerator.plot_model_comparison(
        df_clean, forecast_dates, rf_forecast, lstm_forecast, ticker
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Метрики качества моделей")

    has_rf_metrics = not (rf_metrics["RMSE"] == 0 and rf_metrics["MAE"] == 0)
    has_lstm_metrics = not (lstm_metrics["RMSE"] == 0 and lstm_metrics["MAE"] == 0)

    col_rf, col_lstm = st.columns(2)

    with col_rf:
        st.markdown("**Random Forest**")
        _render_metrics(rf_metrics)

    with col_lstm:
        st.markdown("**LSTM**")
        _render_metrics(lstm_metrics)

    if has_rf_metrics and has_lstm_metrics:
        best_rmse = "Random Forest" if rf_metrics["RMSE"] <= lstm_metrics["RMSE"] else "LSTM"
        best_mae = "Random Forest" if rf_metrics["MAE"] <= lstm_metrics["MAE"] else "LSTM"
        best_mape = "Random Forest" if rf_metrics["MAPE"] <= lstm_metrics["MAPE"] else "LSTM"
        st.info(
            f"По метрике RMSE лучше себя показала модель **{best_rmse}**, "
            f"по MAE — **{best_mae}**, по MAPE — **{best_mape}**."
        )

    st.markdown("#### Сравнительная таблица прогнозов")
    comp_df = TableFormatter.comparison_table(forecast_dates, rf_forecast, lstm_forecast)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    with st.expander("Сводная таблица метрик"):
        metrics_df = TableFormatter.metrics_table({
            "Random Forest": rf_metrics,
            "LSTM": lstm_metrics,
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with st.expander("Результаты на тестовой выборке — Random Forest"):
        _render_test_chart(df_featured, rf_actual, rf_pred, ticker, "RandomForest")

    with st.expander("Результаты на тестовой выборке — LSTM"):
        _render_test_chart(df_featured, lstm_actual, lstm_pred, ticker, "LSTM")


if __name__ == "__main__":
    main()
