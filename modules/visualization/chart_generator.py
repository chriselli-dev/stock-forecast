import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots


class ChartGenerator:
    COLORS = {
        "history": "#636EFA",
        "RandomForest": "#00CC96",
        "LSTM": "#EF553B",
        "volume": "rgba(99, 110, 250, 0.3)",
        "grid": "rgba(128, 128, 128, 0.2)",
    }

    DISPLAY_NAMES = {"RandomForest": "Random Forest", "LSTM": "LSTM"}

    @classmethod
    def _name(cls, key: str) -> str:
        return cls.DISPLAY_NAMES.get(key, key)

    @staticmethod
    def plot_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03, row_heights=[0.75, 0.25],
            subplot_titles=(f"{ticker} — историческая динамика", "Объём торгов"),
        )
        fig.add_trace(
            go.Candlestick(
                x=df["Date"], open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"], name="Цена",
                increasing_line_color="#26A69A", decreasing_line_color="#EF5350",
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Bar(
                x=df["Date"], y=df["Volume"], name="Объём",
                marker_color=ChartGenerator.COLORS["volume"],
            ),
            row=2, col=1,
        )
        fig.update_layout(
            height=600, xaxis_rangeslider_visible=False,
            template="plotly_white", showlegend=False,
            margin=dict(l=50, r=20, t=50, b=30),
        )
        fig.update_xaxes(gridcolor=ChartGenerator.COLORS["grid"])
        fig.update_yaxes(gridcolor=ChartGenerator.COLORS["grid"])
        return fig

    @classmethod
    def plot_history_and_forecast(
        cls, df: pd.DataFrame, forecast_dates, forecast_values,
        ticker: str, model_key: str,
    ) -> go.Figure:
        display = cls._name(model_key)
        fig = go.Figure()
        tail = df.tail(90)
        fig.add_trace(
            go.Scatter(
                x=tail["Date"], y=tail["Close"], mode="lines",
                name="Исторические данные",
                line=dict(color=cls.COLORS["history"], width=2),
            )
        )
        color = cls.COLORS.get(model_key, "#FF6692")
        connector_dates = [tail["Date"].iloc[-1]] + list(forecast_dates)
        connector_vals = [float(tail["Close"].iloc[-1])] + [float(v) for v in forecast_values]
        fig.add_trace(
            go.Scatter(
                x=connector_dates, y=connector_vals, mode="lines+markers",
                name=f"Прогноз ({display})",
                line=dict(color=color, width=2, dash="dash"),
                marker=dict(size=5),
            )
        )
        fig.update_layout(
            title=f"{ticker} — прогноз модели {display}",
            xaxis_title="Дата", yaxis_title="Цена закрытия, USD",
            template="plotly_white", height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=60, b=30),
        )
        fig.update_xaxes(gridcolor=cls.COLORS["grid"])
        fig.update_yaxes(gridcolor=cls.COLORS["grid"])
        return fig

    @classmethod
    def plot_model_comparison(
        cls, df: pd.DataFrame, forecast_dates,
        rf_values, lstm_values, ticker: str,
    ) -> go.Figure:
        fig = go.Figure()
        tail = df.tail(90)
        last_date = tail["Date"].iloc[-1]
        last_close = float(tail["Close"].iloc[-1])

        fig.add_trace(
            go.Scatter(
                x=tail["Date"], y=tail["Close"], mode="lines",
                name="Исторические данные",
                line=dict(color=cls.COLORS["history"], width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[last_date] + list(forecast_dates),
                y=[last_close] + [float(v) for v in rf_values],
                mode="lines+markers", name="Random Forest",
                line=dict(color=cls.COLORS["RandomForest"], width=2, dash="dash"),
                marker=dict(size=5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[last_date] + list(forecast_dates),
                y=[last_close] + [float(v) for v in lstm_values],
                mode="lines+markers", name="LSTM",
                line=dict(color=cls.COLORS["LSTM"], width=2, dash="dot"),
                marker=dict(size=5),
            )
        )
        fig.update_layout(
            title=f"{ticker} — сравнение моделей Random Forest и LSTM",
            xaxis_title="Дата", yaxis_title="Цена закрытия, USD",
            template="plotly_white", height=550,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=60, b=30),
        )
        fig.update_xaxes(gridcolor=cls.COLORS["grid"])
        fig.update_yaxes(gridcolor=cls.COLORS["grid"])
        return fig

    @classmethod
    def plot_test_predictions(
        cls, dates, y_actual, y_predicted, ticker: str, model_key: str,
    ) -> go.Figure:
        display = cls._name(model_key)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=dates, y=y_actual, mode="lines", name="Факт",
                line=dict(color=cls.COLORS["history"], width=2),
            )
        )
        color = cls.COLORS.get(model_key, "#FF6692")
        fig.add_trace(
            go.Scatter(
                x=dates, y=y_predicted, mode="lines",
                name=f"Прогноз ({display})",
                line=dict(color=color, width=2, dash="dash"),
            )
        )
        fig.update_layout(
            title=f"{ticker} — результаты на тестовой выборке ({display})",
            xaxis_title="Дата", yaxis_title="Цена закрытия, USD",
            template="plotly_white", height=450,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=50, r=20, t=60, b=30),
        )
        return fig


class TableFormatter:
    @staticmethod
    def forecast_table(dates, values, model_name: str) -> pd.DataFrame:
        return pd.DataFrame({
            "Дата": [d.strftime("%d.%m.%Y") if hasattr(d, "strftime") else str(d) for d in dates],
            f"Прогноз ({model_name}), USD": [round(float(v), 2) for v in values],
        })

    @staticmethod
    def comparison_table(dates, rf_values, lstm_values) -> pd.DataFrame:
        return pd.DataFrame({
            "Дата": [d.strftime("%d.%m.%Y") if hasattr(d, "strftime") else str(d) for d in dates],
            "Random Forest, USD": [round(float(v), 2) for v in rf_values],
            "LSTM, USD": [round(float(v), 2) for v in lstm_values],
            "Разница, USD": [round(abs(float(r) - float(l)), 2) for r, l in zip(rf_values, lstm_values)],
        })

    @staticmethod
    def metrics_table(metrics_dict: dict) -> pd.DataFrame:
        rows = []
        for model_name, metrics in metrics_dict.items():
            rows.append({
                "Модель": model_name,
                "RMSE": round(metrics["RMSE"], 4),
                "MAE": round(metrics["MAE"], 4),
                "MAPE, %": round(metrics["MAPE"], 4),
            })
        return pd.DataFrame(rows)
