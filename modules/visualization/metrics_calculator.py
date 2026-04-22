import numpy as np


class MetricsCalculator:
    @staticmethod
    def compute(y_actual, y_predicted) -> dict:
        y_actual = np.array(y_actual, dtype=float)
        y_predicted = np.array(y_predicted, dtype=float)
        min_len = min(len(y_actual), len(y_predicted))
        y_actual = y_actual[:min_len]
        y_predicted = y_predicted[:min_len]

        valid = np.isfinite(y_actual) & np.isfinite(y_predicted)
        y_actual = y_actual[valid]
        y_predicted = y_predicted[valid]

        if len(y_actual) == 0:
            return {"RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0}

        rmse = float(np.sqrt(np.mean((y_actual - y_predicted) ** 2)))
        mae = float(np.mean(np.abs(y_actual - y_predicted)))

        mask = y_actual != 0
        if mask.any():
            mape = float(np.mean(np.abs((y_actual[mask] - y_predicted[mask]) / y_actual[mask])) * 100)
        else:
            mape = 0.0

        return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "MAPE": round(mape, 4)}

    @staticmethod
    def describe_metric(name: str) -> str:
        descriptions = {
            "RMSE": "Среднеквадратичная ошибка — показывает среднее отклонение прогноза от факта в единицах цены. Чем меньше значение, тем точнее модель.",
            "MAE": "Средняя абсолютная ошибка — среднее абсолютное отклонение прогноза от факта в долларах. Легко интерпретируется.",
            "MAPE": "Средняя абсолютная процентная ошибка (%). Показывает относительную точность прогноза. Значение ниже 1% — отличный результат, 1–5% — хороший.",
        }
        return descriptions.get(name, "")
