import numpy as np
from typing import Dict

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
) -> Dict[str, float]:
    """
    Calculates common evaluation metrics for a set of true and
    predicted values.

    Args:
        y_true (np.ndarray):
            The true target values.
        y_pred (np.ndarray):
            The predicted target values.
        prefix (str):
            A prefix to add to the metric keys.

    Returns:
        Dict[str, float]:
            A dictionary containing the calculated metric values,
            with keys for MAE, RMSE, MAPE, and R2.
    """
    return {
        f"{prefix}mae": mean_absolute_error(y_true, y_pred),
        f"{prefix}rmse": mean_squared_error(y_true, y_pred) ** (0.5),
        f"{prefix}mape": mean_absolute_percentage_error(y_true, y_pred),
        f"{prefix}r2": r2_score(y_true, y_pred),
    }
