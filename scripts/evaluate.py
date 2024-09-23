import numpy as np
from typing import Dict
from scripts.metrics import mae_score, rmse_score, mape_score, r2_score


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculates common evaluation metrics for a set of true and
    predicted values.

    Args:
        y_true (np.ndarray):
            The true target values.
        y_pred (np.ndarray):
            The predicted target values.

    Returns:
        Dict[str, float]:
            A dictionary containing the calculated metric values,
            with keys for MAE, RMSE, MAPE, and R2.
    """
    return {
        "MAE": mae_score(y_true, y_pred),
        "RMSE": rmse_score(y_true, y_pred),
        "MAPE": mape_score(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
