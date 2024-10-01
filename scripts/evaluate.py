import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    root_mean_squared_error,
    make_scorer,
)

from scripts import logger
from scripts.metrics import mean_cv_scores


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


def eval_model(
    model: object,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    cv: int | None = 5,
) -> Dict[str, Any]:
    """
    Evaluates a machine learning model by calculating various
    performance metrics on the test set and optionally performing
    cross-validation.

    Args:
        model (object):
            The trained machine learning model to evaluate.
        X_train (pd.DataFrame | np.ndarray):
            The training data features.
        y_train (pd.Series | np.ndarray):
            The training data target values.
        X_test (pd.DataFrame | np.ndarray):
            The test data features.
        y_test (pd.Series | np.ndarray):
            The test data target values.
        cv (int | None, optional):
            The number of cross-validation folds to perform.
            Defaults to 5.

    Returns:
        dict:
            A dictionary containing the evaluated model,
            an example of the input data,
            the model's predictions on the test set,
            and the calculated performance metrics
            (including test set and cross-validation metrics).
    """

    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, prediction, prefix="test_")
    logger.info(f"Test metrics: {test_metrics}")

    if cv:
        kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_metrics = cross_validate(
            model,
            X_train,
            y_train,
            cv=kfold.split(X_train, pd.qcut(y_train, 10, labels=False)),
            scoring={
                "neg_mae": make_scorer(
                    mean_absolute_error, greater_is_better=False
                ),
                "neg_rmse": make_scorer(
                    root_mean_squared_error, greater_is_better=False
                ),
                "neg_mape": make_scorer(
                    mean_absolute_percentage_error, greater_is_better=False
                ),
                "r2": make_scorer(r2_score, greater_is_better=True),
            },
            n_jobs=1,
            verbose=2,
        )
        cv_metrics = mean_cv_scores(cv_metrics)
        logger.info(f"CV metrics: {cv_metrics}")

    metrics = test_metrics
    if cv:
        metrics.update(cv_metrics)

    return {
        "model": model,
        "input_example": X_test.iloc[:10],
        "prediction": prediction[:10],
        "metrics": metrics,
    }
