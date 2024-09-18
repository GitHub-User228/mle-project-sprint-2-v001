import mlflow
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature

from scripts import logger


def get_experiment_id(
    experiment_name: str, client: mlflow.tracking.client.MlflowClient
) -> str:
    """
    Get the ID of an MLflow experiment, creating a new one if it
    doesn't exist. Also restores the experiment if it is not active.

    Args:
        experiment_name (str):
            The name of the MLflow experiment.
        client (mlflow.tracking.client.MlflowClient):
            The MLflow client to use for interacting with the MLflow
            tracking server.

    Returns:
        str:
            The ID of the MLflow experiment.
    """

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.info(
            f"Experiment '{experiment_name}' not found. "
            "Creating a new experiment..."
        )
        try:
            experiment_id = mlflow.create_experiment(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_name)
        except Exception as e:
            msg = f"Failed to create experiment '{experiment_name}'"
            logger.error(msg)
            raise Exception(msg) from e
    else:
        logger.info(f"Experiment '{experiment_name}' exists")
        experiment_id = experiment.experiment_id

    if experiment.lifecycle_stage != "active":
        logger.info(
            f"Experiment '{experiment_name}' is not active. Current state: "
            f"{experiment.lifecycle_stage}. Restoring..."
        )
        client.restore_experiment(experiment_id)

    return experiment_id


def log_model_info(
    model: object,
    registry_model_name: str,
    model_loader: mlflow.utils.lazy_load.LazyLoader,
    input_example: np.ndarray | pd.DataFrame,
    prediction: np.ndarray,
    pip_requirements: str | list[str] = "../requirements.txt",
    metrics: dict | None = None,
    params: dict | None = None,
    metadata: dict | None = {"model_type": "monthly"},
    await_registration_for: int | None = 60,
) -> None:
    """
    Logs a machine learning model to the MLflow model registry
    with a parameters and metrics.

    Args:
        model (object):
            The trained machine learning model to be logged.
        registry_model_name (str):
            The name of the registered model in the MLflow model
            registry.
        model_loader (mlflow.utils.lazy_load.LazyLoader):
            The MLflow model loader to use for logging the model.
        input_example (np.ndarray):
            A sample input example to be used for the model signature.
        prediction (np.ndarray):
            A sample prediction from the model to be used for the
            model signature.
        pip_requirements (str | list[str], optional):
            The path to the pip requirements file or a list of pip
            requirements.
            Defaults to "../requirements.txt".
        metrics (dict, optional):
            A dictionary of metrics to be logged with the model.
        params (dict, optional):
            A dictionary of parameters to be logged with the model.
        metadata (dict, optional):
            A dictionary of metadata to be associated with the model.
            Defaults to {"model_type": "monthly"}.
        await_registration_for (int, optional):
            The number of seconds to wait for the model to be
            registered. Defaults to 60.
    """

    if metrics:
        mlflow.log_metrics(metrics)
        logger.info(f"Metrics for model {registry_model_name} has been logged")
    else:
        logger.warning(f"No metrics provided for model {registry_model_name}")
    if params:
        mlflow.log_params(params)
        logger.info(
            f"Paremeters for model {registry_model_name} has been logged"
        )
    else:
        logger.warning(
            f"No parameters provided for model {registry_model_name}"
        )

    try:
        _ = model_loader.log_model(
            model,
            artifact_path="models",
            registered_model_name=registry_model_name,
            signature=infer_signature(input_example, prediction),
            input_example=input_example,
            metadata=metadata,
            await_registration_for=await_registration_for,
            pip_requirements=pip_requirements,
        )
        logger.info(f"Model {registry_model_name} has been logged")
    except Exception as e:
        msg = f"Failed to log model {registry_model_name}"
        logger.error(msg)
        raise Exception(msg) from e
