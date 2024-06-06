import logging
import os
from pathlib import Path

from scripts.data.configure_envs import configure_envs
from scripts.data.download_data import data_download
from scripts.mlflow.models import CLASSIFICATION_METRICS

logger = logging.getLogger("utils.py")


def download_data() -> Path:
    """Load data from Kaggle and return base path to data directory."""

    configure_envs()
    base_path = Path(os.getenv("KAGGLE_FILES_DIR"))
    data_download(Path(base_path, "raw"))
    return base_path


def get_target_metric(target_metric_name: str):
    """Return target metric function for the model."""

    if not target_metric_name:
        target_metric_name = "accuracy_score"
    try:
        target_metric_func = CLASSIFICATION_METRICS[target_metric_name]
    except KeyError:
        raise KeyError(
            "Invalid target metric - must be a valid sklearn classification metric function."
        )
    else:
        return target_metric_func


def get_additional_metric(y_val, y_pred, metric_name: str, target_metric_func):
    """Calculate additional metric for the model."""

    try:
        metric_func = CLASSIFICATION_METRICS[metric_name]

    except KeyError:
        logger.warning(
            f"Invalid metric: {metric_name} - must be a valid sklearn metric function."
        )
        return
    else:
        if metric_func == target_metric_func:
            logger.warning(f"Metric {metric_name} is used as the target metric.")
        else:
            return metric_func(y_val, y_pred)
