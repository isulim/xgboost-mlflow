import logging
import os
from pathlib import Path

import xgboost as xgb

from scripts.data.configure_envs import configure_envs
from scripts.data.download_data import data_download
from scripts.mlflow import CLASSIFICATION_METRICS

logger = logging.getLogger("utils.py")


def load_data() -> Path:
    """Load data from Kaggle and return base path to data directory."""

    configure_envs()
    base_path = Path(os.getenv("KAGGLE_FILES_DIR"))
    data_download(Path(base_path, "raw"))
    return base_path


def prepare_model(model_type: str, params: dict) -> xgb.XGBModel:
    if model_type == "classifier":
        return xgb.XGBClassifier(**params)
    elif model_type == "random-forest":
        return xgb.XGBRFClassifier(**params)
    else:
        raise ValueError("Model type must be either 'classifier' or 'random-forest'.")


def additional_metric(y_val, y_pred, metric_name: str, target_metric_func):
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
