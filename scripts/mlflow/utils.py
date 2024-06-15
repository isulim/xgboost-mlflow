import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import optuna
from optuna import Trial

from scripts.data.configure_envs import configure_envs
from scripts.data.download_data import data_download
from scripts.mlflow.models import CLASSIFICATION_METRICS, TrialParameter, TrialParameters

logger = logging.getLogger()


def download_data() -> Path:
    """Load data from Kaggle and return base path to data directory."""

    configure_envs()
    base_path = Path(os.getenv("KAGGLE_FILES_DIR"))
    data_download(Path(base_path, "raw"))
    return base_path


def get_metric(y_val, y_pred, metric_name: str):
    """Calculate additional metric for the model."""

    try:
        metric_func = CLASSIFICATION_METRICS[metric_name]

    except KeyError:
        logger.warning(
            f"Invalid metric: {metric_name} - must be a valid sklearn metric function (`roc_auc_score` is the target metric)."
        )

    else:
        return metric_func(y_val, y_pred)


def prepare_trial_params(trial: optuna.Trial, params: TrialParameters) -> dict | None:
    """Prepare parameters for optuna trial input."""

    params_dict = {}
    for param_name in params.dict():
        parameter = params.dict()[param_name]
        if not isinstance(parameter, dict):
            params_dict[param_name] = parameter
            continue

        if (values := parameter.get("values")) is not None:
            if len(values) == 1:
                params_dict[param_name] = values[0]
                continue
            else:
                params_dict[param_name] = trial.suggest_categorical(name=param_name, choices=values)
                continue
        else:
            parameter_type = parameter.get("value_type")
            numerical_params = {
                "name": param_name,
                "low": parameter.get("start"),
                "high": parameter.get("end"),
                "step": parameter.get("step"),
                "log": parameter.get("log"),
            }
            if parameter_type == "int":
                params_dict[param_name] = trial.suggest_int(**numerical_params)
                continue
            elif parameter_type == "float":
                params_dict[param_name] = trial.suggest_float(**numerical_params)
                continue

    return params_dict
