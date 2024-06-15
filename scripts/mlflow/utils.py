import logging
import os
from pathlib import Path


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


def trial_params(trial, params: TrialParameters) -> dict | None:
    """Prepare parameters for optuna trial input."""

    params_dict = {}

    for param in params.dict().values():
        if not isinstance(param, TrialParameter):
            params_dict[param.name] = param
        if len(param.values) == 1:
            params_dict[param.name] = param.values[0]
        else:
            params_dict[param.name] = trial.suggest_categorical(param.name, param.values)

    return params_dict
