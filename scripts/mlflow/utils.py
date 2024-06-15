import logging
import os
from dataclasses import dataclass, field
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


@dataclass
class ClassifierParams:
    """Dataclass for classifier parameters."""

    objective: str = field(default="binary:logistic")
    eta: float = field(default=0.3, metadata={"alias": "learning_rate"})
    gamma: float = field(default=0, metadata={"alias": "min_split_loss"})
    max_depth: int = field(default=6)
    min_child_weight: int = field(default=1)
    max_delta_step: float = field(default=0.0)
    subsample: float = field(default=0.8)
    colsample_bytree: float = field(default=0.8)
    lambda_: float = field(default=1.0, metadata={"alias": "reg_lambda"})
    alpha: float = field(default=0.0, metadata={"alias": "reg_alpha"})
    n_estimators: int = field(default=100)
    max_leaf_nodes: int = field(default=None)
    scale_pos_weight: float = field(default=1.0)
    seed: int = field(default=42)


def get_params(params: dict) -> ClassifierParams:
    """Get parameters for model training."""

    parameters = ClassifierParams(**params)

    return parameters
