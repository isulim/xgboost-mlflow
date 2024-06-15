import logging
import os
from pathlib import Path
from pydantic import BaseModel, Field

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


class ClassifierParams(BaseModel):
    """Dataclass for classifier parameters."""

    objective: str = Field("binary:logistic")
    eta: float = Field(0.3, ge=0, le=1.0, alias="learning_rate")
    gamma: float = Field(0, ge=0, alias="min_split_loss")
    max_depth = Field(6, ge=0)
    min_child_weight: int = Field(1, ge=0)
    max_delta_step: float = Field(0.0)
    subsample: float = Field(0.8, gt=0, le=1.0)
    colsample_bytree: float = Field(0.8, gt=0, le=1.0)
    lambda_: float = Field(1.0, ge=0, alias="reg_lambda")
    alpha: float = Field(0.0, ge=0, alias="reg_alpha")
    n_estimators: int = Field(100, gt=0, le=1000)
    max_leaf_nodes: int = Field(None, ge=0)
    scale_pos_weight: float = Field(1.0, ge=0)
    seed: int = Field(42, ge=0)


def get_params(params: dict) -> ClassifierParams:
    """Get parameters for model training."""

    parameters = ClassifierParams(**params)

    return parameters
