import logging
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, Extra, ValidationError, model_validator

from scripts.data.configure_envs import configure_envs
from scripts.data.download_data import data_download
from scripts.mlflow.models import CLASSIFICATION_METRICS

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


class ClassifierParam(BaseModel):
    """Base model of single classifier parameter."""

    name: str
    value_type: str = Field("float", pattern="^(int|float|str|bool)$")
    values: Optional[list[str | int | float | bool]] = None
    start: Optional[float | int] = Field(None, ge=0)
    end: Optional[float | int] = Field(None, ge=0)
    step: Optional[float | None] = Field(None, ge=0)
    log: Optional[bool] = False

    class Config:
        extra = "ignore"

    @model_validator(mode="after")
    def validate(cls, params):
        if params.value_type in ["str", "bool"]:
            if not params.values:
                raise ValidationError("Field `values` must be provided for string and bool type parameters.")
            if params.start or params.end or params.step or params.log:
                raise ValidationError(
                    "Fields `start`, `end`, `step`, `log` are not allowed for string and bool type parameters.")

        if params.values and (params.start or params.end or params.step or params.log):
            raise ValidationError("Fields `values` and `start`, `end`, `step`, `log` are mutually exclusive.")

        if (params.start and params.end is None) or (params.end and params.start is None):
            raise ValidationError("Fields `start` and `end` must be provided together.")

        if params.start and params.start and params.start >= params.end:
            raise ValidationError("Field `start` must be less than `end`.")


class ClassifierParams(BaseModel):
    """Model of classifier parameters."""

    objective: ClassifierParam = Field(ClassifierParam(name="objective", values=["binary:logistic"]))
    eta: ClassifierParam = Field(ClassifierParam(name="eta", start=0.2, end=0.3, step=0.01))
    gamma: ClassifierParam = Field(ClassifierParam(name="gamma", start=0, end=0.1, step=0.01))
    max_depth: ClassifierParam = Field(ClassifierParam(name="max_depth", start=4, end=6, step=1))
    min_child_weight: ClassifierParam = Field(ClassifierParam(name="min_child_weight", start=1, end=3, step=1))
    max_delta_step: ClassifierParam = Field(ClassifierParam(name="max_delta_step", start=0, end=0.1, step=0.01))
    subsample: ClassifierParam = Field(ClassifierParam(name="subsample", start=0.7, end=0.9, step=0.1))
    colsample_bytree: ClassifierParam = Field(ClassifierParam(name="colsample_bytree", start=0.7, end=0.9, step=0.1))
    lambda_: ClassifierParam = Field(ClassifierParam(name="reg_lambda", start=1, end=3, step=1))
    alpha: ClassifierParam = Field(ClassifierParam(name="reg_alpha", start=0, end=1, step=0.1))
    n_estimators: ClassifierParam = Field(ClassifierParam(name="n_estimators", start=50, end=150, step=50))
    max_leaf_nodes: ClassifierParam = Field(ClassifierParam(name="max_leaf_nodes", start=3, end=7, log=True))
    scale_pos_weight: ClassifierParam = Field(ClassifierParam(name="scale_pos_weight", values=[1.0]))
    random_state: int = Field(42, ge=0)

    class Config:
        extra = "ignore"


def trial_params(trial, params: ClassifierParams) -> dict | None:
    """Prepare parameters for optuna trial input."""

    params_dict = {}

    for param in params.dict().values():
        if not isinstance(param, ClassifierParam):
            params_dict[param.name] = param
        if len(param.values) == 1:
            params_dict[param.name] = param.values[0]
        else:
            params_dict[param.name] = trial.suggest_categorical(param.name, param.values)

    return params_dict
