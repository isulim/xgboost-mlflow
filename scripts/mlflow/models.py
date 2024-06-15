from xgboost import XGBClassifier, XGBRFClassifier

from sklearn.metrics import *
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, model_validator


CLASSIFICATION_METRICS = {
    "accuracy_score": accuracy_score,
    "balanced_accuracy_score": balanced_accuracy_score,
    "average_precision_score": average_precision_score,
    "brier_score_loss": brier_score_loss,
    "f1_score": f1_score,
    "log_loss": log_loss,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "confusion_matrix": confusion_matrix,
}


def prepare_model(model_type: str, params: dict) -> XGBClassifier | XGBRFClassifier:
    if model_type == "xgb-classifier":
        return XGBClassifier(**params)
    elif model_type == "xgb-random-forest":
        return XGBRFClassifier(**params)
    else:
        raise ValueError("Invalid model type. Use one of: 'xgb-classifier', 'xgb-random-forest'.")


class TrialParameter(BaseModel):
    """Base model of single optuna trial parameter."""

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


class TrialParameters(BaseModel):
    """Model of optuna trial parameters."""

    objective: TrialParameter = Field(TrialParameter(name="objective", values=["binary:logistic"]))
    eta: TrialParameter = Field(TrialParameter(name="eta", start=0.2, end=0.3, step=0.01))
    gamma: TrialParameter = Field(TrialParameter(name="gamma", start=0, end=0.1, step=0.01))
    max_depth: TrialParameter = Field(TrialParameter(name="max_depth", start=4, end=6, step=1))
    min_child_weight: TrialParameter = Field(TrialParameter(name="min_child_weight", start=1, end=3, step=1))
    max_delta_step: TrialParameter = Field(TrialParameter(name="max_delta_step", start=0, end=0.1, step=0.01))
    subsample: TrialParameter = Field(TrialParameter(name="subsample", start=0.7, end=0.9, step=0.1))
    colsample_bytree: TrialParameter = Field(TrialParameter(name="colsample_bytree", start=0.7, end=0.9, step=0.1))
    lambda_: TrialParameter = Field(TrialParameter(name="reg_lambda", start=1, end=3, step=1))
    alpha: TrialParameter = Field(TrialParameter(name="reg_alpha", start=0, end=1, step=0.1))
    n_estimators: TrialParameter = Field(TrialParameter(name="n_estimators", start=50, end=150, step=50))
    max_leaf_nodes: TrialParameter = Field(TrialParameter(name="max_leaf_nodes", start=3, end=7, log=True))
    scale_pos_weight: TrialParameter = Field(TrialParameter(name="scale_pos_weight", values=[1.0]))
    random_state: int = Field(42, ge=0)

    class Config:
        extra = "ignore"
