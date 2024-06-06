from xgboost import XGBModel, XGBClassifier, XGBRFClassifier

from sklearn.metrics import *

CLASSIFICATION_METRICS = {
    "accuracy_score": accuracy_score,
    "balanced_accuracy_score": balanced_accuracy_score,
    "average_precision_score": average_precision_score,
    "brier_score_loss": brier_score_loss,
    "f1_score": f1_score,
    "log_loss": log_loss,
    "precision_score": precision_score,
    "recall_score": recall_score,
    "roc_auc_score": roc_auc_score,
    "confusion_matrix": confusion_matrix,
}


def prepare_model(model_type: str, params: dict) -> XGBModel:
    if model_type == "classifier":
        return XGBClassifier(**params)
    elif model_type == "random-forest":
        return XGBRFClassifier(**params)
    else:
        raise ValueError("Model type must be either 'classifier' or 'random-forest'.")
