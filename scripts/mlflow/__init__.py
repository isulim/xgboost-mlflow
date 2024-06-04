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
