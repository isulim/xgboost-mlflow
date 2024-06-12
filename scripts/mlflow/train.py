"""Load data and split to train/validate/test sets."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from scripts.data import preprocess_data, split_and_save
from scripts.mlflow.models import prepare_model
from scripts.mlflow.utils import download_data, get_metric

logger = logging.getLogger("train.py")


def objective(
        trial,
        model_type: str,
        additional_metrics: list[str],
        train_val_ds: list[pd.DataFrame],
):
    with mlflow.start_run(nested=True):
        params = {
            "objective": "binary:logistic",
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=50),
            "max_leaves": trial.suggest_int("max_leaves", 3, 7, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10, log=True),
            "random_state": 42,
        }
        mlflow.log_params(params)
        X_train, y_train, X_val, y_val = train_val_ds

        model = prepare_model(model_type, params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        probs = model.predict_proba(X_val)

        y_pred = probs[:, 1]
        target_val = roc_auc_score(y_val, y_pred)
        mlflow.log_metric("roc_auc_score", target_val)

        for metric_name in additional_metrics:
            metric_value = get_metric(
                y_val, y_pred, metric_name
            )
            if metric_value:
                mlflow.log_metric(metric_name, metric_value)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return target_val


def train_model() -> None:
    # Input data:
    # - raw data -> to preprocess
    # - data_path
    # - model -> XGB Classifier or XGB RF Classifier
    # - experiments: MLFlow URI, experiment name, target metric, additional metrics (display list)
    # - model hyperparams: **kwargs

    # 0. load data (optional download with Kaggle API?)
    base_path = Path(os.getenv("KAGGLE_FILES_DIR"))
    if not Path(base_path, "raw").exists():
        base_path: Path = download_data()

    # 1. preprocess dataset
    if not Path(base_path, "preprocessed").exists():
        preprocess_data(base_path)

    # 2. split dataset
    split_size: float = 0.1
    seed: int = 42
    if not Path(base_path, "train").exists():
        split_and_save(base_path, split_size=split_size, seed=seed)

    X_train: pd.DataFrame = pd.read_csv(Path(base_path, "train", "X.csv"))
    y_train: pd.DataFrame = pd.read_csv(Path(base_path, "train", "y.csv"))
    X_val: pd.DataFrame = pd.read_csv(Path(base_path, "val", "X.csv"))
    y_val: pd.DataFrame = pd.read_csv(Path(base_path, "val", "y.csv"))
    train_val_ds = [X_train, y_train, X_val, y_val]

    # 3. set mlflow experiment
    MLFLOW_URI: str = "http://localhost:8000"
    experiment_name: str = "XGBoost Random Forest"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()  # logging config

    # 4. set optuna study
    study_name: str = experiment_name.lower().replace(" ", "-")
    storage: str = "sqlite:///mlflow.db"
    direction: str = "maximize"
    load_if_exists: bool = True

    model_type: str = "random-forest"
    n_trials: int = 2
    target_metric: str = "accuracy_score"
    metrics = ["f1_score", "roc_auc_score", "precision_score", "recall_score"]

    study_params = {
        "study_name": study_name,
        "storage": storage,
        "direction": direction,
        "load_if_exists": load_if_exists,
    }

    obj_func = lambda trial: objective(
        trial, model_type, additional_metrics, train_val_ds
    )
    # 5. optimize hyperparams
    with mlflow.start_run():
        study = optuna.create_study(**study_params)
        study.optimize(obj_func, n_trials=n_trials, n_jobs=1, gc_after_trial=True)

        best_trial = study.best_trial
        print(f"Best trial (score): {best_trial.value}")
        print("Best trial (params):")
        for k, v in best_trial.params.items():
            print(f"    {k}: {v}")

        model = prepare_model(model_type, best_trial.params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        probs = model.predict_proba(X_val)
        y_pred = probs[:, 1]

        fpr, tpr, thresholds = roc_curve(y_val, y_pred)
        fig = plt.figure(figsize=(12, 8))

        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.savefig("ROC-Curve.png")
        mlflow.log_params(best_trial.params)
        mlflow.log_metric("roc_auc_score", best_trial.value)
        mlflow.log_artifact("ROC-Curve.png")
