"""Load data and split to train/validate/test sets."""

import logging
import os
from pathlib import Path

import xgboost as xgb
import pandas as pd
import mlflow
import optuna

from sklearn.model_selection import train_test_split
from scripts.data import preprocess_data
from scripts.mlflow import CLASSIFICATION_METRICS
from scripts.mlflow.utils import additional_metric, prepare_model, load_data

logger = logging.getLogger("train.py")


def objective(
        trial,
        model_type: str,
        target_metric_name: str,
        metrics: list[str],
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
        y_pred = model.predict(X_val)

        if not target_metric_name:
            target_metric_name = "accuracy_score"

        try:
            target_metric_func = CLASSIFICATION_METRICS[target_metric_name]
        except KeyError:
            raise KeyError(
                "Invalid target metric - must be a valid sklearn classification metric function."
            )

        target = target_metric_func(y_val, y_pred)

        for metric_name in metrics:
            metric_value = additional_metric(
                y_val, y_pred, metric_name, target_metric_func
            )
            if metric_value:
                mlflow.log_metric(metric_name, metric_value)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return target


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
        base_path: Path = load_data()

    # 1. preprocess dataset
    if not Path(base_path, "preprocessed").exists():
        preprocess_data(base_path)
    X: pd.DataFrame = pd.read_csv(Path(base_path, "preprocessed", "X_pre.csv"))
    y: pd.DataFrame = pd.read_csv(Path(base_path, "preprocessed", "y_pre.csv"))

    # 2. split dataset
    split_size: float = 0.1
    seed: int = 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=split_size, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=split_size, random_state=seed
    )

    # 3. set mlflow experiment
    MLFLOW_URI: str = "http://localhost:8000"
    experiment_name: str = "XGBoost Random Forest"

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment_name)
    mlflow.autolog()  # logging config

    # 4. set optuna study
    study_name: str = "xgboost-random-forest"
    storage: str = "sqlite:///mlflow.db"
    direction: str = "maximize"
    load_if_exists: bool = True

    model_type: str = "random-forest"
    n_trials: int = 10
    target_metric: str = "accuracy_score"
    metrics = ["f1_score", "roc_auc_score"]

    study_params = {
        "study_name": study_name,
        "storage": storage,
        "direction": direction,
        "load_if_exists": load_if_exists,
    }

    train_val_ds = [X_train, y_train, X_val, y_val]
    obj_func = lambda trial: objective(
        trial, model_type, target_metric, metrics, train_val_ds
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

        signature = mlflow.models.infer_signature(X_train, model.predict(X_train))
        mlflow.xgboost.log_model(
            model,
            f"{study_name}.json",
            signature=signature,
            input_example=X_train[:5],
            registered_model_name=f"{study_name}.model",
        )
        mlflow.log_params(best_trial.params)
        mlflow.log_metric(target_metric, best_trial.value)
