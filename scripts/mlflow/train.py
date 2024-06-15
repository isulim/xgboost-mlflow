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
from scripts.mlflow.utils import TrialParameters, download_data, get_metric, prepare_trial_params

logger = logging.getLogger("train.py")


def objective(
        trial,
        model_type: str,
        additional_metrics: list[str],
        train_val_ds: list[pd.DataFrame],
        params: dict | TrialParameters = TrialParameters()
):
    with mlflow.start_run(nested=True):
        parameters = prepare_trial_params(trial, params)

        mlflow.log_params(parameters)
        X_train, y_train, X_val, y_val = train_val_ds
        model = prepare_model(model_type, parameters)
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


def train_model(
        base_path: str = "data",
        model_type: str = "xgb-random-forest",
        n_trials: int = 10,
        additional_metrics: list[str] = (),
        mlflow_host: str = "localhost",
        mlflow_port: int = 8000,
        experiment_name: str = "XGBoostRF",
        split_size: float = 0.1,
        seed: int = 42,

) -> None:
    # Input data:
    # - raw data -> to preprocess
    # - data_path
    # - model -> XGB Classifier or XGB RF Classifier
    # - experiments: MLFlow URI, experiment name, target metric, additional metrics (display list)
    # - model hyperparams: **kwargs

    # 0. download data
    if not Path(base_path, "raw").exists():
        base_path: Path = download_data()

    # 1. preprocess dataset
    if not Path(base_path, "preprocessed").exists():
        preprocess_data(base_path)

    # 2. split dataset
    if not Path(base_path, "train").exists():
        split_and_save(base_path, split_size=split_size, seed=seed)

    X_train: pd.DataFrame = pd.read_csv(Path(base_path, "train", "X.csv"))
    y_train: pd.DataFrame = pd.read_csv(Path(base_path, "train", "y.csv"))
    X_val: pd.DataFrame = pd.read_csv(Path(base_path, "val", "X.csv"))
    y_val: pd.DataFrame = pd.read_csv(Path(base_path, "val", "y.csv"))
    train_val_ds = [X_train, y_train, X_val, y_val]

    # 3. set mlflow experiment
    mlflow_uri: str = f"http://{mlflow_host}:{mlflow_port}"

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.autolog(
        log_input_examples=True,
    )

    # 4. set optuna study
    study_name: str = experiment_name.lower().replace(" ", "-")

    study_params = {
        "study_name": study_name,
        "storage": "sqlite:///mlflow.db",
        "direction": "maximize",
        "load_if_exists": True,
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
