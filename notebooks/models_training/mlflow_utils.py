
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import clone

from nlp_disaster_utils import (
    classification_metrics_from_predictions,
)

def setup_mlflow_tracking(
    experiment_name: str,
    tracking_uri: str | None = None,
) -> str:
    """
    Configure MLflow tracking for a notebook.

    If no tracking_uri is provided, a local file-based store is used in ../../outputs/mlruns
    relative to the notebook location.
    """
    if tracking_uri is None:
        tracking_uri = str(Path("../../outputs/mlruns").resolve())

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def safe_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


def evaluate_fitted_sklearn_model(
    name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
) -> Dict[str, float]:
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_score = safe_scores(model, X_train)
    test_score = safe_scores(model, X_test)

    result: Dict[str, float] = {"pipeline": name}
    result.update(
        classification_metrics_from_predictions(
            y_train, train_pred, train_score, prefix="train"
        )
    )
    result.update(
        classification_metrics_from_predictions(
            y_test, test_pred, test_score, prefix="test"
        )
    )
    return result


def get_loggable_params(estimator) -> Dict[str, str]:
    """
    Return a simplified dict of estimator params that MLflow can log safely.
    """
    params = estimator.get_params(deep=True)
    out: Dict[str, str] = {}
    for key, value in params.items():
        if isinstance(value, (str, int, float, bool, type(None))):
            out[key] = value
        else:
            out[key] = str(value)
    return out


def log_metrics_to_mlflow(metrics: Dict[str, Any], metric_columns: list[str] | None = None) -> None:
    if metric_columns is None:
        metric_columns = [
            "train_accuracy", "test_accuracy",
            "train_precision_macro", "test_precision_macro",
            "train_recall_macro", "test_recall_macro",
            "train_f1_macro", "test_f1_macro",
            "train_precision_weighted", "test_precision_weighted",
            "train_recall_weighted", "test_recall_weighted",
            "train_f1_weighted", "test_f1_weighted",
            "train_precision_class_0", "test_precision_class_0",
            "train_recall_class_0", "test_recall_class_0",
            "train_f1_class_0", "test_f1_class_0",
            "train_precision_class_1", "test_precision_class_1",
            "train_recall_class_1", "test_recall_class_1",
            "train_f1_class_1", "test_f1_class_1",
            "train_balanced_accuracy", "test_balanced_accuracy",
            "train_roc_auc", "test_roc_auc",
            "train_pr_auc", "test_pr_auc",
        ]
    for col in metric_columns:
        if col in metrics and pd.notna(metrics[col]):
            try:
                mlflow.log_metric(col, float(metrics[col]))
            except Exception:
                pass


def log_run_artifacts(
    metrics: Dict[str, Any],
    output_dir: str | Path,
    run_name: str,
    extra_artifacts: Optional[list[str | Path]] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_csv = output_dir / f"{run_name}_run_metrics.csv"
    pd.DataFrame([metrics]).to_csv(run_csv, index=False)
    mlflow.log_artifact(str(run_csv), artifact_path="tables")

    if extra_artifacts:
        for artifact in extra_artifacts:
            artifact_path = Path(artifact)
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path), artifact_path="extra_artifacts")
    return run_csv


def fit_evaluate_and_log_sklearn_pipeline(
    name: str,
    estimator,
    X_train,
    X_test,
    y_train,
    y_test,
    notebook_name: str,
    family_name: str,
    output_dir: str | Path,
    log_model: bool = False,
    extra_tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    model = clone(estimator)
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)

        metrics = evaluate_fitted_sklearn_model(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        mlflow.set_tag("notebook", notebook_name)
        mlflow.set_tag("family", family_name)
        mlflow.set_tag("pipeline_name", name)

        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, v)

        mlflow.log_params(get_loggable_params(model))
        log_metrics_to_mlflow(metrics)
        log_run_artifacts(metrics, output_dir=output_dir, run_name=name)

        if log_model:
            try:
                mlflow.sklearn.log_model(model, artifact_path="model")
            except Exception as exc:
                mlflow.set_tag("model_logging_error", str(exc))

        return metrics
