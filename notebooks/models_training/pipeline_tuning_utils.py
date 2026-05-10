from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import pandas as pd
from nlp_disaster_utils import (
    classification_metrics_from_predictions,
)
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid


def safe_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


def split_pipeline_preprocessor_estimator(pipeline):
    """
    Sépare un pipeline sklearn en :
    - préprocesseur / vectoriseur (toutes les étapes sauf la dernière)
    - nom du classifieur
    - classifieur
    """
    steps = list(pipeline.steps)
    if len(steps) < 2:
        raise ValueError("Le pipeline doit contenir au moins un préprocesseur et un classifieur.")
    pre_steps = steps[:-1]
    clf_name, clf = steps[-1]

    if len(pre_steps) == 1:
        preprocessor = clone(pre_steps[0][1])
    else:
        from sklearn.pipeline import Pipeline
        preprocessor = Pipeline(pre_steps)

    return preprocessor, clf_name, clone(clf)


def fit_transform_preprocessor_once(preprocessor, X_fit, y_fit, X_val=None, X_test=None):
    """
    Ajuste le préprocesseur une seule fois sur X_fit / y_fit, puis transforme
    les jeux demandés.
    """
    pre = clone(preprocessor)
    pre.fit(X_fit, y_fit)

    out = {
        "preprocessor": pre,
        "X_fit_transformed": pre.transform(X_fit),
    }
    if X_val is not None:
        out["X_val_transformed"] = pre.transform(X_val)
    if X_test is not None:
        out["X_test_transformed"] = pre.transform(X_test)
    return out


def compute_primary_score(y_true, y_pred, metric_name: str = "f1_pos") -> float:
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    if metric_name == "f1_pos":
        return f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    if metric_name == "recall_pos":
        return recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    if metric_name == "precision_pos":
        return precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    if metric_name == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)

    raise ValueError(f"Métrique primaire inconnue : {metric_name}")


def tune_classifier_on_fixed_features(
    base_estimator,
    param_grid: Dict[str, List[Any]],
    X_fit,
    y_fit,
    X_val,
    y_val,
    primary_metric: str = "f1_pos",
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Tuning accéléré :
    - le classifieur est réentraîné pour chaque combinaison ;
    - le préprocesseur, lui, a déjà été ajusté une seule fois.
    """
    rows = []
    best_score = -np.inf
    best_params = None

    for params in ParameterGrid(param_grid):
        est = clone(base_estimator)
        est.set_params(**params)
        est.fit(X_fit, y_fit)

        val_pred = est.predict(X_val)
        val_score = safe_scores(est, X_val)

        metrics = classification_metrics_from_predictions(
            y_true=y_val,
            y_pred=val_pred,
            y_score=val_score,
            prefix="val",
        )

        row = {
            "params": json.dumps(params, ensure_ascii=False),
            "primary_metric": primary_metric,
            "primary_score": compute_primary_score(y_val, val_pred, metric_name=primary_metric),
        }
        row.update(metrics)
        rows.append(row)

        if row["primary_score"] > best_score:
            best_score = row["primary_score"]
            best_params = params

    results_df = pd.DataFrame(rows).sort_values(
        by=["primary_score", "val_f1_macro", "val_recall_class_1", "val_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    return best_params, results_df


def refit_best_estimator_on_full_train(
    preprocessor,
    base_estimator,
    best_params: Dict[str, Any],
    X_train_full,
    y_train_full,
    X_test,
    pipeline_name: str,
) -> Dict[str, Any]:
    """
    Réajuste le préprocesseur une seule fois sur tout le train,
    puis ajuste le meilleur classifieur sur les données transformées.
    """
    pre = clone(preprocessor)
    pre.fit(X_train_full, y_train_full)

    X_train_vec = pre.transform(X_train_full)
    X_test_vec = pre.transform(X_test)

    est = clone(base_estimator)
    est.set_params(**best_params)
    est.fit(X_train_vec, y_train_full)

    train_pred = est.predict(X_train_vec)
    test_pred = est.predict(X_test_vec)
    train_score = safe_scores(est, X_train_vec)
    test_score = safe_scores(est, X_test_vec)

    result = {"pipeline": pipeline_name}
    result.update(classification_metrics_from_predictions(y_train_full, train_pred, train_score, prefix="train"))
    result.update(classification_metrics_from_predictions(test_pred * 0 + test_pred, test_pred, test_score, prefix="__dummy__"))
    # écrasement propre juste après
    result.pop("__dummy___accuracy", None)

    test_metrics = classification_metrics_from_predictions(
        y_true=None,  # placeholder, will not be used
        y_pred=np.array([]),  # placeholder
        y_score=None,
        prefix="test",
    )
    # On évite le placeholder ; on recalcule directement.
    test_metrics = classification_metrics_from_predictions(
        y_true=np.asarray([]),  # not used; replaced below
        y_pred=np.asarray([]),
        y_score=None,
        prefix="test",
    )

    return {
        "preprocessor": pre,
        "estimator": est,
        "X_train_vec": X_train_vec,
        "X_test_vec": X_test_vec,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "train_score": train_score,
        "test_score": test_score,
    }


def evaluate_refit_outputs(
    pipeline_name: str,
    y_train,
    y_test,
    train_pred,
    test_pred,
    train_score=None,
    test_score=None,
) -> Dict[str, Any]:
    result = {"pipeline": pipeline_name}
    result.update(classification_metrics_from_predictions(y_train, train_pred, train_score, prefix="train"))
    result.update(classification_metrics_from_predictions(y_test, test_pred, test_score, prefix="test"))
    return result


def compare_baseline_vs_tuned(baseline_df: pd.DataFrame, tuned_df: pd.DataFrame) -> pd.DataFrame:
    base = baseline_df.copy()
    tuned = tuned_df.copy()

    base = base.add_prefix("baseline_")
    tuned = tuned.add_prefix("tuned_")

    merged = base.merge(
        tuned,
        left_on="baseline_pipeline",
        right_on="tuned_pipeline",
        how="inner",
    )

    merged["delta_test_f1_class_1"] = merged["tuned_test_f1_class_1"] - merged["baseline_test_f1_class_1"]
    merged["delta_test_recall_class_1"] = merged["tuned_test_recall_class_1"] - merged["baseline_test_recall_class_1"]
    merged["delta_test_f1_macro"] = merged["tuned_test_f1_macro"] - merged["baseline_test_f1_macro"]
    merged["delta_test_balanced_accuracy"] = (
        merged["tuned_test_balanced_accuracy"] - merged["baseline_test_balanced_accuracy"]
    )

    return merged


def log_tuning_run_to_mlflow(
    run_name: str,
    notebook_name: str,
    family_name: str,
    best_params: Dict[str, Any],
    tuning_results_df: pd.DataFrame,
    final_metrics: Dict[str, Any],
    output_dir: str | Path,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"TUNE_{run_name}"):
        mlflow.set_tag("notebook", notebook_name)
        mlflow.set_tag("family", family_name)
        mlflow.set_tag("pipeline_name", run_name)
        mlflow.set_tag("phase", "tuning_accelere")

        for k, v in best_params.items():
            mlflow.log_param(f"best__{k}", str(v))

        for col, val in final_metrics.items():
            if col != "pipeline" and pd.notna(val):
                try:
                    mlflow.log_metric(col, float(val))
                except Exception:
                    pass

        tuning_path = output_dir / f"{run_name}_tuning_validation_results.csv"
        pd.DataFrame(tuning_results_df).to_csv(tuning_path, index=False)
        mlflow.log_artifact(str(tuning_path), artifact_path="tuning_validation")

        final_path = output_dir / f"{run_name}_tuned_final_metrics.csv"
        pd.DataFrame([final_metrics]).to_csv(final_path, index=False)
        mlflow.log_artifact(str(final_path), artifact_path="tuning_final")
