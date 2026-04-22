from __future__ import annotations

import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


class ResultsDisplayFrame(pd.DataFrame):
    """
    DataFrame personnalisé :
    - reste utilisable comme un DataFrame classique ;
    - dans Jupyter, il s'affiche automatiquement dans une vue stylée.

    Modes d'affichage :
    - "results"       : pipelines en colonnes, métriques en lignes
    - "metric_matrix" : matrice déjà transposée, stylée telle quelle
    """
    _metadata = ["_display_mode", "_round_digits"]

    @property
    def _constructor(self):
        return ResultsDisplayFrame

    def _build_display_view(self) -> pd.DataFrame:
        base_df = pd.DataFrame(self).copy()
        mode = getattr(self, "_display_mode", None)

        if mode == "results":
            if "pipeline" in base_df.columns:
                return base_df.set_index("pipeline").T
            return base_df

        if mode == "metric_matrix":
            return base_df

        return base_df

    def _repr_html_(self):
        view = self._build_display_view()
        digits = getattr(self, "_round_digits", 4)
        return style_metric_matrix(view, digits=digits).to_html()

    def __repr__(self):
        return repr(pd.DataFrame(self))


def _wrap_results_display(df: pd.DataFrame, display_mode: str, digits: int = 4) -> ResultsDisplayFrame:
    out = ResultsDisplayFrame(df.copy())
    out._display_mode = display_mode
    out._round_digits = digits
    return out


def style_metric_matrix(metric_df: pd.DataFrame, digits: int = 4):
    """
    Stylise une matrice de métriques :
    - lignes = métriques
    - colonnes = pipelines
    - maximum de chaque ligne en vert
    """
    df = metric_df.copy()

    # Conversion prudente : uniquement si possible, sans FutureWarning
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    def highlight_max_per_row(row):
        numeric_row = pd.to_numeric(row, errors="coerce")

        if numeric_row.notna().sum() == 0:
            return [""] * len(row)

        max_val = numeric_row.max()

        return [
            "background-color: #C6EFCE; color: #006100; font-weight: bold;"
            if pd.notna(val) and val == max_val
            else ""
            for val in numeric_row
        ]

    return (
        df.style
        .format(
            lambda x: f"{x:.{digits}f}"
            if isinstance(x, (int, float, np.integer, np.floating))
            else x
        )
        .apply(highlight_max_per_row, axis=1)
        .set_table_styles([
            {"selector": "th", "props": [("white-space", "nowrap")]},
            {"selector": "td", "props": [("white-space", "nowrap")]},
        ])
    )


def seed_everything(seed: int = RANDOM_STATE, use_tensorflow: bool = False) -> None:
    """
    Fixe les graines aléatoires Python / NumPy.
    TensorFlow n'est importé que si use_tensorflow=True.
    Cela évite de casser les notebooks ML quand TensorFlow est
    incompatible avec la version de NumPy installée.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if use_tensorflow:
        try:
            import tensorflow as tf  # type: ignore
            tf.random.set_seed(seed)
        except Exception as exc:
            print(
                "Avertissement : TensorFlow n'a pas pu être initialisé pour fixer la graine. "
                f"Détail : {exc}"
            )


def build_text_series(
    df: pd.DataFrame,
    text_col: str = "text",
    extra_cols: Tuple[str, ...] = ("keyword", "location"),
    use_extra_cols: bool = False,
    lowercase: bool = False,
) -> pd.Series:
    if text_col not in df.columns:
        raise ValueError(f"La colonne '{text_col}' est introuvable dans le fichier fourni.")
    text = df[text_col].fillna("").astype(str)

    if use_extra_cols:
        extras = []
        for col in extra_cols:
            if col in df.columns:
                extras.append(df[col].fillna("").astype(str))
        if extras:
            text = pd.concat(extras + [text], axis=1).agg(" ".join, axis=1)

    text = text.str.replace(r"\s+", " ", regex=True).str.strip()
    if lowercase:
        text = text.str.lower()
    return text


def load_labeled_xy(
    data_path: str,
    text_col: str = "text",
    label_col: str = "target",
    use_extra_cols: bool = False,
    lowercase: bool = False,
    role_name: str = "train",
):
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Le fichier {role_name} est introuvable : {path}. "
            f"Vérifie le chemin relatif depuis le dossier du notebook."
        )
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(
            f"La colonne cible '{label_col}' est absente du fichier {role_name} ({path.name})."
        )
    X = build_text_series(
        df,
        text_col=text_col,
        use_extra_cols=use_extra_cols,
        lowercase=lowercase,
    )
    y = df[label_col].astype(int)
    return df, X, y


def load_train_test_xy(
    train_path: str,
    test_path: str,
    text_col: str = "text",
    label_col: str = "target",
    use_extra_cols: bool = False,
    lowercase: bool = False,
):
    df_train, X_train, y_train = load_labeled_xy(
        train_path,
        text_col=text_col,
        label_col=label_col,
        use_extra_cols=use_extra_cols,
        lowercase=lowercase,
        role_name="train",
    )
    df_test, X_test, y_test = load_labeled_xy(
        test_path,
        text_col=text_col,
        label_col=label_col,
        use_extra_cols=use_extra_cols,
        lowercase=lowercase,
        role_name="test",
    )
    return df_train, X_train, y_train, df_test, X_test, y_test


def stratified_validation_split(
    X,
    y,
    val_size: float = 0.10,
    random_state: int = RANDOM_STATE,
):
    return train_test_split(
        X,
        y,
        test_size=val_size,
        random_state=random_state,
        stratify=y,
    )


def _safe_scores(estimator, X):
    if hasattr(estimator, "predict_proba"):
        proba = estimator.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        return proba.ravel()
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(X)
    return None


def _extract_report_metrics(report: Dict, prefix: str) -> Dict[str, float]:
    out = {
        f"{prefix}_accuracy": report.get("accuracy", np.nan),
        f"{prefix}_precision_macro": report.get("macro avg", {}).get("precision", np.nan),
        f"{prefix}_recall_macro": report.get("macro avg", {}).get("recall", np.nan),
        f"{prefix}_f1_macro": report.get("macro avg", {}).get("f1-score", np.nan),
        f"{prefix}_precision_weighted": report.get("weighted avg", {}).get("precision", np.nan),
        f"{prefix}_recall_weighted": report.get("weighted avg", {}).get("recall", np.nan),
        f"{prefix}_f1_weighted": report.get("weighted avg", {}).get("f1-score", np.nan),
    }
    for cls in ("0", "1"):
        cls_dict = report.get(cls, {})
        out[f"{prefix}_precision_class_{cls}"] = cls_dict.get("precision", np.nan)
        out[f"{prefix}_recall_class_{cls}"] = cls_dict.get("recall", np.nan)
        out[f"{prefix}_f1_class_{cls}"] = cls_dict.get("f1-score", np.nan)
        out[f"{prefix}_support_class_{cls}"] = cls_dict.get("support", np.nan)
    return out


def classification_metrics_from_predictions(
    y_true,
    y_pred,
    y_score=None,
    prefix: str = "test",
) -> Dict[str, float]:
    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )
    metrics = _extract_report_metrics(report, prefix)
    metrics[f"{prefix}_balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)

    if y_score is not None:
        try:
            metrics[f"{prefix}_roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            metrics[f"{prefix}_roc_auc"] = np.nan
        try:
            metrics[f"{prefix}_pr_auc"] = average_precision_score(y_true, y_score)
        except Exception:
            metrics[f"{prefix}_pr_auc"] = np.nan
    else:
        metrics[f"{prefix}_roc_auc"] = np.nan
        metrics[f"{prefix}_pr_auc"] = np.nan

    return metrics


def evaluate_sklearn_pipeline(name, estimator, X_train, X_test, y_train, y_test) -> Dict[str, float]:
    model = clone(estimator)
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_score = _safe_scores(model, X_train)
    test_score = _safe_scores(model, X_test)

    result = {"pipeline": name}
    result.update(classification_metrics_from_predictions(y_train, train_pred, train_score, prefix="train"))
    result.update(classification_metrics_from_predictions(y_test, test_pred, test_score, prefix="test"))
    return result


def metric_matrix_from_results(results_df: pd.DataFrame, index_col: str = "pipeline") -> pd.DataFrame:
    """
    Conserve cette fonction pour ne pas casser les notebooks existants.
    Elle renvoie une matrice transposée :
    - lignes = métriques
    - colonnes = pipelines
    """
    if index_col not in results_df.columns:
        raise ValueError(f"La colonne '{index_col}' est absente du DataFrame de résultats.")

    matrix = pd.DataFrame(results_df).set_index(index_col).T
    return _wrap_results_display(matrix, display_mode="metric_matrix", digits=4)


def round_results(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    """
    Arrondit les colonnes numériques.
    Si la colonne 'pipeline' est présente, l'affichage notebook sera automatiquement
    transposé et stylé sans modifier le contenu interne du DataFrame.
    """
    base_df = pd.DataFrame(df).copy()

    for col in base_df.columns:
        if pd.api.types.is_numeric_dtype(base_df[col]):
            base_df[col] = base_df[col].round(digits)

    display_mode = "results" if "pipeline" in base_df.columns else "metric_matrix"
    return _wrap_results_display(base_df, display_mode=display_mode, digits=digits)


def save_results_bundle(results_df: pd.DataFrame, output_dir: str, stem: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"{stem}_resultats_par_pipeline.csv")
    results_xlsx = os.path.join(output_dir, f"{stem}_resultats_par_pipeline.xlsx")
    metric_matrix_csv = os.path.join(output_dir, f"{stem}_metriques_par_metrique.csv")
    metric_matrix_xlsx = os.path.join(output_dir, f"{stem}_metriques_par_metrique.xlsx")

    round_results(results_df).to_csv(results_path, index=False)
    round_results(results_df).to_excel(results_xlsx, index=False)

    metric_view = round_results(metric_matrix_from_results(results_df))
    metric_view.to_csv(metric_matrix_csv, index=True)
    metric_view.to_excel(metric_matrix_xlsx, index=True)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_#@']+", str(text).lower())


class GensimMeanEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "glove-twitter-200", normalize: bool = False):
        self.model_name = model_name
        self.normalize = normalize
        self.model_ = None
        self.vector_size_ = None

    def fit(self, X, y=None):
        import gensim.downloader as api
        self.model_ = api.load(self.model_name)
        self.vector_size_ = self.model_.vector_size
        return self

    def transform(self, X):
        if self.model_ is None:
            raise RuntimeError("Le vectoriseur n'a pas encore été entraîné.")
        vectors = []
        for text in X:
            tokens = simple_tokenize(text)
            token_vecs = [self.model_[tok] for tok in tokens if tok in self.model_]
            if token_vecs:
                vec = np.mean(token_vecs, axis=0)
            else:
                vec = np.zeros(self.vector_size_, dtype=float)
            if self.normalize:
                denom = np.linalg.norm(vec)
                if denom > 0:
                    vec = vec / denom
            vectors.append(vec)
        return np.vstack(vectors)


class SentenceTransformerVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_ = None

    def fit(self, X, y=None):
        from sentence_transformers import SentenceTransformer
        self.model_ = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if self.model_ is None:
            raise RuntimeError("Le vectoriseur n'a pas encore été entraîné.")
        emb = self.model_.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(emb)


def evaluate_probability_outputs(name, y_train, train_scores, y_test, test_scores, threshold: float = 0.5):
    train_pred = (np.asarray(train_scores).ravel() >= threshold).astype(int)
    test_pred = (np.asarray(test_scores).ravel() >= threshold).astype(int)

    result = {"pipeline": name}
    result.update(classification_metrics_from_predictions(y_train, train_pred, train_scores, prefix="train"))
    result.update(classification_metrics_from_predictions(y_test, test_pred, test_scores, prefix="test"))
    return result
