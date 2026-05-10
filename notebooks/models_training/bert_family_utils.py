from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset
from nlp_disaster_utils import classification_metrics_from_predictions
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def resolve_train_test_paths() -> Tuple[str, str]:
    train_candidates = [
        Path("../../data/processed_data/train.csv"),
        Path("../../data/train.csv"),
    ]
    test_candidates = [
        Path("../../data/processed_data/test.csv"),
        Path("../../data/test.csv"),
    ]

    train_path = next((p for p in train_candidates if p.exists()), None)
    test_path = next((p for p in test_candidates if p.exists()), None)

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"Impossible de trouver train/test. "
            f"Train testés : {[str(p) for p in train_candidates]} | "
            f"Test testés : {[str(p) for p in test_candidates]}"
        )

    return str(train_path), str(test_path)


def make_hf_dataset(tokenizer, X, y, max_len: int):
    ds = Dataset.from_dict({"text": list(X), "label": list(map(int, y))})

    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    ds = ds.map(tokenize_batch, batched=True)
    return ds


def compute_metrics_binary(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, pos_label=1, zero_division=0),
        "recall": recall_score(labels, preds, pos_label=1, zero_division=0),
        "f1": f1_score(labels, preds, pos_label=1, zero_division=0),
    }


def softmax_rows(logits):
    logits = np.asarray(logits)
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def evaluate_trainer_on_dataset(trainer, dataset, y_true, split_name: str):
    pred_output = trainer.predict(dataset)
    logits = pred_output.predictions
    probs = softmax_rows(logits)[:, 1]
    preds = np.argmax(logits, axis=1)

    metrics = {"pipeline": split_name}
    metrics.update(
        classification_metrics_from_predictions(
            y_true=np.asarray(y_true),
            y_pred=preds,
            y_score=probs,
            prefix=split_name,
        )
    )
    return metrics


def trainer_history_to_dataframe(trainer) -> pd.DataFrame:
    history = getattr(getattr(trainer, "state", None), "log_history", None)
    if history is None:
        return pd.DataFrame()
    return pd.DataFrame(history)


def make_jsonable_config(cfg: Dict) -> str:
    return json.dumps(cfg, ensure_ascii=False, indent=2)
