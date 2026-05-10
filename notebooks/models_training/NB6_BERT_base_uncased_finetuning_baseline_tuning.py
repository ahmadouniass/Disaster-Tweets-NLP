# %pip install -q pandas numpy scikit-learn torch transformers datasets accelerate openpyxl mlflow

import json
import os
import warnings
from contextlib import nullcontext
from pathlib import Path

import mlflow
import pandas as pd


os.environ["TOKENIZERS_PARALLELISM"] = "false"


from bert_family_utils import (
    compute_metrics_binary,
    evaluate_trainer_on_dataset,
    make_hf_dataset,
    make_jsonable_config,
    trainer_history_to_dataframe,
)
from nlp_disaster_utils import (
    load_train_test_xy,
    round_results,
    seed_everything,
    stratified_validation_split,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)


seed_everything(42)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 200)
warnings.filterwarnings("ignore")
print("MLflow :", mlflow.__version__)


MODEL_NAME = "distilbert-base-uncased"
PIPELINE_NAME = "BERT_base_uncased"

TEXT_COL = "text"
LABEL_COL = "target"
USE_AUX_TEXT_COLUMNS = False
LOWERCASE_TEXT = False

VAL_SIZE_WITHIN_TRAIN = 0.10
RANDOM_STATE = 42

BASELINE_CONFIG = {
  "learning_rate": 2e-05,
  "batch_size": 8,
  "num_epochs": 3,
  "weight_decay": 0.01,
  "max_len": 96
}

TUNING_CANDIDATES = [
  {
    "learning_rate": 2e-05,
    "batch_size": 8,
    "num_epochs": 2,
    "weight_decay": 0.01,
    "max_len": 96
  },
  {
    "learning_rate": 2e-05,
    "batch_size": 8,
    "num_epochs": 3,
    "weight_decay": 0.01,
    "max_len": 96
  },
  {
    "learning_rate": 3e-05,
    "batch_size": 8,
    "num_epochs": 2,
    "weight_decay": 0.01,
    "max_len": 96
  },
  {
    "learning_rate": 2e-05,
    "batch_size": 16,
    "num_epochs": 2,
    "weight_decay": 0.01,
    "max_len": 96
  }
]

OUTPUT_DIR = Path("../../outputs/BERT_bert_base_uncased")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_ARTIFACTS_DIR = OUTPUT_DIR / "baseline_artifacts"
TUNING_ARTIFACTS_DIR = OUTPUT_DIR / "tuning_artifacts"
BASELINE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
TUNING_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_STEM = "BERT_bert_base_uncased"

USE_MLFLOW = True
MLFLOW_EXPERIMENT_NAME = "DT_BERT_BERTBaseUncased"
MLFLOW_TRACKING_URI = Path("../../outputs/mlruns").resolve().as_uri()
MLFLOW_LOG_MODEL = False



if USE_MLFLOW:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print("MLflow tracking URI :", MLFLOW_TRACKING_URI)
    print("MLflow experiment   :", MLFLOW_EXPERIMENT_NAME)
else:
    print("MLflow désactivé.")



TRAIN_PATH, TEST_PATH = "C:/Users/DELL/Desktop/TP Ml/Disaster-Tweets-NLP/data/processed_data/train_cleaned.csv", "C:/Users/DELL/Desktop/TP Ml/Disaster-Tweets-NLP/data/processed_data/test_cleaned.csv"

df_train_full, X_train_full, y_train_full, df_test, X_test, y_test = load_train_test_xy(
    train_path=TRAIN_PATH,
    test_path=TEST_PATH,
    text_col=TEXT_COL,
    label_col=LABEL_COL,
    use_extra_cols=USE_AUX_TEXT_COLUMNS,
    lowercase=LOWERCASE_TEXT,
)

X_train, X_val, y_train, y_val = stratified_validation_split(
    X_train_full,
    y_train_full,
    val_size=VAL_SIZE_WITHIN_TRAIN,
    random_state=RANDOM_STATE,
)

print("Train complet :", len(X_train_full))
print("Train tuning  :", len(X_train))
print("Validation    :", len(X_val))
print("Test          :", len(X_test))


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=False
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_ds = make_hf_dataset(tokenizer, X_train, y_train, max_len=BASELINE_CONFIG["max_len"])
val_ds = make_hf_dataset(tokenizer, X_val, y_val, max_len=BASELINE_CONFIG["max_len"])
test_ds = make_hf_dataset(tokenizer, X_test, y_test, max_len=BASELINE_CONFIG["max_len"])
train_full_ds = make_hf_dataset(tokenizer, X_train_full, y_train_full, max_len=BASELINE_CONFIG["max_len"])

def make_training_args(
    output_dir: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    weight_decay: float,
    do_eval: bool = True,
    load_best_model_at_end: bool = True,
):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch" if do_eval else "no",
        save_strategy="epoch" if do_eval else "no",
        logging_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=load_best_model_at_end if do_eval else False,
        metric_for_best_model="f1" if do_eval else None,
        greater_is_better=True if do_eval else None,
        report_to="none",
        save_total_limit=1 if do_eval else None,
        seed=42,
    )


baseline_run_ctx = mlflow.start_run(run_name=f"BASE_{PIPELINE_NAME}") if USE_MLFLOW else nullcontext()

with baseline_run_ctx:
    baseline_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    baseline_args = make_training_args(
        output_dir=str(BASELINE_ARTIFACTS_DIR / "trainer_outputs"),
        learning_rate=BASELINE_CONFIG["learning_rate"],
        batch_size=BASELINE_CONFIG["batch_size"],
        num_epochs=BASELINE_CONFIG["num_epochs"],
        weight_decay=BASELINE_CONFIG["weight_decay"],
        do_eval=True,
        load_best_model_at_end=True,
    )

    baseline_trainer = Trainer(
        model=baseline_model,
        args=baseline_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_binary,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    baseline_trainer.train()

    baseline_history_df = trainer_history_to_dataframe(baseline_trainer)
    if not baseline_history_df.empty:
        baseline_history_df.to_csv(BASELINE_ARTIFACTS_DIR / f"{PIPELINE_NAME}_baseline_history.csv", index=False)

    baseline_train_metrics = evaluate_trainer_on_dataset(baseline_trainer, train_ds, y_train, "train")
    baseline_test_metrics = evaluate_trainer_on_dataset(baseline_trainer, test_ds, y_test, "test")

    baseline_result = {"pipeline": PIPELINE_NAME}
    baseline_result.update(baseline_train_metrics)
    baseline_result.update(baseline_test_metrics)
    baseline_result["config"] = make_jsonable_config(BASELINE_CONFIG)

    if USE_MLFLOW:
        mlflow.set_tag("notebook", PIPELINE_NAME)
        mlflow.set_tag("family", "bert_like_models")
        mlflow.set_tag("phase", "baseline")
        mlflow.log_param("model_name", MODEL_NAME)
        for k, v in BASELINE_CONFIG.items():
            mlflow.log_param(f"baseline__{k}", v)
        for col, val in baseline_result.items():
            if col not in {"pipeline", "config"} and pd.notna(val):
                try:
                    mlflow.log_metric(col, float(val))
                except Exception:
                    pass
        if (BASELINE_ARTIFACTS_DIR / f"{PIPELINE_NAME}_baseline_history.csv").exists():
            mlflow.log_artifact(str(BASELINE_ARTIFACTS_DIR / f"{PIPELINE_NAME}_baseline_history.csv"), artifact_path="baseline_history")

baseline_df = round_results(pd.DataFrame([baseline_result]))


tuning_rows = []

for idx, cfg in enumerate(TUNING_CANDIDATES, start=1):
    print("=" * 100)
    print(f"Essai tuning {idx}/{len(TUNING_CANDIDATES)}")
    print(cfg)

    run_ctx = mlflow.start_run(run_name=f"TUNE_TRIAL_{idx}_{PIPELINE_NAME}") if USE_MLFLOW else nullcontext()

    with run_ctx:
        trial_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        trial_train_ds = make_hf_dataset(tokenizer, X_train, y_train, max_len=cfg["max_len"])
        trial_val_ds = make_hf_dataset(tokenizer, X_val, y_val, max_len=cfg["max_len"])

        trial_args = make_training_args(
            output_dir=str(TUNING_ARTIFACTS_DIR / f"trial_{idx}_trainer_outputs"),
            learning_rate=cfg["learning_rate"],
            batch_size=cfg["batch_size"],
            num_epochs=cfg["num_epochs"],
            weight_decay=cfg["weight_decay"],
            do_eval=True,
            load_best_model_at_end=True,
        )

        trial_trainer = Trainer(
            model=trial_model,
            args=trial_args,
            train_dataset=trial_train_ds,
            eval_dataset=trial_val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_binary,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        )

        trial_trainer.train()

        val_metrics = evaluate_trainer_on_dataset(trial_trainer, trial_val_ds, y_val, "val")
        row = {
            "trial_id": idx,
            "config": make_jsonable_config(cfg),
            "learning_rate": cfg["learning_rate"],
            "batch_size": cfg["batch_size"],
            "num_epochs": cfg["num_epochs"],
            "weight_decay": cfg["weight_decay"],
            "max_len": cfg["max_len"],
        }
        row.update(val_metrics)
        tuning_rows.append(row)

        history_df = trainer_history_to_dataframe(trial_trainer)
        history_path = TUNING_ARTIFACTS_DIR / f"{PIPELINE_NAME}_trial_{idx}_history.csv"
        if not history_df.empty:
            history_df.to_csv(history_path, index=False)

        if USE_MLFLOW:
            mlflow.set_tag("notebook", PIPELINE_NAME)
            mlflow.set_tag("family", "bert_like_models")
            mlflow.set_tag("phase", "tuning_trial")
            mlflow.log_param("model_name", MODEL_NAME)
            for k, v in cfg.items():
                mlflow.log_param(f"trial__{k}", v)
            for col, val in row.items():
                if col not in {"config"} and pd.notna(val):
                    try:
                        mlflow.log_metric(col, float(val))
                    except Exception:
                        pass
            if history_path.exists():
                mlflow.log_artifact(str(history_path), artifact_path="tuning_history")


tuning_df = pd.DataFrame(tuning_rows).sort_values(
    by=["val_f1_class_1", "val_recall_class_1", "val_f1_macro", "val_balanced_accuracy", "val_roc_auc"],
    ascending=False,
).reset_index(drop=True)

best_tuning_row = tuning_df.iloc[0].copy()
best_config = json.loads(best_tuning_row["config"])

print("Meilleure configuration retenue :")
print(json.dumps(best_config, ensure_ascii=False, indent=2))


final_run_ctx = mlflow.start_run(run_name=f"TUNED_FINAL_{PIPELINE_NAME}") if USE_MLFLOW else nullcontext()

with final_run_ctx:
    final_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    final_train_full_ds = make_hf_dataset(tokenizer, X_train_full, y_train_full, max_len=best_config["max_len"])
    final_test_ds = make_hf_dataset(tokenizer, X_test, y_test, max_len=best_config["max_len"])

    final_args = make_training_args(
        output_dir=str(TUNING_ARTIFACTS_DIR / "best_final_trainer_outputs"),
        learning_rate=best_config["learning_rate"],
        batch_size=best_config["batch_size"],
        num_epochs=best_config["num_epochs"],
        weight_decay=best_config["weight_decay"],
        do_eval=False,
        load_best_model_at_end=False,
    )

    final_trainer = Trainer(
        model=final_model,
        args=final_args,
        train_dataset=final_train_full_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_binary,
    )

    final_trainer.train()

    tuned_train_metrics = evaluate_trainer_on_dataset(final_trainer, final_train_full_ds, y_train_full, "train")
    tuned_test_metrics = evaluate_trainer_on_dataset(final_trainer, final_test_ds, y_test, "test")

    tuned_result = {"pipeline": PIPELINE_NAME}
    tuned_result.update(tuned_train_metrics)
    tuned_result.update(tuned_test_metrics)
    tuned_result["best_config"] = make_jsonable_config(best_config)
    tuned_result["best_val_f1_class_1"] = float(best_tuning_row["val_f1_class_1"])
    tuned_result["best_val_recall_class_1"] = float(best_tuning_row["val_recall_class_1"])
    tuned_result["best_val_f1_macro"] = float(best_tuning_row["val_f1_macro"])
    tuned_result["best_val_balanced_accuracy"] = float(best_tuning_row["val_balanced_accuracy"])
    tuned_result["best_val_roc_auc"] = float(best_tuning_row["val_roc_auc"])

    if USE_MLFLOW:
        mlflow.set_tag("notebook", PIPELINE_NAME)
        mlflow.set_tag("family", "bert_like_models")
        mlflow.set_tag("phase", "tuned_final")
        mlflow.log_param("model_name", MODEL_NAME)
        for k, v in best_config.items():
            mlflow.log_param(f"best__{k}", v)
        for col, val in tuned_result.items():
            if col not in {"pipeline", "best_config"} and pd.notna(val):
                try:
                    mlflow.log_metric(col, float(val))
                except Exception:
                    pass


tuned_df = round_results(pd.DataFrame([tuned_result]))


comparison_df = pd.DataFrame([
    {
        "pipeline": PIPELINE_NAME,
        "baseline_test_f1_class_1": baseline_result["test_f1_class_1"],
        "tuned_test_f1_class_1": tuned_result["test_f1_class_1"],
        "delta_test_f1_class_1": tuned_result["test_f1_class_1"] - baseline_result["test_f1_class_1"],
        "baseline_test_recall_class_1": baseline_result["test_recall_class_1"],
        "tuned_test_recall_class_1": tuned_result["test_recall_class_1"],
        "delta_test_recall_class_1": tuned_result["test_recall_class_1"] - baseline_result["test_recall_class_1"],
        "baseline_test_f1_macro": baseline_result["test_f1_macro"],
        "tuned_test_f1_macro": tuned_result["test_f1_macro"],
        "delta_test_f1_macro": tuned_result["test_f1_macro"] - baseline_result["test_f1_macro"],
        "baseline_test_balanced_accuracy": baseline_result["test_balanced_accuracy"],
        "tuned_test_balanced_accuracy": tuned_result["test_balanced_accuracy"],
        "delta_test_balanced_accuracy": tuned_result["test_balanced_accuracy"] - baseline_result["test_balanced_accuracy"],
    }
])




baseline_export_path = OUTPUT_DIR / f"{OUTPUT_STEM}_baseline_results.csv"
tuning_export_path = OUTPUT_DIR / f"{OUTPUT_STEM}_tuning_validation_results.csv"
tuned_export_path = OUTPUT_DIR / f"{OUTPUT_STEM}_tuned_results.csv"
comparison_export_path = OUTPUT_DIR / f"{OUTPUT_STEM}_baseline_vs_tuned.csv"

pd.DataFrame([baseline_result]).to_csv(baseline_export_path, index=False)
pd.DataFrame(tuning_rows).to_csv(tuning_export_path, index=False)
pd.DataFrame([tuned_result]).to_csv(tuned_export_path, index=False)
comparison_df.to_csv(comparison_export_path, index=False)

print("Export baseline  :", baseline_export_path)
print("Export tuning    :", tuning_export_path)
print("Export tuned     :", tuned_export_path)
print("Export comparaison :", comparison_export_path)
