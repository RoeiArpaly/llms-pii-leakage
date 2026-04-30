"""Evaluation orchestration: loads datasets and predictions, applies span-level
scoring across all models, and saves per-row evaluation results to CSV.
"""
from pathlib import Path

from pandas import read_csv

from config import Config
from evaluation import spans_scorer
from utils import (
    cast_to_json,
    infer_json,
)


DATASET_PATH = Path("datasets/dataset.csv")
PREDICTIONS_PATH = Path("datasets/predictions.csv")
EVALUATIONS_PATH = Path("datasets/evaluations.csv")

SPANS_METRICS = ["true_positive", "false_positive", "false_negative"]


def evaluate_predictions(models: list[str], match_level: str, method: str):
    dataset = read_csv(DATASET_PATH).apply(infer_json)
    predictions = read_csv(PREDICTIONS_PATH).apply(infer_json)

    predictions = predictions[predictions["model"].isin(models)]

    merged = predictions.merge(
        dataset[["uid", "pii_spans", "category", "attack_target"]],
        on="uid",
        how="left",
    )

    # Filter by configured attacks
    pii_allowed = set(Config.ATTACKS)
    content_allowed = set(Config.CONTENT_ATTACKS)
    mask = [True] * len(merged)
    for i, row in merged.iterrows():
        target = row.get("attack_target")
        if isinstance(target, dict):
            if not all(v in pii_allowed for v in target.get("pii", [])):
                mask[i] = False
            if not all(v in content_allowed for v in target.get("context", [])):
                mask[i] = False
    merged = merged[mask]

    merged["spans_score"] = merged.apply(
        lambda row: spans_scorer(
            spans_true=row["pii_spans"],
            spans_pred=row["prediction"],
            match_level=match_level,
            method=method,
        ),
        axis=1,
    )

    eval_df = merged[["uid", "model", "prediction", "spans_score"]]
    eval_df.apply(cast_to_json).to_csv(EVALUATIONS_PATH, index=False)
    return eval_df


def evaluate_and_save_datasets(models: list[str], match_level: str, method: str) -> None:
    evaluate_predictions(models=models, match_level=match_level, method=method)
