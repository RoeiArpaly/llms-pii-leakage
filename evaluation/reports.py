from typing import List

from pandas import (
    concat,
    DataFrame,
    read_csv,
)

from config import Config
from constants import DATASETS
from evaluation import spans_scorer
from evaluation.constants import SPANS_METRICS
from utils import (
    cast_to_json,
    infer_json,
)


def evaluate_predictions(models, match_level: str):
    datasets = []
    for dataset in DATASETS:
        for model in models:
            data = read_csv(f"datasets/predictions/{dataset}_{model}.csv").apply(infer_json)
            if "fuzzy_techniques" in data.columns:
                data = data[data["fuzzy_techniques"].apply(
                    lambda x: all([v in Config.ATTACKS for v in x]))
                ]
            if "adv_content_techniques" in data.columns:
                data = data[data["adv_content_techniques"].apply(
                    lambda x: all([v in Config.CONTENT_ATTACKS for v in x]))
                ]

            data["spans_score"] = data.apply(
                lambda row: spans_scorer(
                    spans_true=row["pii_spans"],
                    spans_pred=row["prediction"],
                    match_level=match_level,
                ),
                axis=1,
            )
            _data = data[["uid", "prediction", "spans_score"]].apply(cast_to_json)
            _data.to_csv(f"datasets/evaluations/{dataset}_{model}.csv", index=False)
            data["model"] = model
            datasets.append(data)
    columns = ["uid", "model", "fuzzy_techniques", "adv_content_techniques", "spans_score"]
    raw = concat(datasets, ignore_index=True)[columns].apply(cast_to_json)
    raw.to_csv("datasets/evaluations/0_raw.csv", index=False)


def load_and_preprocess_data(dataset: str, model: str) -> DataFrame:
    """Load, merge, and process prediction and dataset CSVs."""
    data = read_csv(f"datasets/{dataset}_dataset.csv").apply(infer_json)
    data_pred = read_csv(f"datasets/evaluations/{dataset}_{model}.csv").apply(infer_json)
    data = data_pred.merge(data, on="uid", how="left")
    for col in ["fuzzy_techniques", "adv_content_techniques"]:
        data[col] = data[col].apply(lambda x: "_".join(x)) if col in data.columns else None
    for col in SPANS_METRICS:
        data[col] = data["spans_score"].apply(lambda x: x.get(col))
    return data


def compute_aggregated_scores(data: DataFrame, groupby_cols: List[str] = None) -> DataFrame:
    """Aggregate raw counts (overall or by group) and compute evaluation metrics."""
    if groupby_cols:
        data = data.groupby(groupby_cols)[SPANS_METRICS].sum().reset_index()
    else:
        data = DataFrame([data[SPANS_METRICS].sum()])
    data["Precision"] = data["true_positive"] / (data["true_positive"] + data["false_positive"])
    data["Recall"] = data["true_positive"] / (data["true_positive"] + data["false_negative"])
    data["F1"] = 2 * data["Precision"] * data["Recall"] / (data["Precision"] + data["Recall"])
    return data.drop(columns=SPANS_METRICS).fillna(0)


def evaluate_and_save_datasets(models: List[str], match_level: str) -> None:
    """Evaluate and save aggregated scores for all dataset-model pairs."""

    models = models + ["ensemble"]
    evaluate_predictions(models=models, match_level=match_level)

    groupings = {
        "dataset_level": [],
        "fuzzy": ["fuzzy_techniques"],
        "adv": ["adv_content_techniques"],
        "both": ["fuzzy_techniques", "adv_content_techniques"],
    }

    results = {key: [] for key in groupings}
    for dataset in DATASETS:
        for model in models:
            data = load_and_preprocess_data(dataset=dataset, model=model)
            for key, cols in groupings.items():
                agg = compute_aggregated_scores(data=data, groupby_cols=cols)
                if not agg.empty:
                    agg["Dataset"] = dataset
                    agg["Model"] = model
                    results[key].append(agg)

    for i, key in enumerate(groupings, 1):
        if results[key]:
            idx = ["Dataset", "Model"] + groupings[key]
            file_name = f"datasets/evaluations/{i}_{key}.csv"
            concat(results[key], ignore_index=True).set_index(idx).to_csv(file_name)
