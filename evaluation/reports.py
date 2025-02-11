from typing import List

from pandas import (
    concat,
    DataFrame,
    read_csv,
)

from evaluation.constants import SPANS_METRICS
from utils import infer_json


def load_and_preprocess_data(dataset: str, model: str) -> DataFrame:
    """Load, merge, and process prediction and dataset CSVs."""
    data = read_csv(f"../datasets/{dataset}_dataset.csv").apply(infer_json)
    data_pred = read_csv(f"../datasets/{dataset}_{model}_prediction.csv").apply(infer_json)
    data = data_pred.merge(data, on="uid", how="left")
    for col in ["fuzzy_techniques", "adv_content_techniques"]:
        data[col] = data[col].apply(lambda x: x[0] if x else None) if col in data.columns else None
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


def evaluate_and_save_datasets(datasets: List[str], models: List[str]) -> None:
    """Evaluate and save aggregated scores for all dataset-model pairs."""
    groupings = {
        "dataset_level": [],
        "fuzzy": ["fuzzy_techniques"],
        "adv": ["adv_content_techniques"],
        "both": ["fuzzy_techniques", "adv_content_techniques"],
    }

    results = {key: [] for key in groupings}
    for dataset in datasets:
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
            file_name = f"../datasets/score_results_00000{i}.csv"
            concat(results[key], ignore_index=True).set_index(idx).to_csv(file_name)


evaluate_and_save_datasets(
    datasets=["baseline", "fuzzy", "fuzzy_adv"],
    models=["Presidio", "gpt-4o-mini"],
)
