import pandas as pd

from evaluation.spans import safe_divide
from utils import infer_json


def calculate_scores(file_path: str):
    """
    Reads a CSV file, applies JSON inference, and calculates mean F1, Recall, and Precision scores.
    """
    data = pd.read_csv(file_path).apply(infer_json)
    true_positives = data["spans_score"].apply(lambda x: x.get("true_positive")).sum()
    false_positives = data["spans_score"].apply(lambda x: x.get("false_positive")).sum()
    false_negatives = data["spans_score"].apply(lambda x: x.get("false_negative")).sum()
    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "F1": f1,
        "Recall": recall,
        "Precision": precision,
    }


def evaluation_datasets():
    results = []
    for dataset in ["baseline", "fuzzy", "fuzzy_adv"]:
        for model in ["Presidio", "gpt-4o-mini"]:
            file_path = f"../datasets/{dataset}_{model}_prediction.csv"
            scores = calculate_scores(file_path=file_path)
            scores["Dataset"] = dataset
            scores["Model"] = model
            results.append(scores)
    return pd.DataFrame(results).set_index(keys=["Dataset", "Model"])


df_results = evaluation_datasets()
df_results.to_csv("../datasets/score_results_001.csv")
