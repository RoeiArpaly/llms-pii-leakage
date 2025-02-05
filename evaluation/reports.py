import pandas as pd

from evaluation.spans import safe_divide
from utils import infer_json


def calculate_scores(file_path, columns):
    """
    Reads a CSV file, applies JSON inference, and calculates mean F1, Recall, and Precision scores.
    """
    df = pd.read_csv(file_path).apply(infer_json)
    results = []
    for col, col_name in columns:
        true_positives = df[col].apply(lambda x: x.get("true_positive")).sum()
        false_positives = df[col].apply(lambda x: x.get("false_positive")).sum()
        false_negatives = df[col].apply(lambda x: x.get("false_negative")).sum()
        precision = safe_divide(true_positives, true_positives + false_positives)
        recall = safe_divide(true_positives, true_positives + false_negatives)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        results.append(
            {
                "Model": col_name,
                "F1": f1,
                "Recall": recall,
                "Precision": precision,
            },
        )
    return results


def evaluation_datasets(datasets):
    results = []
    for dataset, (file_path, columns) in datasets.items():
        scores = calculate_scores(file_path, columns)
        for score in scores:
            score["Dataset"] = dataset
            results.append(score)
    return pd.DataFrame(results).set_index(keys=["Dataset", "Model"])


df_results = evaluation_datasets(
    datasets={
        "Baseline": (
            "../datasets/llm_detection_results_02.csv",
            [
                ("spans_score_analyzer", "Presidio"),
                ("spans_score_llm", "gpt-4o-mini"),
            ],
        ),
        "Fuzzy PII": (
            "../datasets/fuzzy_pii_generation_results_03.csv",
            [
                ("spans_score_analyzer", "Presidio"),
                ("spans_score_llm_restored_analyzer", "gpt-4o-mini"),
            ],
        ),
        "Fuzzy PII + Adv Content": (
            "../datasets/fuzzy_pii_adv_content_generation_results_04.csv",
            [
                ("spans_score_analyzer", "Presidio"),
                ("spans_score_llm_restored_analyzer", "gpt-4o-mini"),
            ]
        ),
    },
)

df_results.to_csv("../datasets/score_results.csv")
