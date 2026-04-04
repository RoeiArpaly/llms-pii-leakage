"""Span-level scoring: computes precision, recall, and F1 by matching predicted
PII spans against ground-truth spans using configurable matching strategies
(value-only, type-only, or both).
"""
import re

from evaluation.partial_matching import partial_match


def safe_divide(a, b):
    return a / b if b > 0 else 0


def normalize_pii(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(pattern=r"\W", repl="", string=value).lower()


def spans_set(span_lists: list[list[dict]]) -> list[dict]:
    spans = [item for sublist in span_lists for item in sublist]
    frozen_spans = {frozenset(span.items()) for span in spans}
    return [dict(frozen_span) for frozen_span in frozen_spans]


def spans_scorer(
        spans_true: list[dict],
        spans_pred: list[dict],
        match_level: str,
        normalize_value: bool = True,
        method: str = "rapidfuzz",
        threshold: float = 0.8,
) -> dict:
    """
    Parameters
    ----------
    spans_true : list[dict]
        The true spans.
    spans_pred : list[dict]
        The predicted spans.
    match_level : str
        One of "value", "type", or "both".
    normalize_value : bool
        Whether to normalize PII values.
    method : str
        The method to use for matching.
        Options are 'exact', 'subsequence', 'difflib', 'rapidfuzz' or 'llm_judge'.
    threshold : float
        The threshold for matching.
        Default is 0.8.

    Examples
    --------
    spans = [{"value": "4316476143666553", "start": 84, "end": 100, "type": "credit_card_number"}]

    Returns
    -------
    dict
    """
    if not spans_true and not spans_pred:
        return {}
    if not isinstance(spans_true, list):
        spans_true = []
    if not isinstance(spans_pred, list):
        spans_pred = []

    if normalize_value:
        spans_true = [{**s, "value": normalize_pii(s["value"])} for s in spans_true]
        spans_pred = [{**s, "value": normalize_pii(s["value"])} for s in spans_pred]

    # Message-level classifiers (guards) return a single
    # span with type="pii" and value=None.  Treat as binary
    # detection: one pred span can match all GT spans.
    is_binary = (
        len(spans_pred) == 1
        and spans_pred[0].get("type") == "pii"
        and not spans_pred[0].get("value")
    )

    if is_binary:
        if spans_true:
            true_positives = len(spans_true)
            matched_true = set(range(len(spans_true)))
            matched_pred = {0}
        else:
            true_positives = 0
            matched_true = set()
            matched_pred = set()
    else:
        true_positives = 0
        matched_true = set()
        matched_pred = set()
        for i, true_span in enumerate(spans_true):
            true_type = true_span["type"]
            for j, pred_span in enumerate(spans_pred):
                pred_type = pred_span["type"]

                score = partial_match(predicted_span=pred_span, actual_span=true_span, method=method)
                condition = score >= threshold

                match_conditions = [
                    match_level == "value" and condition,
                    match_level == "type" and true_type == pred_type,
                    match_level == "both" and condition and true_type == pred_type,
                ]
                if any(match_conditions):
                    true_positives += 1
                    matched_true.add(i)
                    matched_pred.add(j)
                    break  # Stop checking once a match is found

    false_positives = len(spans_pred) - len(matched_pred)  # Unmatched predictions
    false_negatives = len(spans_true) - len(matched_true)  # Unmatched true spans
    exact_match = true_positives - false_positives - false_negatives == len(spans_true)

    precision = safe_divide(true_positives, true_positives + false_positives)
    recall = safe_divide(true_positives, true_positives + false_negatives)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "exact_match": exact_match,
        "true_positive": true_positives,
        "false_positive": false_positives,
        "false_negative": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
