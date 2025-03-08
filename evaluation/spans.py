import re


def safe_divide(a, b):
    return a / b if b > 0 else 0


def normalize_pii(value: str) -> str:
    """Normalizes PII by removing non-alphanumeric characters and converting to lowercase."""
    return re.sub(pattern=r"\W", repl="", string=value).lower()


def spans_set(span_lists: list[list[dict]]) -> list[dict]:
    """Converts a list of spans to a set of spans."""
    spans = [item for sublist in span_lists for item in sublist]
    frozen_spans = set([frozenset(span.items()) for span in spans])
    return [dict(frozen_span) for frozen_span in frozen_spans]


def spans_scorer(spans_true, spans_pred, match_level, reverse_match=True):
    """

    Parameters
    ----------
    spans_true : list
        The true spans.
    spans_pred : list
        The predicted spans.
    match_level : str
        One of "value", "type", or "both".
    reverse_match : bool
        Whether to check for reverse match.

    Examples
    --------
    spans = [{"value": "4316476143666553", "start": 84, "end": 100, "type": "credit_card_number"}]

    Returns
    -------

    """
    if not spans_true and not spans_pred:
        return {}
    if not isinstance(spans_true, list):
        spans_true = []
    if not isinstance(spans_pred, list):
        spans_pred = []

    true_positives = 0
    matched_true = set()  # Tracks matched true spans
    matched_pred = set()  # Tracks matched predicted spans
    for i, true_span in enumerate(spans_true):
        true_value = normalize_pii(true_span["value"])
        true_type = true_span["type"]
        for j, pred_span in enumerate(spans_pred):
            pred_value = normalize_pii(pred_span["value"])
            pred_type = pred_span["type"]
            match_conditions = [
                match_level == "both" and true_value == pred_value and true_type == pred_type,
                match_level == "value" and true_value == pred_value,
                match_level == "type" and true_type == pred_type,
            ]
            if reverse_match:
                if match_level == "both":
                    match_conditions.append(
                        true_value == pred_value[::-1] and true_type == pred_type
                    )
                else:
                    match_conditions.append(true_value == pred_value[::-1])
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
