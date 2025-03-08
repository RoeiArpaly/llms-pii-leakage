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


def spans_scorer(spans_true, spans_pred, reverse_match=True):
    """

    Parameters
    ----------
    spans_true : list
        The true spans.
    spans_pred : list
        The predicted spans.
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
    for span in spans_true:
        true_value = normalize_pii(span["value"])
        for pred_span in spans_pred:
            pred_value = normalize_pii(pred_span["value"])
            if true_value == pred_value:
                true_positives += 1
                break
            if reverse_match and true_value == pred_value[::-1]:
                true_positives += 1
                break

    false_negatives = len(spans_true) - true_positives
    false_positives = len(spans_pred) - true_positives
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
