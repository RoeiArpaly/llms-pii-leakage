def safe_divide(a, b):
    return a / b if b > 0 else 0


def spans_scorer(spans_true, spans_pred):
    """

    Parameters
    ----------
    spans_true : list
        The true spans.
    spans_pred : list
        The predicted spans.

    Examples
    --------
    spans = [{"value": "4316476143666553", "start": 84, "end": 100, "type": "credit_card_number"}]

    Returns
    -------

    """

    if not spans_true and not spans_pred:
        return

    if not spans_true:
        spans_true = []
    if not spans_pred:
        spans_pred = []

    true_values = set([span["value"] for span in spans_true])
    pred_values = set([span["value"] for span in spans_pred])
    exact_match = true_values == pred_values

    true_positive = len(true_values.intersection(pred_values))
    false_positive = len(pred_values - true_values)
    false_negative = len(true_values - pred_values)

    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "exact_match": exact_match,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
