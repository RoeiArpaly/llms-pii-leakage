import pytest

from evaluation import spans_scorer


@pytest.mark.parametrize(
    "spans_true, spans_pred, expected",
    [
        (
            [{"value": "1111111111111111"}],
            [{"value": "1111111111111111"}],
            {
                "exact_match": True,
                "true_positive": 1,
                "false_positive": 0,
                "false_negative": 0,
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
            },
        ),
        (
            [{"value": "1111111111111111"}],
            [{"value": "2222222222222222"}],
            {
                "exact_match": False,
                "true_positive": 0,
                "false_positive": 1,
                "false_negative": 1,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        ),
        (
            [{"value": "1111111111111111"}, {"value": "2222222222222222"}],
            [{"value": "2222222222222222"}],
            {
                "exact_match": False,
                "true_positive": 1,
                "false_positive": 0,
                "false_negative": 1,
                "precision": 1.0,
                "recall": 0.5,
                "f1": 0.6666666666666666,
            },
        ),
        (
            [{"value": "1111111111111111"}],
            [{"value": "1111111111111111"}, {"value": "2222222222222222"}],
            {
                "exact_match": False,
                "true_positive": 1,
                "false_positive": 1,
                "false_negative": 0,
                "precision": 0.5,
                "recall": 1.0,
                "f1": 0.6666666666666666,
            },
        ),
        (
            [{"value": "1111111111111111"}],
            None,
            {
                "exact_match": False,
                "true_positive": 0,
                "false_positive": 0,
                "false_negative": 1,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        ),
        (
            None,
            [{"value": "1111111111111111"}],
            {
                "exact_match": False,
                "true_positive": 0,
                "false_positive": 1,
                "false_negative": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        ),
        (
            None,
            None,
            None,
        ),
    ],
)
def test_spans_scorer(spans_true, spans_pred, expected):
    result = spans_scorer(spans_true, spans_pred)
    assert result == expected
