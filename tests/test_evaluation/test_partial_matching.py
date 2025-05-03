import pytest

from evaluation.partial_matching import (
    is_subsequence,
    partial_match,
)


@pytest.mark.parametrize(
    "query, string, expected",
    [
        ("abcdefg", "abc", True),
        ("abcdefg", "ace", True),
        ("abcdefg", "aec", False),
        ("12-3", "123", True),
        ("1-2-3", "132", False),
        ("hello world", "hlo", True),
        ("hello", "hello", True),
        ("hello", "helloo", False),
        ("", "", True),
        ("", "a", False),
        ("abc", "", True),
        ("abc", "abc", True),
        ("abc", "abcd", False),
    ]
)
def test_is_subsequence(query, string, expected):
    assert is_subsequence(query, string) == expected


@pytest.mark.parametrize(
    "predicted_span, actual_span, method, expected",
    [
        (
                {"value": "123"},
                {"value": "123"},
                "exact",
                1.0,
        ),
        (
                {"value": "12-3"},
                {"value": "123"},
                "exact",
                0.0,
        ),
        (
                {"value": "1-2-3"},
                {"value": "123"},
                "subsequence",
                1.0,
        ),
        (
                {"value": "12-4"},
                {"value": "123"},
                "subsequence",
                0.0,
        ),
        (
                {"value": "123-45-6789"},
                {"value": "123456789"},
                "difflib",
                0.9,
        ),
        (
                {"value": "123-45-6789"},
                {"value": "123456789"},
                "rapidfuzz",
                0.78,
        ),
        (
                {"value": "123-45-6789"},
                {"value": "123456789"},
                "llm_judge",
                0.95,
        ),
    ],
)
def test_partial_match(mocker, predicted_span, actual_span, method, expected):
    """
    Test the partial_match function with different methods.
    """
    if method == "llm_judge":
        mock_post = mocker.patch("evaluation.partial_matching.post_request_openai")
        mock_post.return_value = {"score": expected}

    result = partial_match(predicted_span, actual_span, method=method)
    assert result == expected
