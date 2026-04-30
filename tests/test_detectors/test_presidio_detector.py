import pytest

from detectors.presidio import (
    filter_results,
    get_presidio_model,
    presidio_pii_analyzer,
)


class TestGetPresidioModel:

    def test_returns_analyzer_engine(self):
        model = get_presidio_model()
        assert model is not None

    def test_caching(self):
        assert get_presidio_model() is get_presidio_model()

    def test_no_cache(self):
        m1 = get_presidio_model(use_cache=False)
        m2 = get_presidio_model(use_cache=False)
        assert m1 is not m2

    def test_with_recognizers(self):
        from presidio_analyzer.predefined_recognizers import IbanRecognizer
        from detectors.presidio import fuzzy_pii_recognizer
        recognizers = fuzzy_pii_recognizer(
            [(IbanRecognizer(), [dict(substitutions=1)])]
        )
        model = get_presidio_model(recognizers=recognizers, use_cache=False)
        assert model is not None


class TestFilterResults:

    def test_empty(self):
        assert filter_results([]) == []

    def test_no_overlap(self):
        results = [
            {"start": 0, "end": 5, "score": 0.9, "type": "ssn"},
            {"start": 10, "end": 15, "score": 0.8, "type": "email"},
        ]
        assert len(filter_results(results)) == 2

    def test_overlapping_keeps_highest_score(self):
        results = [
            {"start": 0, "end": 10, "score": 0.5, "type": "ssn"},
            {"start": 5, "end": 15, "score": 0.9, "type": "ssn"},
        ]
        filtered = filter_results(results)
        assert len(filtered) == 1
        assert filtered[0]["score"] == 0.9


@pytest.mark.parametrize(
    "text, expected",
    [
        ("My phone number is 123.", []),
        (
            "My phone number is 123-456-7890.",
            [{
                "end": 31,
                "recognizer": "PhoneRecognizer",
                "score": 0.75,
                "start": 19,
                "type": "phone_number",
                "value": "123-456-7890"
            }]
        ),
        ("My phone number is ₁₂₃-₄₅₆-₇₈₉₀", []),
    ]
)
def test_presidio_pii_analyzer(text, expected):
    assert presidio_pii_analyzer(text) == expected


def test_presidio_none_returns_empty():
    assert presidio_pii_analyzer(None) == []
