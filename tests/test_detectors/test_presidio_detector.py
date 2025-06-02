import pytest

from detectors.presidio_detector import presidio_pii_analyzer


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "My phone number is 123.", []
        ),
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
        (
            "My phone number is ₁₂₃-₄₅₆-₇₈₉₀", []
        ),
    ]
)
def test_presidio_pii_analyzer(text, expected):
    assert presidio_pii_analyzer(text) == expected
