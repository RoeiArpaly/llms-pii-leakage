import pytest

from data_manipulation.pii.utils import fuzzy_pii_injection
from data_manipulation.pii import (
    number_to_word,
    homoglyph,
)


@pytest.mark.parametrize(
    "text, spans, fuzzy_func, fuzzy_func_kwargs, expected",
    [
        (
            "My Credit Card is 1234",
            [
                {"value": "1234", "start": 18, "end": 22, "type": "CREDIT_CARD"},
            ],
            homoglyph,
            None,
            (
                "My Credit Card is １２３４",
                [
                    {
                        "value": "1234",
                        "value_fuzzy": "１２３４",
                        "start": 18,
                        "end": 22,
                        "type": "CREDIT_CARD",
                    },
                ]
            ),
        ),
        (
            "My Credit Card is 1234",
            [
                {"value": "1234", "start": 18, "end": 22, "type": "CREDIT_CARD"},
            ],
            number_to_word,
            {"lang": "english"},
            (
                "My Credit Card is one two three four",
                [
                    {
                        "value": "1234",
                        "value_fuzzy": "one two three four",
                        "start": 18,
                        "end": 36,
                        "type": "CREDIT_CARD",
                    },
                ]
            ),
        ),
    ],
)
def test_emojify_text(text, spans, fuzzy_func, fuzzy_func_kwargs, expected):
    assert fuzzy_pii_injection(
        text=text,
        spans=spans,
        fuzzy_func=fuzzy_func,
        fuzzy_func_kwargs=fuzzy_func_kwargs,
    ) == expected
