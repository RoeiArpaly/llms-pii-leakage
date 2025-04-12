import pytest

from data_manipulation.pii.utils import fuzzy_pii_injection
from data_manipulation.pii import (
    homoglyph,
    number_to_word,
    reverse_pii,
)


@pytest.mark.parametrize(
    "text, spans, fuzzy_func, fuzzy_func_kwargs, expected",
    [
        (
            "My Credit Card is 1234 and my IBAN is DE89370400440532013000 ok?",
            [
                {"value": "1234", "start": 18, "end": 21, "type": "CREDIT_CARD"},
                {"value": "DE89370400440532013000", "start": 38, "end": 59, "type": "IBAN"},
            ],
            reverse_pii,
            None,
            (
                "My Credit Card is 4321 and my IBAN is 00031023504400407398ED ok?",
                [
                    {
                        "value": "1234",
                        "value_fuzzy": "4321",
                        "start": 18,
                        "end": 21,
                        "type": "CREDIT_CARD",
                    },
                    {
                        "value": "DE89370400440532013000",
                        "value_fuzzy": "00031023504400407398ED",
                        "start": 38,
                        "end": 59,
                        "type": "IBAN",
                    },
                ]
            ),
        ),
        (
            "My Credit Card is 1234",
            [
                {"value": "1234", "start": 18, "end": 21, "type": "CREDIT_CARD"},
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
                        "end": 21,
                        "type": "CREDIT_CARD",
                    },
                ]
            ),
        ),
        (
            "My Credit Card is 1234",
            [
                {"value": "1234", "start": 18, "end": 21, "type": "CREDIT_CARD"},
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
                        "end": 35,
                        "type": "CREDIT_CARD",
                    },
                ]
            ),
        ),
        (
            "My Credit Card is 1234 and email is john.doe@gmail.com",
            [
                {"value": "1234", "start": 18, "end": 21, "type": "CREDIT_CARD"},
                {"value": "john.doe@gmail.com", "start": 36, "end": 53, "type": "EMAIL"},
            ],
            homoglyph,
            None,
            (
                "My Credit Card is １２３４ and email is јоһп．ԁое＠ɡмаіӏ．сом",
                [
                    {
                        "value": "1234",
                        "value_fuzzy": "１２３４",
                        "start": 18,
                        "end": 21,
                        "type": "CREDIT_CARD",
                    },
                    {
                        "value": "john.doe@gmail.com",
                        "value_fuzzy": "јоһп．ԁое＠ɡмаіӏ．сом",
                        "start": 36,
                        "end": 53,
                        "type": "EMAIL"
                    },
                ]
            ),
        ),
        (
            "My Credit Card is 1234 and email is john.doe@gmail.com",
            [
                {"value": "1234", "start": 18, "end": 21, "type": "CREDIT_CARD"},
                {"value": "john.doe@gmail.com", "start": 36, "end": 53, "type": "EMAIL"},
            ],
            number_to_word,
            {"lang": "english"},
            (
                "My Credit Card is one two three four and email is john.doe@gmail.com",
                [
                    {
                        "value": "1234",
                        "value_fuzzy": "one two three four",
                        "start": 18,
                        "end": 35,
                        "type": "CREDIT_CARD",
                    },
                    {
                        "value": "john.doe@gmail.com",
                        "value_fuzzy": "john.doe@gmail.com",
                        "start": 50,
                        "end": 67,
                        "type": "EMAIL"
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
