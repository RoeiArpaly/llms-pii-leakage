import pytest

from data_manipulation.attacks.red_teaming import (
    homoglyph,
    char_to_word,
    reverse_pii,
)
from data_manipulation.attacks.red_teaming.pii.utils import fuzzy_pii_injection


@pytest.mark.parametrize(
    "text, spans, fuzzy_func, fuzzy_func_kwargs, expected",
    [
        (
            "My Credit Card is 1234 and my IBAN is DE89370400440532013000 ok?",
            [
                {"value": "1234", "start": 18, "end": 22, "type": "CREDIT_CARD"},
                {"value": "DE89370400440532013000", "start": 38, "end": 60, "type": "IBAN"},
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
                        "end": 22,
                        "type": "CREDIT_CARD",
                    },
                    {
                        "value": "DE89370400440532013000",
                        "value_fuzzy": "00031023504400407398ED",
                        "start": 38,
                        "end": 60,
                        "type": "IBAN",
                    },
                ]
            ),
        ),
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
            char_to_word,
            {"lang": "english"},
            (
                "My Credit Card is (one)(two)(three)(four)",
                [
                    {
                        "value": "1234",
                        "value_fuzzy": "(one)(two)(three)(four)",
                        "start": 18,
                        "end": 41,
                        "type": "CREDIT_CARD",
                    },
                ]
            ),
        ),
        (
            "My Credit Card is 1234 and email is john.doe@gmail.com",
            [
                {"value": "1234", "start": 18, "end": 22, "type": "CREDIT_CARD"},
                {"value": "john.doe@gmail.com", "start": 36, "end": 54, "type": "EMAIL"},
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
                        "end": 22,
                        "type": "CREDIT_CARD",
                    },
                    {
                        "value": "john.doe@gmail.com",
                        "value_fuzzy": "јоһп．ԁое＠ɡмаіӏ．сом",
                        "start": 36,
                        "end": 54,
                        "type": "EMAIL"
                    },
                ]
            ),
        ),
        (
            "My Credit Card is 123 and email is john.doe@gmail.com",
            [
                {"value": "123", "start": 18, "end": 21, "type": "CREDIT_CARD"},
                {"value": "john.doe@gmail.com", "start": 35, "end": 53, "type": "EMAIL"},
            ],
            char_to_word,
            {"lang": "english"},
            (
                "My Credit Card is (one)(two)(three) and email is john(dot)doe(at)gmail(dot)com",
                [
                    {
                        "value": "123",
                        "value_fuzzy": "(one)(two)(three)",
                        "start": 18,
                        "end": 35,
                        "type": "CREDIT_CARD",
                    },
                    {
                        "value": "john.doe@gmail.com",
                        "value_fuzzy": "john(dot)doe(at)gmail(dot)com",
                        "start": 49,
                        "end": 78,
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
