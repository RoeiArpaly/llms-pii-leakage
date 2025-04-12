import pytest

from data_manipulation.constants import (
    CREDIT_CARD_VARIATIONS,
    IBAN_VARIATIONS,
    PII_EMOJI_MAP,
)
from data_manipulation.content.utils import replacer


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "While making a purchase, I used my 3533434814041068. Costing $499.99",
            "While making a purchase, I used my 3533434814041068. Costing $499.99",
        ),
        (
            "My Credit Card is 1234 and my IBAN is DE89370400440532013000 ok?",
            "My 💳 is 1234 and my 🏦 is DE89370400440532013000 ok?",
        ),
    ],
)
def test_replacer(text, expected):
    configs = [
        {
            "pii_entity": "CREDIT_CARD",
            "replace_value": PII_EMOJI_MAP["CREDIT_CARD"],
            "variations": CREDIT_CARD_VARIATIONS,
        },
        {
            "pii_entity": "BANK_ACCOUNT",
            "replace_value": PII_EMOJI_MAP["BANK_ACCOUNT"],
            "variations": IBAN_VARIATIONS,
        },
    ]
    result = replacer(text=text, configs=configs)
    assert result == expected
