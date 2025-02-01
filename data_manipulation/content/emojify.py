from data_manipulation.constants import (
    CREDIT_CARD_VARIATIONS,
    EMAIL_VARIATIONS,
    IBAN_VARIATIONS,
    PHONE_VARIATIONS,
    PII_EMOJI_MAP,
    SSN_VARIATIONS,
)
from data_manipulation.content.utils import replacer


def emojify_pii_entity(text: str) -> str:
    """
    Emojify PII names with emojis.
    """
    configs = [
        {
            "pii_entity": "CREDIT_CARD",
            "replace_value": PII_EMOJI_MAP["CREDIT_CARD"],
            "variations": CREDIT_CARD_VARIATIONS,
        },
        {
            "pii_entity": "SSN",
            "replace_value": PII_EMOJI_MAP["SSN"],
            "variations": SSN_VARIATIONS,
        },
        {
            "pii_entity": "BANK_ACCOUNT",
            "replace_value": PII_EMOJI_MAP["BANK_ACCOUNT"],
            "variations": IBAN_VARIATIONS,
        },
        {
            "pii_entity": "PHONE_NUMBER",
            "replace_value": PII_EMOJI_MAP["PHONE_NUMBER"],
            "variations": PHONE_VARIATIONS,
        },
        {
            "pii_entity": "EMAIL",
            "replace_value": PII_EMOJI_MAP["EMAIL"],
            "variations": EMAIL_VARIATIONS,
        },
    ]
    result = replacer(text=text, configs=configs)
    return result
