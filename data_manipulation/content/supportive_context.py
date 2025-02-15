from data_manipulation.constants import (
    CREDIT_CARD_VARIATIONS,
    EMAIL_VARIATIONS,
    IBAN_VARIATIONS,
    PHONE_VARIATIONS,
    PII_EMOJI_MAP,
    SSN_VARIATIONS,
)
from data_manipulation.content.utils import replacer


def supportive_context(text: str, replace_with: str = "emoji") -> str:

    if replace_with == "emoji":
        replace_value_map = PII_EMOJI_MAP
    else:
        raise ValueError(f"Unsupported replacement value: {replace_with}")

    configs = [
        {
            "pii_entity": "CREDIT_CARD",
            "replace_value": replace_value_map["CREDIT_CARD"],
            "variations": CREDIT_CARD_VARIATIONS,
        },
        {
            "pii_entity": "SSN",
            "replace_value": replace_value_map["SSN"],
            "variations": SSN_VARIATIONS,
        },
        {
            "pii_entity": "BANK_ACCOUNT",
            "replace_value": replace_value_map["BANK_ACCOUNT"],
            "variations": IBAN_VARIATIONS,
        },
        {
            "pii_entity": "PHONE_NUMBER",
            "replace_value": replace_value_map["PHONE_NUMBER"],
            "variations": PHONE_VARIATIONS,
        },
        {
            "pii_entity": "EMAIL",
            "replace_value": replace_value_map["EMAIL"],
            "variations": EMAIL_VARIATIONS,
        },
    ]
    result = replacer(text=text, configs=configs)
    return result
