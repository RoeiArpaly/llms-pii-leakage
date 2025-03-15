from copy import deepcopy

from data_manipulation.constants import (
    CREDIT_CARD_VARIATIONS,
    EMAIL_VARIATIONS,
    PII_HOMOGLYPH_MAP,
    IBAN_VARIATIONS,
    PHONE_VARIATIONS,
    PII_EMOJI_MAP,
    SSN_VARIATIONS,
)
from data_manipulation.content.utils import replacer


def supportive_context(
        text: str,
        spans: list[dict],
        replace_with: str = "emoji",
        update_spans: bool = True,
) -> tuple:

    if replace_with == "emoji":
        replace_value_map = PII_EMOJI_MAP
    elif replace_with == "homoglyph":
        replace_value_map = PII_HOMOGLYPH_MAP
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

    new_spans = deepcopy(spans)
    result = replacer(text=text, configs=configs)

    if update_spans:  # Search the spans in the new text
        last_idx = 0
        for i, span in enumerate(spans):
            # search the span in the new text (result), handle case of multiple occurrences
            new_span = new_spans[i]
            start = result.find(span["value"], last_idx)
            end = start + len(span["value"])
            new_span["start"] = start
            new_span["end"] = end
            if start != -1:
                last_idx = end
    return result, new_spans
