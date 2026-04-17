"""Supportive context attack: replaces PII entity names (e.g. "credit card")
with emoji, homoglyph, or slang equivalents to remove contextual cues that
help detectors identify nearby PII values.

Optionally also fuzzes detection-enhancing surrounding terms (financial
vocabulary for CC PII, identity vocabulary for SSN, etc.) with character-
level homoglyph or emojify — preserving human readability while splitting
the tokenizer's topic signal.
"""
import re

from copy import deepcopy

from data_manipulation.attacks.red_teaming.content.utils import replacer
from data_manipulation.attacks.red_teaming.pii.emojify import emojify_pii
from data_manipulation.attacks.red_teaming.pii.homoglyph import homoglyph
from data_manipulation.constants import (
    CREDIT_CARD_VARIATIONS,
    DETECTION_ENHANCING_TERMS,
    EMAIL_VARIATIONS,
    IBAN_VARIATIONS,
    PHONE_VARIATIONS,
    PII_EMOJI_MAP,
    PII_HOMOGLYPH_MAP,
    PII_SLANG_MAP,
    SSN_VARIATIONS,
)
from logger import logger


_FUZZERS = {
    "homoglyph": homoglyph,
    "emojify": emojify_pii,
}


def _fuzz_detection_terms(text: str, pii_types: list[str], fuzz_method: str) -> str:
    """Apply the chosen fuzzer to whole-word matches of detection-enhancing
    terms for each PII type present in the sample.
    """
    fuzz_fn = _FUZZERS[fuzz_method]
    terms: set[str] = set()
    for t in pii_types:
        terms.update(DETECTION_ENHANCING_TERMS.get(t, []))
    # Longest first to avoid sub-matches like "account" inside "accounts".
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(
            r"\b" + re.escape(term) + r"\b", flags=re.IGNORECASE,
        )
        text = pattern.sub(lambda m: fuzz_fn(m.group(0)), text)
    return text


def supportive_context(
        text: str,
        spans: list[dict],
        replace_with: str = "emoji",
        update_spans: bool = True,
        fuzz_surrounding: bool = True,
        fuzz_method: str = "homoglyph",
) -> tuple:

    if replace_with == "emoji":
        replace_value_map = PII_EMOJI_MAP
    elif replace_with == "homoglyph":
        replace_value_map = PII_HOMOGLYPH_MAP
    elif replace_with == "slang":
        replace_value_map = PII_SLANG_MAP
    else:
        raise ValueError(f"Unsupported replacement value: {replace_with}")

    if fuzz_surrounding and fuzz_method not in _FUZZERS:
        raise ValueError(f"Unsupported fuzz_method: {fuzz_method}")

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

    if fuzz_surrounding:
        pii_types = list({s.get("type") for s in spans if s.get("type")})
        result = _fuzz_detection_terms(result, pii_types, fuzz_method)

    if update_spans:  # Search the spans in the new text
        last_idx = 0
        for i, span in enumerate(spans):
            # search the span in the new text (result), handle case of multiple occurrences
            new_span = new_spans[i]
            key = "value_fuzzy" if "value_fuzzy" in span else "value"
            start = result.find(span[key], last_idx)
            if start == -1:
                logger.warning(f"Span {span[key]} not found in the text.")
                continue
            end = start + len(span[key])
            new_span["start"] = start
            new_span["end"] = end
            if start != -1:
                last_idx = end
    return result, new_spans
