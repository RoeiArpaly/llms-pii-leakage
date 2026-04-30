"""Homoglyph substitution attack: replaces ASCII characters with visually
similar Unicode characters (Cyrillic, Greek, fullwidth) to evade regex-based
PII detectors.
"""
from data_manipulation.constants import HOMOGLYPH_MAP


def homoglyph(text: str) -> str:
    text = list(text)
    for i, char in enumerate(text):
        text[i] = HOMOGLYPH_MAP.get(char, char)
    return "".join(text)
