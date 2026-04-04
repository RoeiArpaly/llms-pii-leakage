"""Separator injection attack: inserts random separator characters between
characters of a PII value to disrupt pattern-based detection.
"""
import random

from data_manipulation.constants import SEPARATORS


def inject_separator(text: str, separator: str = None, proba: float = 1.0) -> str:

    if not separator:
        separator = random.choice(SEPARATORS)

    text = list(text)
    for i, letter in enumerate(text):
        if 0 < i < len(text):
            if random.random() < proba:
                text[i] = separator + letter
    return "".join(text)
