"""Emojify attack: replaces alphanumeric characters and symbols in PII values
with their emoji equivalents (keycap digits, squared letters, etc.).
"""
import random

from data_manipulation.constants import (
    ALPHABET_EMOJI_MAP,
    NUMBER_EMOJI_MAP,
    SYMBOL_EMOJI_MAP,
)


def emojify_pii(text: str, proba: float = 1.0) -> str:
    """Replacing all numbers and alphabet characters with corresponding emojis."""
    text = list(text)
    for i, l in enumerate(text):
        key = l.upper() if random.random() < proba else None
        if l.isalpha():
            text[i] = ALPHABET_EMOJI_MAP.get(key, l)
        elif l.isdigit():
            text[i] = NUMBER_EMOJI_MAP.get(key, l)
        elif l.isascii():
            text[i] = SYMBOL_EMOJI_MAP.get(key, l)
    return "".join(text)
