import random

from data_manipulation.constants import (
    ALPHABET_EMOJI_MAP,
    NUMBER_EMOJI_MAP,
)


def emojify_pii(text: str, proba: float = 1.0) -> str:
    """Replacing all numbers and alphabet characters with corresponding emojis."""
    text = list(text)
    for i, l in enumerate(text):
        key = l.lower() if random.random() < proba else None
        if l.isalpha():
            text[i] = ALPHABET_EMOJI_MAP.get(key, l)
        elif l.isdigit():
            text[i] = NUMBER_EMOJI_MAP.get(key, l)
    return "".join(text)
