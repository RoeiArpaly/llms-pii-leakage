import random

from data_manipulation.constants import (
    ALPHABET_EMOJI_MAP,
    NUMBER_EMOJI_MAP,
)


def emojify_pii(text: str, spans: list, proba=1) -> str:
    """
    Replacing all numbers and alphabet characters with corresponding emojis.

    Parameters
    ----------
    text : str
        The text to replace the PII entities.
    spans : dict
        A dictionary containing the start and end indices of the PII entities.
    proba : float
        The probability of replacing the characters with

    Returns
    -------
    str

    """
    text = list(text)
    for span in spans:
        start, end = span["start"], span["end"]
        for i in range(start, end):
            char = text[i]
            key = char.lower() if random.random() < proba else None
            if char.isalpha():
                text[i] = ALPHABET_EMOJI_MAP.get(key, char)
            elif char.isdigit():
                text[i] = NUMBER_EMOJI_MAP.get(key, char)
    return "".join(text)
