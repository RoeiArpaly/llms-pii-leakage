import random

from data_manipulation.constants import (
    NUMBER_WORD_MAP,
    WORD_SYMBOLS_MAP,
)


def char_to_word(text: str, sep: str = "", separators: str = "()", lang: str = None) -> str:
    if not lang:
        lang = random.choice(list(NUMBER_WORD_MAP.keys()))
    number_word_map = NUMBER_WORD_MAP[lang]
    text = list(text)
    for i, char in enumerate(text):
        if char in WORD_SYMBOLS_MAP:
            text[i] = separators[0] + WORD_SYMBOLS_MAP[char] + separators[1]
        elif char.isdigit():
            value = number_word_map.get(char, char)
            text[i] = sep + separators[0] + value + separators[1]
    return "".join(text)
