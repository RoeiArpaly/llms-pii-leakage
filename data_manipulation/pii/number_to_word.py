import random

from data_manipulation.constants import NUMBER_WORD_MAP


def number_to_word(text: str, sep: str = " ", lang: str = None) -> str:

    if not lang:
        lang = random.choice(list(NUMBER_WORD_MAP.keys()))

    number_word_map = NUMBER_WORD_MAP[lang]

    text = list(text)
    prev = False
    for i, char in enumerate(text):
        if char.isdigit():
            value = number_word_map.get(char, char)
            text[i] = value if not prev else f"{sep}{value}"
            prev = True
        else:
            prev = False

    return "".join(text)
