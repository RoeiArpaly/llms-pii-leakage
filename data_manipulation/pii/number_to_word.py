import random

from data_manipulation.constants import NUMBER_WORD_MAP


def number_to_word(text: str, spans: list, proba: float = 1.0) -> str:

    text = list(text)
    prev = False
    for span in spans:
        start, end = span["start"], span["end"]
        for i in range(start, end):
            if i < len(text):
                char = text[i]
                key = char.lower() if random.random() < proba else None
                if char.isdigit():
                    word = NUMBER_WORD_MAP.get(key, char)
                    if prev:
                        word = f"-{word}"
                    text[i] = word
                    prev = True
                else:
                    prev = False
    return "".join(text)
