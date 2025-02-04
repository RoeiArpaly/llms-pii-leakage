import random

from data_manipulation.constants import NUMBER_WORD_MAP


def number_to_word(text: str, spans: list, lang: str = None, proba: float = 1.0) -> str:

    if not lang:
        lang = random.choice(list(NUMBER_WORD_MAP.keys()))

    number_word_map = NUMBER_WORD_MAP[lang]
    text = list(text)
    prev = False
    for span in spans:
        start, end = span["start"], span["end"]
        for i in range(start, end):
            if i < len(text):
                char = text[i]
                key = char.lower() if random.random() < proba else None
                if char.isdigit():
                    word = number_word_map.get(key, char)
                    if prev:
                        word = f"-{word}"
                    text[i] = word
                    prev = True
                else:
                    prev = False
    return "".join(text)
