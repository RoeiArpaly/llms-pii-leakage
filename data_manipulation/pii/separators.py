import random

from data_manipulation.constants import SEPARATORS


def inject_separator(text: str, spans: list, separator: str = None, proba: float = 1.0) -> str:

    if not separator:
        separator = random.choice(SEPARATORS)

    text = list(text)
    for span in spans[::-1]:
        start, end = span["start"], span["end"]
        for i in range(end - 1, start, -1):
            if i < len(text):
                if random.random() < proba:
                    text.insert(i, separator)
    return "".join(text)
