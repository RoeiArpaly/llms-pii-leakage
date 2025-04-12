from data_manipulation.constants import WORD_SYMBOLS_MAP


def word_symbols(text: str, separators="()") -> str:
    text = list(text)
    for i, char in enumerate(text):
        if char in WORD_SYMBOLS_MAP:
            text[i] = separators[0] + WORD_SYMBOLS_MAP[char] + separators[1]
    return "".join(text)
