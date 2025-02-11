import random
import re


def int_to_roman(num: int) -> str:
    """Convert an integer to a Roman numeral."""
    if not 0 < num < 4000:  # The Romans didn't have a symbol for 0 or numbers >= 4000
        return str(num)
    val_map = [
        (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
        (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
        (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")
    ]
    roman = ""
    for val, symbol in val_map:
        while num >= val:
            roman += symbol
            num -= val
    return roman


def number_to_roman(text: str, spans: list, proba: float = 1.0) -> str:
    """Replace numbers in a string with their Roman numeral equivalent."""

    text_list = list(text)
    for span in spans[::-1]:
        start, end = span["start"], span["end"]
        if random.random() < proba:
            roman_num = re.sub(
                pattern=r"\b\d+\b",
                repl=lambda match: int_to_roman(int(match.group())),
                string=text[start:end],
            )
            text_list[start:end] = list(roman_num)
    return "".join(text_list)
