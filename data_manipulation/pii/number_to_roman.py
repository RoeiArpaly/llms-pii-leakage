import random
import re


def int_to_roman(num: int) -> str:
    """Convert an integer to a Roman numeral."""
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


# Example usage
print(number_to_roman(
    text="The number 2021 is the current year 3.",
    spans=[{"start": 11, "end": 15}, {"start": 36, "end": 37}],
))

print(
    number_to_roman(
        text="is 1990 and 2021 and 2025",
        spans=[{"start": 3, "end": 7}, {"start": 21, "end": 25}],
    )
)
