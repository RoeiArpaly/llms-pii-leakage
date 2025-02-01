import random

from fuzzers.constants import (
    CREDIT_CARD_VARIATIONS,
    EMAIL_VARIATIONS,
    IBAN_VARIATIONS,
    PHONE_VARIATIONS,
    SSN_VARIATIONS,
)
from fuzzers.utils import replacer


NUMBER_EMOJI_MAP = {
    "0": "0️⃣",
    "1": "1️⃣",
    "2": "2️⃣",
    "3": "3️⃣",
    "4": "4️⃣",
    "5": "5️⃣",
    "6": "6️⃣",
    "7": "7️⃣",
    "8": "8️⃣",
    "9": "9️⃣",
}
ALPHABET_EMOJI_MAP = {
    "a": "🅰",
    "b": "🅱",
    "c": "🅲",
    "d": "🅳",
    "e": "🅴",
    "f": "🅵",
    "g": "🅶",
    "h": "🅷",
    "i": "🅸",
    "j": "🅹",
    "k": "🅺",
    "l": "🅻",
    "m": "🅼",
    "n": "🅽",
    "o": "🅾",
    "p": "🅿",
    "q": "🆀",
    "r": "🆁",
    "s": "🆂",
    "t": "🆃",
    "u": "🆄",
    "v": "🆅",
    "w": "🆆",
    "x": "🆇",
    "y": "🆈",
    "z": "🆉",
}
PII_EMOJI_MAP = {
    "CREDIT_CARD": "💳",
    "SSN": "🔒",
    "BANK_ACCOUNT": "🏦",
    "PHONE_NUMBER": "📞",
    "EMAIL": "📧",
}


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


def emojify_pii_entity(text: str) -> str:
    """
    Emojify PII names with emojis.
    """
    configs = [
        {
            "pii_entity": "CREDIT_CARD",
            "replace_value": PII_EMOJI_MAP["CREDIT_CARD"],
            "variations": CREDIT_CARD_VARIATIONS,
        },
        {
            "pii_entity": "SSN",
            "replace_value": PII_EMOJI_MAP["SSN"],
            "variations": SSN_VARIATIONS,
        },
        {
            "pii_entity": "BANK_ACCOUNT",
            "replace_value": PII_EMOJI_MAP["BANK_ACCOUNT"],
            "variations": IBAN_VARIATIONS,
        },
        {
            "pii_entity": "PHONE_NUMBER",
            "replace_value": PII_EMOJI_MAP["PHONE_NUMBER"],
            "variations": PHONE_VARIATIONS,
        },
        {
            "pii_entity": "EMAIL",
            "replace_value": PII_EMOJI_MAP["EMAIL"],
            "variations": EMAIL_VARIATIONS,
        },
    ]
    result = replacer(text=text, configs=configs)
    return result
