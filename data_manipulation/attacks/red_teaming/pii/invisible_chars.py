"""Invisible character injection: inserts invisible or near-invisible
Unicode characters into PII values to break pattern matching.

Two categories of characters:
- Zero-width: inserted between chars, completely invisible (ZWSP, ZWJ, etc.)
- Combining: stacked on chars as faint diacritics (dot above, macron, etc.)

Both break regex detectors while preserving human readability.
SLM tokenizers generally normalize these away.
"""
import random


_INVISIBLE_CHARS = [
    # Zero-width (inserted between chars)
    "\u200B",  # zero-width space
    "\u200C",  # zero-width non-joiner
    "\u200D",  # zero-width joiner
    "\u2060",  # word joiner
    "\u00AD",  # soft hyphen
    "\uFEFF",  # BOM / zero-width no-break space
    # Combining (stacked on chars)
    "\u0307",  # combining dot above
    "\u0304",  # combining macron
    "\u0308",  # combining diaeresis
    "\u0301",  # combining acute accent
    "\u0327",  # combining cedilla
]

# Subsets for targeted use
ZERO_WIDTH_CHARS = _INVISIBLE_CHARS[:6]
COMBINING_CHARS = _INVISIBLE_CHARS[6:]


def invisible_inject(
    text: str, char: str = None, proba: float = 1.0,
) -> str:
    """Insert invisible characters into a PII value.

    Parameters
    ----------
    text
        The PII value to inject into.
    char
        Specific character to use. If None, randomly chosen from full pool.
    proba
        Probability of injection at each position.
    """
    if not char:
        char = random.choice(_INVISIBLE_CHARS)

    is_combining = char in COMBINING_CHARS
    result = []
    for i, c in enumerate(text):
        result.append(c)
        if is_combining:
            # Combining chars go after alphanumeric chars
            if c.isalnum() and random.random() < proba:
                result.append(char)
        else:
            # Zero-width chars go between any chars
            if i < len(text) - 1 and random.random() < proba:
                result.append(char)
    return "".join(result)
