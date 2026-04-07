"""Locate PII span values inside defensively-preprocessed text.

defensive_preprocess applies homoglyph reversal, separator cleanup,
repeat-character collapse, and sandwich wrapping. This module finds
where a PII value (which may have been obfuscated) ends up in the
defended text, accounting for all those transformations.
"""
import re

from data_manipulation.defenses.preprocess import (
    defensive_preprocess,
    transform_homoglyphs_to_alphabets,
)


def _clean_homoglyphs(value: str) -> str:
    """Reverse homoglyphs and strip leftover emoji delimiters."""
    cleaned = transform_homoglyphs_to_alphabets(value)["text"]
    return re.sub(r"\|+", "", cleaned)


def _build_fuzzy_pattern(cleaned: str) -> str:
    """Build a regex that allows collapsed character repeats."""
    parts = []
    i = 0
    while i < len(cleaned):
        ch = re.escape(cleaned[i])
        j = i + 1
        while j < len(cleaned) and cleaned[j] == cleaned[i]:
            j += 1
        run_len = j - i
        if run_len >= 2:
            parts.append(f"{ch}{{1,{run_len}}}")
        else:
            parts.append(ch)
        i = j
    return "".join(parts)


def locate_span_in_defended(
    raw_fragment: str,
    defended_text: str,
    original_value: str = "",
) -> tuple[int | None, int | None]:
    """Find a PII value in defended text.

    Parameters
    ----------
    raw_fragment
        The actual text extracted from the raw input at the span offsets.
        May be obfuscated (homoglyphs, emojified, chunked, etc.).
    defended_text
        The full defensively-preprocessed text (with sandwich wrapper).
    original_value
        The clean, unobfuscated PII value. Used as a fallback.

    Returns
    -------
    tuple
        (start, end) character offsets, or (None, None) if not found.
    """
    if not raw_fragment and not original_value:
        return None, None

    candidates = [raw_fragment, original_value]

    for val in candidates:
        if not val:
            continue

        # 1. Exact match
        idx = defended_text.find(val)
        if idx >= 0:
            return idx, idx + len(val)

    for val in candidates:
        if not val:
            continue

        # 2. Homoglyph-cleaned match
        cleaned = _clean_homoglyphs(val)
        idx = defended_text.find(cleaned)
        if idx >= 0:
            return idx, idx + len(cleaned)

    for val in candidates:
        if not val:
            continue

        # 3. Preprocess the fragment the same way as the full text
        #    (without sandwich) — handles all attack reversals
        preprocessed = defensive_preprocess(val, include_sandwich=False)
        if preprocessed:
            idx = defended_text.find(preprocessed)
            if idx >= 0:
                return idx, idx + len(preprocessed)

    for val in candidates:
        if not val:
            continue

        # 4. Regex allowing collapsed character repeats
        cleaned = _clean_homoglyphs(val)
        pattern = _build_fuzzy_pattern(cleaned)
        m = re.search(pattern, defended_text)
        if m:
            return m.start(), m.end()

    return None, None
