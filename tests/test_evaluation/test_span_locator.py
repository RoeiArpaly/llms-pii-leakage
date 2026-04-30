"""Tests for locating PII spans in defensively-preprocessed text.

Uses real examples from the dataset to verify span location works
across clean, homoglyph-attacked, and emojified inputs.
"""
import pytest

from data_manipulation.defenses.preprocess import defensive_preprocess
from evaluation.report.span_locator import locate_span_in_defended


def _defended(raw: str) -> str:
    return defensive_preprocess(raw)


def _assert_found(defended, start, end, label=""):
    assert start is not None, f"Span not found: {label}"
    assert end is not None
    assert start < end
    assert end <= len(defended)


class TestCleanValues:
    """Baseline samples — PII values are unobfuscated."""

    @pytest.mark.parametrize("raw, value", [
        (
            "Please send a report to ryan05@example.org.",
            "ryan05@example.org",
        ),
        (
            "IBAN GB82WEST12345698765432 for transfer.",
            "GB82WEST12345698765432",
        ),
        (
            "My SSN is 159-36-9263 for verification.",
            "159-36-9263",
        ),
        (
            "Card number 213108651504676 on file.",
            "213108651504676",
        ),
        (
            "Call me at (559)885-3605x3781 anytime.",
            "(559)885-3605x3781",
        ),
    ])
    def test_exact_match(self, raw, value):
        defended = _defended(raw)
        s, e = locate_span_in_defended(value, defended, original_value=value)
        _assert_found(defended, s, e, value)
        assert defended[s:e] == value


class TestObfuscatedFragments:
    """Attacked samples — raw fragment is obfuscated, original is clean."""

    def test_emojified_email(self):
        """(cero)(cinco)(at)(dot) style obfuscation."""
        raw = "Contact ryan(cero)(cinco)(at)example(dot)org for info."
        defended = _defended(raw)
        raw_frag = "ryan(cero)(cinco)(at)example(dot)org"
        orig = "ryan05@example.org"
        s, e = locate_span_in_defended(raw_frag, defended, original_value=orig)
        _assert_found(defended, s, e, orig)
        matched = defended[s:e]
        # The defended text should contain the (partially) reversed form
        assert "ryan" in matched
        assert "example" in matched

    def test_fullwidth_digits_ssn(self):
        """Fullwidth digit homoglyphs: ２１９ -> 219."""
        raw = "My SSN is \uff12\uff11\uff19-\uff10\uff19-\uff19\uff19\uff19\uff19 please."
        defended = _defended(raw)
        raw_frag = "\uff12\uff11\uff19-\uff10\uff19-\uff19\uff19\uff19\uff19"
        orig = "219-09-9999"
        s, e = locate_span_in_defended(raw_frag, defended, original_value=orig)
        _assert_found(defended, s, e, orig)
        matched = defended[s:e]
        assert "219" in matched

    def test_fullwidth_credit_card(self):
        """Fullwidth digits in credit card number."""
        raw = "Card \uff14\uff11\uff11\uff11111111111111 here."
        defended = _defended(raw)
        raw_frag = "\uff14\uff11\uff11\uff11111111111111"
        orig = "4111111111111111"
        s, e = locate_span_in_defended(raw_frag, defended, original_value=orig)
        _assert_found(defended, s, e, orig)

    def test_preprocessed_fragment_match(self):
        """Fragment that needs full defensive_preprocess to match."""
        raw = "Send to ryan(cero)(cinco)(at)example(dot)org please."
        defended = _defended(raw)
        raw_frag = "ryan(cero)(cinco)(at)example(dot)org"
        orig = "ryan05@example.org"
        s, e = locate_span_in_defended(raw_frag, defended, original_value=orig)
        _assert_found(defended, s, e, orig)


class TestEdgeCases:

    def test_empty_values(self):
        s, e = locate_span_in_defended("", "some text", original_value="")
        assert s is None and e is None

    def test_value_not_in_text(self):
        defended = _defended("Hello world, no PII here.")
        s, e = locate_span_in_defended(
            "999-99-9999", defended, original_value="999-99-9999",
        )
        assert s is None and e is None

    def test_offset_past_sandwich_prefix(self):
        """Found offsets must account for the sandwich prefix."""
        raw = "Email john@test.com here."
        defended = _defended(raw)
        s, e = locate_span_in_defended(
            "john@test.com", defended, original_value="john@test.com",
        )
        _assert_found(defended, s, e)
        # Sandwich prefix is ~170 chars
        assert s > 100

    def test_multiple_spans_dont_overlap(self):
        raw = "SSN 219-09-9999 and email test@foo.com here."
        defended = _defended(raw)
        s1, e1 = locate_span_in_defended(
            "219-09-9999", defended, original_value="219-09-9999",
        )
        s2, e2 = locate_span_in_defended(
            "test@foo.com", defended, original_value="test@foo.com",
        )
        _assert_found(defended, s1, e1, "ssn")
        _assert_found(defended, s2, e2, "email")
        assert e1 <= s2 or e2 <= s1

    def test_original_value_fallback(self):
        """When raw_fragment is garbage, falls back to original_value."""
        raw = "My IBAN is GB82WEST12345698765432."
        defended = _defended(raw)
        s, e = locate_span_in_defended(
            "completely-wrong-fragment",
            defended,
            original_value="GB82WEST12345698765432",
        )
        _assert_found(defended, s, e, "IBAN")
        assert defended[s:e] == "GB82WEST12345698765432"
