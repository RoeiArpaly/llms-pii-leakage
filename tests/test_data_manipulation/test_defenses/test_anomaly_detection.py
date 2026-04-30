"""Tests for character frequency anomaly detection.

is_suspicious() detects text that has been adversarially modified
using homoglyphs, emoji, combining marks, or zero-width characters.
It returns True when the weighted non-ASCII score exceeds a threshold.

Used in the defense pipeline to decide between light (SLM-friendly)
and full (aggressive) preprocessing.
"""
import pytest

from data_manipulation.defenses.preprocess import is_suspicious


class TestCleanTextNotFlagged:
    """Normal English text should never be flagged."""

    @pytest.mark.parametrize("text", [
        "My SSN is 219-09-9999.",
        "Send the report to john@example.com by Friday.",
        "What are the best practices for code review?",
        "The SHA-256 hash is a3f2b8c91d4e7f00.",
        "Order #4532-8891-0012 shipped yesterday.",
        "",
    ])
    def test_normal_text(self, text):
        assert is_suspicious(text) is False


class TestAttackedTextFlagged:
    """Text with homoglyphs, emoji, combining marks, or zero-width
    chars should be flagged as suspicious.
    """

    def test_homoglyph_cyrillic(self):
        """Single Cyrillic letter (score=1) doesn't trigger,
        but 3+ homoglyphs (score=3) exceed threshold=2.
        """
        assert is_suspicious("My SSN is \u0430 219-09-9999.") is False
        assert is_suspicious(
            "M\u0443 SS\u041d i\u0455 219-09-9999.",
        ) is True

    def test_fullwidth_digits(self):
        """Fullwidth digits ２１９ (U+FF12 etc.)."""
        assert is_suspicious("My SSN is \uff12\uff11\uff19-09-9999.")

    def test_emoji_keycap(self):
        """Keycap digits 2️⃣1️⃣9️⃣."""
        assert is_suspicious("My SSN is 2\ufe0f\u20e31\ufe0f\u20e39.")

    def test_combining_marks(self):
        """Combining dot above (U+0307)."""
        assert is_suspicious("My SSN is 2\u03071\u03079\u0307.")

    def test_zero_width_chars(self):
        """Zero-width space (U+200B)."""
        assert is_suspicious("My SSN is 2\u200b1\u200b9-09-9999.")

    def test_bom_injection(self):
        """BOM (U+FEFF) between chars."""
        assert is_suspicious("My SSN is 2\ufeff1\ufeff9-09-9999.")

    def test_invisible_chars_attack(self):
        """Mixed invisible characters."""
        assert is_suspicious("2\u200b1\u200c9\u200d-\u20600\u00ad9")


class TestThreshold:

    def test_single_non_ascii_below_threshold(self):
        """One accented letter (score=1) is below default threshold=2."""
        assert is_suspicious("café") is False

    def test_custom_threshold(self):
        """Lower threshold catches more."""
        text = "My SSN is \u0430 219."  # one Cyrillic, score=1
        assert is_suspicious(text, threshold=0) is True
        assert is_suspicious(text, threshold=2) is False

    def test_combining_mark_high_weight(self):
        """A single combining mark (weight=3) exceeds default threshold=2."""
        assert is_suspicious("a\u0307") is True
