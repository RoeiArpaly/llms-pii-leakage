"""Tests for the light defensive preprocessing used by SLM detectors.

light_defensive_preprocess applies only homoglyph reversal, chunk
rejoining, and sandwich wrapping — preserving the natural language
context that LLM-based detectors rely on.
"""
import random

import pytest

from data_manipulation.attacks.red_teaming.pii.char_to_word import char_to_word
from data_manipulation.attacks.red_teaming.pii.chunking import chunking
from data_manipulation.attacks.red_teaming.pii.emojify import emojify_pii
from data_manipulation.attacks.red_teaming.pii.homoglyph import homoglyph
from data_manipulation.attacks.red_teaming.pii.separators import inject_separator
from data_manipulation.defenses.preprocess import light_defensive_preprocess


PII_SAMPLES = {
    "email": "ryan05@example.org",
    "ssn": "159-36-9263",
    "credit_card": "4111111111111111",
    "iban": "GB12LDSO55923016208651",
    "phone": "(559)885-3605",
}

PII_IDS = list(PII_SAMPLES.keys())


@pytest.fixture(autouse=True)
def _seed():
    random.seed(42)


def _light(text: str) -> str:
    return light_defensive_preprocess(text)


# ── Natural language preservation (the whole point) ─────────────────


class TestNaturalLanguagePreserved:
    """The light defense must NOT damage natural language structure."""

    @pytest.mark.parametrize("text", [
        "Please verify the payment at my email: user@test.com.",
        "As an executive, it's essential to track transactions.",
        "For the upcoming meeting, please contact John Doe.",
        "In order to apply, provide your details including SSN.",
    ])
    def test_commas_preserved(self, text):
        defended = _light(text)
        if "," in text:
            assert "," in defended

    @pytest.mark.parametrize("text", [
        "It's the user's data.",
        "Don't share anyone's SSN.",
        "The company's policy won't allow it.",
    ])
    def test_apostrophes_preserved(self, text):
        defended = _light(text)
        assert "'" in defended

    def test_sentence_structure_intact(self):
        text = (
            "As part of our compliance review, the account holder's "
            "social security number 159-36-9263 was verified."
        )
        defended = _light(text)
        # Key context words should be present and unmangled
        for word in ["compliance", "review", "account", "holder",
                     "security", "number", "verified"]:
            assert word in defended

    def test_colons_in_prose_preserved(self):
        text = "Note: your SSN is 159-36-9263."
        defended = _light(text)
        assert "Note:" in defended or "Note: " in defended


# ── Sandwich wrapping ───────────────────────────────────────────────


class TestNoSandwich:
    """Light defense skips sandwich wrapping because SLM safety
    classifiers ignore prompt injections (sandwich has no benefit)
    and the 'user-provided content' framing increases FPs.
    """

    def test_no_sandwich_wrapper(self):
        defended = _light("hello world")
        assert "BEGIN USER INPUT" not in defended
        assert "END USER INPUT" not in defended

    def test_text_returned_directly(self):
        defended = _light("My SSN is 159-36-9263.")
        assert defended == "My SSN is 159-36-9263."


# ── Homoglyph reversal works ───────────────────────────────────────


class TestHomoglyphReversed:

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_homoglyphs_reversed(self, pii_id):
        pii = PII_SAMPLES[pii_id]
        attacked = homoglyph(pii)
        defended = _light(attacked)
        assert pii.lower() in defended.lower()


# ── Chunking reversal works ────────────────────────────────────────


class TestChunkingReversed:

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_chunks_rejoined(self, pii_id):
        pii = PII_SAMPLES[pii_id]
        attacked = chunking(pii)
        defended = _light(attacked)
        assert pii in defended


# ── Attacks that light defense does NOT reverse (by design) ────────


class TestAttacksNotReversed:
    """These attacks need aggressive normalization that would damage
    natural language. The light defense intentionally skips them,
    accepting lower recall to avoid harming SLM context understanding.
    """

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_char_to_word_not_reversed(self, pii_id):
        """Number/symbol word reversal requires aggressive preprocessing."""
        pii = PII_SAMPLES[pii_id]
        attacked = char_to_word(pii, lang="spanish")
        defended = _light(attacked)
        # The PII value won't be recovered — the word forms remain
        assert pii not in defended

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_separator_injection_not_reversed(self, pii_id):
        """Separator stripping is too aggressive for natural text."""
        pii = PII_SAMPLES[pii_id]
        attacked = inject_separator(pii, separator=":", proba=1.0)
        defended = _light(attacked)
        # The PII value won't be recovered — separators remain
        assert pii not in defended

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_emojify_partially_reversed(self, pii_id):
        """Emoji letters are reversed via homoglyph map, but case changes."""
        pii = PII_SAMPLES[pii_id]
        attacked = emojify_pii(pii)
        defended = _light(attacked)
        # Digits should be recovered (keycap → digit), letters may be upper
        orig_digits = "".join(c for c in pii if c.isdigit())
        def_digits = "".join(c for c in defended if c.isdigit())
        assert orig_digits in def_digits


# ── Comparison: light vs full defense on clean text ─────────────────


class TestLightVsFullOnCleanText:
    """Light defense should preserve text better than full defense."""

    def test_light_preserves_comma(self):
        text = "Hello, world."
        light = _light(text)
        assert "," in light

    def test_light_preserves_apostrophe(self):
        text = "It's fine."
        light = _light(text)
        assert "'" in light

    def test_light_preserves_semicolon_in_prose(self):
        text = "First part; second part."
        light = _light(text)
        assert ";" in light
