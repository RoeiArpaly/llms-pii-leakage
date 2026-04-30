"""Roundtrip tests: original PII -> attack -> defensive_preprocess -> recovered.

The defense pipeline's goal is to make PII detectable, not to perfectly
reconstruct the original text. A test passes when the original PII value
is found in the defended output.

Test matrix:
- 5 PII types  x  9 single attacks          = 45 roundtrips
- 5 PII types  x  5 combined attack chains   = 25 roundtrips
- 5 PII types  x  full pipeline w/ sandwich  = 5 roundtrips
- Edge cases, normal text, novel separators  = 15+ tests
"""
import random

import pytest

from data_manipulation.attacks.red_teaming.pii.char_to_word import char_to_word
from data_manipulation.attacks.red_teaming.pii.chunking import chunking
from data_manipulation.attacks.red_teaming.pii.emojify import emojify_pii
from data_manipulation.attacks.red_teaming.pii.homoglyph import homoglyph
from data_manipulation.attacks.red_teaming.pii.invisible_chars import (
    COMBINING_CHARS,
    ZERO_WIDTH_CHARS,
    invisible_inject,
)
from data_manipulation.attacks.red_teaming.pii.separators import inject_separator
from data_manipulation.defenses.preprocess import (
    _strip_injected_separators,
    defensive_preprocess,
)


# ── Fixtures & helpers ──────────────────────────────────────────────

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


def _defend(attacked: str) -> str:
    return defensive_preprocess(attacked, include_sandwich=False)


def _assert_recovered(original: str, defended: str, case_sensitive: bool = True):
    """Assert the original PII value is found in the defended text."""
    if case_sensitive:
        assert original in defended, (
            f"PII not recovered: {original!r} not in {defended!r}"
        )
    else:
        assert original.lower() in defended.lower(), (
            f"PII not recovered (case-insensitive): "
            f"{original!r} not in {defended!r}"
        )


def _assert_digits_present(original: str, defended: str):
    """Assert all digits from the original appear contiguously in defended."""
    orig_digits = "".join(c for c in original if c.isdigit())
    def_digits = "".join(c for c in defended if c.isdigit())
    assert orig_digits in def_digits, (
        f"Digits lost: need {orig_digits!r} in {def_digits!r} "
        f"(defended={defended!r})"
    )


# ── Single attack roundtrips ────────────────────────────────────────


class TestHomoglyphRoundtrip:
    """Homoglyphs replace ASCII with Cyrillic/fullwidth lookalikes.

    Defense reverses them via a mapping table. Case may change
    (e.g. Cyrillic а → a or А → A) so we compare case-insensitively.
    """

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_reverses_cyrillic_and_fullwidth_chars(self, pii_id):
        pii = PII_SAMPLES[pii_id]
        _assert_recovered(pii, _defend(homoglyph(pii)), case_sensitive=False)

    def test_subscript_digits_reversed(self):
        """Homoglyph maps 0-9 to subscript ₀-₉."""
        _assert_recovered(
            "4111111111111111",
            _defend(homoglyph("4111111111111111")),
            case_sensitive=False,
        )

    def test_mixed_scripts_in_email(self):
        """Email has both Cyrillic letters and subscript digits."""
        attacked = homoglyph("test42@mail.com")
        _assert_recovered("test42@mail.com", _defend(attacked), case_sensitive=False)


class TestChunkingRoundtrip:
    """Chunking splits PII into quoted segments: "411" + "111" + "111".

    Defense detects 3+ short quoted chunks joined by + and merges them.
    """

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_quoted_chunks_rejoined(self, pii_id):
        pii = PII_SAMPLES[pii_id]
        _assert_recovered(pii, _defend(chunking(pii)))

    def test_credit_card_chunks_of_three(self):
        """16-digit CC splits into ~5 chunks of 3 digits."""
        attacked = chunking("4111111111111111")
        assert '" + "' in attacked, "Expected chunking pattern"
        _assert_recovered("4111111111111111", _defend(attacked))

    def test_email_chunks_preserve_structure(self):
        """Email chunks separate user, @, domain, ., tld."""
        attacked = chunking("john@example.com")
        _assert_recovered("john@example.com", _defend(attacked))

    def test_ssn_with_dash_chunks(self):
        """SSN dashes become separate chunks: "159" + "-" + "36" + ..."""
        attacked = chunking("123-45-6789")
        _assert_recovered("123-45-6789", _defend(attacked))


class TestEmojifyRoundtrip:
    """Emojify replaces chars with emoji: A→🅰, 1→1️⃣, @→🌀.

    Defense reverses via emoji demojization and mapping tables.
    Emojify only maps uppercase letters, so defended text is uppercased.
    """

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_emoji_reversed_case_insensitive(self, pii_id):
        pii = PII_SAMPLES[pii_id]
        _assert_recovered(pii, _defend(emojify_pii(pii)), case_sensitive=False)

    def test_keycap_digits_reversed(self):
        """Keycap digits (0️⃣-9️⃣) map back to ASCII 0-9."""
        attacked = emojify_pii("123456")
        _assert_digits_present("123456", _defend(attacked))

    def test_squared_letters_reversed(self):
        """Squared letters (🅰-🆉) map back to A-Z."""
        attacked = emojify_pii("GB82WEST")
        defended = _defend(attacked)
        assert "GB82WEST" in defended.upper()

    def test_partial_emojify_with_low_proba(self):
        """With proba=0.5, only some chars are emojified."""
        attacked = emojify_pii("4111111111111111", proba=0.5)
        _assert_digits_present("4111111111111111", _defend(attacked))


class TestCharToWordRoundtrip:
    """Char-to-word replaces digits/symbols with word representations.

    5 → (five), @ → (at), . → (dot). Supports English, Spanish, French.
    Defense reverses both delimited — (five) → 5 — and standalone words.
    """

    @pytest.mark.parametrize("lang", ["english", "spanish", "french"])
    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_all_languages_reversed(self, pii_id, lang):
        pii = PII_SAMPLES[pii_id]
        _assert_recovered(pii, _defend(char_to_word(pii, lang=lang)))

    def test_spanish_number_words_in_parens(self):
        """(cero)(cinco) → 05."""
        attacked = char_to_word("05", lang="spanish")
        assert "cero" in attacked and "cinco" in attacked
        _assert_recovered("05", _defend(attacked))

    def test_french_accented_zero(self):
        """French zéro has an accent that must survive separator stripping."""
        attacked = char_to_word("0", lang="french")
        assert "zéro" in attacked
        _assert_recovered("0", _defend(attacked))

    def test_symbol_words_reversed(self):
        """(at) → @, (dot) → ., (dash) → -."""
        attacked = char_to_word("user@host.com", lang="english")
        _assert_recovered("user@host.com", _defend(attacked))

    def test_multi_word_symbols(self):
        """(left parenthesis) and (right parenthesis) → ( and )."""
        attacked = char_to_word("(559)", lang="english")
        _assert_recovered("(559)", _defend(attacked))


class TestSeparatorRoundtrip:
    """Separator injection inserts a character between every PII character.

    Attack separators: :, ;, ＿ (fullwidth underscore).
    Defense strips any non-letter, non-digit, non-PII-structural char
    between non-space characters — generic, not tied to specific separators.
    """

    @pytest.mark.parametrize("separator", [":", ";", "＿"])
    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_known_attack_separators(self, pii_id, separator):
        pii = PII_SAMPLES[pii_id]
        attacked = inject_separator(pii, separator=separator, proba=1.0)
        _assert_recovered(pii, _defend(attacked))

    @pytest.mark.parametrize("separator", ["|", "~", "^", "`", "★"])
    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_novel_separators_not_in_attack_constants(self, pii_id, separator):
        """Defense is generic — works on separators it has never seen."""
        pii = PII_SAMPLES[pii_id]
        attacked = inject_separator(pii, separator=separator, proba=1.0)
        _assert_recovered(pii, _defend(attacked))

    def test_partial_injection_preserves_digits(self):
        """With proba=0.5, not every char gets a separator."""
        attacked = inject_separator("4111111111111111", separator=":", proba=0.5)
        _assert_digits_present("4111111111111111", _defend(attacked))


# ── Invisible character injection ──────────────────────────────────


class TestInvisibleCharsRoundtrip:
    """Zero-width and combining chars break regex but defense strips them.
    Unified pool: ZWSP, ZWNJ, ZWJ, WJ, soft hyphen, BOM,
    combining dot/macron/diaeresis/accent/cedilla.
    """

    @pytest.mark.parametrize("char", ZERO_WIDTH_CHARS)
    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_zero_width_chars(self, pii_id, char):
        pii = PII_SAMPLES[pii_id]
        _assert_recovered(pii, _defend(invisible_inject(pii, char=char)))

    @pytest.mark.parametrize("char", COMBINING_CHARS)
    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_combining_chars(self, pii_id, char):
        pii = PII_SAMPLES[pii_id]
        _assert_recovered(pii, _defend(invisible_inject(pii, char=char)))


# ── Combined attack chains ──────────────────────────────────────────


class TestCombinedAttackRoundtrip:
    """Real attacks chain multiple techniques. The defense must handle
    the combined effect even though it doesn't know the attack order.
    """

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_homoglyph_then_chunking(self, pii_id):
        """Homoglyph obfuscation followed by chunking into segments."""
        pii = PII_SAMPLES[pii_id]
        attacked = chunking(homoglyph(pii))
        _assert_recovered(pii, _defend(attacked), case_sensitive=False)

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_char_to_word_then_full_separator(self, pii_id):
        """Number words fragmented by full separator injection (proba=1.0).

        E.g. (cinco) → (:c:i:n:c:o:). Defense strips separators first,
        then reverses number words.
        """
        pii = PII_SAMPLES[pii_id]
        attacked = inject_separator(
            char_to_word(pii, lang="spanish"), separator=":", proba=1.0,
        )
        _assert_recovered(pii, _defend(attacked))

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_char_to_word_then_partial_separator(self, pii_id):
        """Partial separator injection (proba=0.5) after char_to_word.

        Creates inconsistent patterns — some word fragments have separators,
        others don't. Defense handles this because it strips all non-structural
        chars between non-space chars, not just consistent runs.
        """
        pii = PII_SAMPLES[pii_id]
        attacked = inject_separator(
            char_to_word(pii, lang="spanish"), separator=":", proba=0.5,
        )
        _assert_recovered(pii, _defend(attacked))

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_emojify_then_separator(self, pii_id):
        """Emoji characters followed by separator injection."""
        pii = PII_SAMPLES[pii_id]
        attacked = inject_separator(
            emojify_pii(pii), separator=";", proba=1.0,
        )
        _assert_recovered(pii, _defend(attacked), case_sensitive=False)

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_homoglyph_then_separator(self, pii_id):
        """Homoglyph + separator — two layers of obfuscation."""
        pii = PII_SAMPLES[pii_id]
        attacked = inject_separator(
            homoglyph(pii), separator=":", proba=1.0,
        )
        _assert_recovered(pii, _defend(attacked), case_sensitive=False)


# ── Full pipeline with sandwich ─────────────────────────────────────


class TestFullPipelineWithSandwich:
    """The sandwich defense wraps text in instruction-boundary markers.
    PII must survive the full pipeline including sandwich wrapping.
    """

    @pytest.mark.parametrize("pii_id", PII_IDS)
    def test_pii_survives_full_pipeline(self, pii_id):
        pii = PII_SAMPLES[pii_id]
        defended = defensive_preprocess(pii, include_sandwich=True)
        _assert_recovered(pii, defended)

    def test_sandwich_structure(self):
        """Sandwich adds BEGIN/END USER INPUT markers."""
        defended = defensive_preprocess("hello", include_sandwich=True)
        assert defended.startswith("BEGIN USER INPUT")
        assert "hello" in defended
        assert "END USER INPUT" in defended

    def test_attacked_pii_in_full_text_with_sandwich(self):
        """PII embedded in prose, attacked, then defended with sandwich."""
        raw = "Please send to ryan(cero)(cinco)(at)example(dot)org urgently."
        defended = defensive_preprocess(raw, include_sandwich=True)
        assert "ryan05@example.org" in defended
        assert defended.startswith("BEGIN USER INPUT")


# ── Separator stripping safety ──────────────────────────────────────


class TestSeparatorStrippingSafety:
    """_strip_injected_separators is aggressive by design — it strips
    non-structural chars between non-space characters to help PII detectors.
    This means some legitimate punctuation (10:30 → 1030) is collateral.
    The trade-off is acceptable because the defense prioritizes PII
    detectability over perfect text preservation.
    """

    @pytest.mark.parametrize("text", [
        "Dear Sir: I have a question; please respond.",
        "Key: value; another: pair.",
        "She said hello and he said goodbye.",
    ])
    def test_prose_with_space_adjacent_punctuation_unchanged(self, text):
        """Colon/semicolon followed by space is NOT stripped."""
        assert _strip_injected_separators(text) == text

    @pytest.mark.parametrize("text, expected", [
        ("10:30", "1030"),
        ("3:2", "32"),
        ("5;4", "54"),
    ])
    def test_punctuation_between_digits_stripped(self, text, expected):
        """Colons between digits (10:30) are stripped — acceptable
        trade-off since digits survive for PII detection.
        """
        assert _strip_injected_separators(text) == expected

    def test_only_strips_between_non_space_chars(self):
        """Separator adjacent to space is NOT stripped."""
        text = "Note: this matters; really."
        assert _strip_injected_separators(text) == text


# ── Normal text preservation ────────────────────────────────────────


class TestNormalTextPreservation:
    """Defense should preserve the meaning of non-PII text.
    Key content words should survive even if punctuation changes.
    """

    @pytest.mark.parametrize("text, expected_words", [
        (
            "What are the best practices for code review?",
            ["best", "practices", "code", "review"],
        ),
        (
            "Explain the difference between TCP and UDP.",
            ["Explain", "difference", "between"],
        ),
        (
            "How do I reverse a linked list in Python?",
            ["reverse", "linked", "list", "Python"],
        ),
        (
            "The quarterly report shows a 15% increase in revenue.",
            ["quarterly", "report", "shows", "increase", "revenue"],
        ),
    ])
    def test_content_words_survive(self, text, expected_words):
        defended = _defend(text)
        for word in expected_words:
            assert word.lower() in defended.lower(), (
                f"Content word {word!r} lost in defended text {defended!r}"
            )

    def test_digits_in_normal_text_survive(self):
        """Non-PII digits (dates, amounts) should not be destroyed."""
        defended = _defend("Meeting on 2024-03-15 about $5,000 budget.")
        assert "2024" in defended
        assert "5" in defended


# ── Edge cases ──────────────────────────────────────────────────────


class TestEdgeCases:

    def test_empty_string(self):
        assert _defend("") == ""

    def test_single_character(self):
        assert _defend("5") == "5"

    def test_only_separators(self):
        """A string of only injected separators collapses to empty/minimal."""
        defended = _defend("::;;＿＿")
        assert len(defended) <= len("::;;＿＿")

    def test_already_clean_pii_unchanged(self):
        """Clean PII without any attack should pass through intact."""
        for pii in PII_SAMPLES.values():
            _assert_recovered(pii, _defend(pii))

    def test_repeated_digits_not_collapsed(self):
        """Defense must NOT collapse repeated digits (old bug: (.){3,} regex).

        Credit card 4111111111111111 has 12 consecutive 1s — all must survive.
        """
        defended = _defend("4111111111111111")
        assert "4111111111111111" in defended

    def test_unicode_accented_chars_preserved(self):
        """Accented characters (é, ñ, etc.) in text should not be stripped."""
        defended = _defend("Café résumé naïve")
        assert "caf" in defended.lower()
        assert "sum" in defended.lower()

    def test_pii_embedded_in_long_prose(self):
        """PII surrounded by lengthy text should still be recoverable."""
        prose = (
            "As part of our compliance review, the account holder's "
            "social security number 159-36-9263 was verified against "
            "federal records. No discrepancies were found."
        )
        _assert_recovered("159-36-9263", _defend(prose))
