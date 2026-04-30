import pytest

from detectors.hard_negatives import (
    filter_hard_negative_spans,
    is_hard_negative_input,
    is_hard_negative_value,
)


class TestIsHardNegativeValue:

    @pytest.mark.parametrize("value,expected", [
        ("550e8400-e29b-41d4-a716-446655440000", "uuid"),
        ("00:1A:2B:3C:4D:5E", "mac"),
        ("00-1A-2B-3C-4D-5E", "mac"),
        ("2001:0db8:85a3:0000:0000:8a2e:0370:7334", "ipv6_full"),
        ("2606:4700::1111", "ipv6_compressed"),
        ("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256"),
        ("da39a3ee5e6b4b0d3255bfef95601890afd80709", "sha1"),
        ("d41d8cd98f00b204e9800998ecf8427e", "md5"),
        ("#aabbcc", "hex_color"),
        ('W/"7a8b12e34b-4d8c"', "etag"),
        ("SN123456789", "serial"),
        ("INV-2023456789", None),
        ("INV-2023-4567-89", "invoice_dashed"),
        ("4567-8904-3210", "invoice_dashed"),
    ])
    def test_recognises_known_formats(self, value, expected):
        assert is_hard_negative_value(value) == expected

    @pytest.mark.parametrize("value", [
        "4111111111111111",
        "GB29NWBK60161331926819",
        "123-45-6789",
        "+1 (555) 123-4567",
        "user@example.com",
        "",
        "065488",        # short pure-digit chunk; must not match hex_color
        "12345678",      # 8 pure digits; must not match hex_color
        "1234567890123456789012345678901234567890",  # 40 pure digits; not sha1
        "aabbcc",        # 6 hex chars without '#' prefix; not hex_color
    ])
    def test_does_not_match_real_pii_or_empty(self, value):
        assert is_hard_negative_value(value) is None


class TestIsHardNegativeInput:

    @pytest.mark.parametrize("text,expected", [
        ("My MAC is 00:1A:2B:3C:4D:5E for the router", "mac"),
        ("UUID 550e8400-e29b-41d4-a716-446655440000 in the log", "uuid"),
        ("IPv6 2001:0db8:85a3::8a2e:0370:7334 routes here", "ipv6_compressed"),
        ("hash: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256"),
        ('ETag W/"7a8b12e34b-4d8c" returned', "etag"),
        ("color #ff5500 selected", "hex_color"),
    ])
    def test_finds_pattern_in_context(self, text, expected):
        assert is_hard_negative_input(text) == expected

    @pytest.mark.parametrize("text", [
        "Hi, my credit card is 4111111111111111 thanks",
        "Phone is 7:6:8: :4:7:8: :1:7:3",  # chunked-with-colons attack: digits only
        "no special tokens here, just regular text",
    ])
    def test_skips_clean_text(self, text):
        assert is_hard_negative_input(text) is None


class TestFilterHardNegativeSpans:

    def test_drops_hard_negative_span_keeps_real_pii(self):
        spans = [
            {"value": "550e8400-e29b-41d4-a716-446655440000", "type": "uuid"},
            {"value": "4111111111111111", "type": "credit_card_number"},
        ]
        out = filter_hard_negative_spans(spans)
        assert len(out) == 1
        assert out[0]["value"] == "4111111111111111"

    def test_empty_input_returns_empty(self):
        assert filter_hard_negative_spans([]) == []

    def test_missing_value_field_kept(self):
        spans = [{"type": "phone_number"}]
        assert filter_hard_negative_spans(spans) == spans
