import pytest

from detectors.validators import (
    _is_known_non_pii,
    _is_valid_ssn,
    validate_pii_spans,
)


class TestIsKnownNonPii:

    @pytest.mark.parametrize("value", [
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "da39a3ee5e6b4b0d3255bfef95601890afd80709",
        "d41d8cd98f00b204e9800998ecf8427e",
        "550e8400-e29b-41d4-a716-446655440000",
        "00:1A:2B:3C:4D:5E",
        "a3f2b8c91d4e7f0062ab5cd8",
    ])
    def test_rejects_non_pii(self, value):
        assert _is_known_non_pii(value) is True

    @pytest.mark.parametrize("value", [
        "123-45-6789",
        "4111111111111111",
        "+1-555-123-4567",
        "ryan05@example.org",
    ])
    def test_allows_pii(self, value):
        assert _is_known_non_pii(value) is False

    def test_iban_looks_like_hex(self):
        # IBANs with hex-only country codes (DE, AB, etc.) match hex pattern.
        # validate_pii_spans handles this via the Mod-97 type validator.
        assert _is_known_non_pii("DE44500105175407324931") is True


class TestIsValidSsn:

    @pytest.mark.parametrize("ssn,expected", [
        ("123-45-6789", True),
        ("001-01-0001", True),
        ("000-45-6789", False),
        ("666-45-6789", False),
        ("900-45-6789", False),
        ("999-45-6789", False),
        ("123-00-6789", False),
        ("123-45-0000", False),
        ("abc-de-fghi", False),
        ("12345", False),
    ])
    def test_ssn_validation(self, ssn, expected):
        assert _is_valid_ssn(ssn) == expected, f"SSN {ssn} expected {expected}"


class TestValidatePiiSpans:

    def test_filters_hash_detected_as_ssn(self):
        spans = [{"value": "d41d8cd98f00b204e9800998ecf8427e", "type": "ssn"}]
        assert validate_pii_spans(spans) == []

    def test_filters_uuid_detected_as_ssn(self):
        spans = [{
            "value": "550e8400-e29b-41d4-a716-446655440000",
            "type": "ssn",
        }]
        assert validate_pii_spans(spans) == []

    def test_filters_mac_detected_as_ssn(self):
        spans = [{"value": "00:1A:2B:3C:4D:5E", "type": "ssn"}]
        assert validate_pii_spans(spans) == []

    def test_filters_invalid_credit_card(self):
        spans = [{"value": "1234-5678-9012-3456", "type": "credit_card_number"}]
        assert validate_pii_spans(spans) == []

    def test_keeps_valid_credit_card(self):
        spans = [{"value": "4111111111111111", "type": "credit_card_number"}]
        assert len(validate_pii_spans(spans)) == 1

    def test_filters_invalid_ssn_area_000(self):
        spans = [{"value": "000-12-3456", "type": "ssn"}]
        assert validate_pii_spans(spans) == []

    def test_keeps_valid_ssn(self):
        spans = [{"value": "123-45-6789", "type": "ssn", "start": 10, "end": 21}]
        result = validate_pii_spans(spans)
        assert len(result) == 1
        assert result[0] == spans[0]

    def test_keeps_valid_iban(self):
        spans = [{"value": "DE44500105175407324931", "type": "iban"}]
        assert len(validate_pii_spans(spans)) == 1

    def test_filters_invalid_iban(self):
        spans = [{"value": "DE00000000000000000000", "type": "iban"}]
        assert validate_pii_spans(spans) == []

    def test_keeps_valid_email(self):
        spans = [{"value": "user@example.com", "type": "email"}]
        assert len(validate_pii_spans(spans)) == 1

    def test_keeps_valid_phone(self):
        spans = [{"value": "+1-555-123-4567", "type": "phone_number"}]
        assert len(validate_pii_spans(spans)) == 1

    def test_preserves_span_fields(self):
        spans = [{
            "value": "123-45-6789", "type": "ssn",
            "start": 10, "end": 21, "score": 0.9,
        }]
        result = validate_pii_spans(spans)
        assert result[0] == spans[0]

    def test_empty_input(self):
        assert validate_pii_spans([]) == []

    def test_none_value_span_kept(self):
        spans = [{"value": None, "type": "pii", "start": 0, "end": 5}]
        assert len(validate_pii_spans(spans)) == 1

    def test_mixed_valid_and_invalid(self):
        spans = [
            {"value": "d41d8cd98f00b204e9800998ecf8427e", "type": "ssn"},
            {"value": "123-45-6789", "type": "ssn"},
            {"value": "1234-5678-9012-3456", "type": "credit_card_number"},
        ]
        result = validate_pii_spans(spans)
        assert len(result) == 1
        assert result[0]["value"] == "123-45-6789"
