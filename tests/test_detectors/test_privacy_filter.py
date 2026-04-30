"""Unit tests for the OpenAI privacy-filter detector wrapper."""
import os

import pytest

from detectors.privacy_filter.detector import (
    PRIVACY_FILTER_MODELS,
    _classify_account_number,
    _merge_adjacent,
    _to_project_spans,
    privacy_filter_pii_detector,
    privacy_filter_pii_detector_batch,
)

# Integration tests load the real ~hundreds-of-MB HF model. Off by default;
# enable with `RUN_HF_TESTS=1 uv run pytest tests/test_detectors/test_privacy_filter.py`.
_run_hf = pytest.mark.skipif(
    os.environ.get("RUN_HF_TESTS") != "1",
    reason="set RUN_HF_TESTS=1 to run real-model integration tests",
)


class TestModelMapping:

    def test_has_openai_privacy_filter(self):
        assert "openai-privacy-filter" in PRIVACY_FILTER_MODELS
        assert PRIVACY_FILTER_MODELS["openai-privacy-filter"] == "openai/privacy-filter"


class TestClassifyAccountNumber:

    @pytest.mark.parametrize("value, expected", [
        ("4111-1111-1111-1111", "credit_card_number"),  # Luhn-valid
        ("4111 1111 1111 1111", "credit_card_number"),  # spaces tolerated
        ("GB82 WEST 1234 5698 7654 32", "iban"),  # IBAN with spaces
        ("GB82WEST12345698765432", "iban"),
        ("123-45-6789", "ssn"),
        ("123456789", "ssn"),  # no dashes
    ])
    def test_valid_values_classified(self, value, expected):
        assert _classify_account_number(value) == expected

    @pytest.mark.parametrize("value", [
        "1234567890123456",  # 16 digits but fails Luhn
        "999-99-9999",       # SSN area 999 invalid
        "666-12-3456",       # SSN area 666 invalid
        "000-12-3456",       # SSN area 000 invalid
        "123-00-3456",       # SSN group 00 invalid
        "123-45-0000",       # SSN serial 0000 invalid
        "GB99WEST00000000000000",  # IBAN with bad checksum
        "not a number",
        "",
    ])
    def test_invalid_values_dropped(self, value):
        assert _classify_account_number(value) is None

    def test_strips_whitespace(self):
        assert _classify_account_number("  4111-1111-1111-1111  ") == "credit_card_number"


class TestMergeAdjacent:

    def test_merges_adjacent_same_category(self):
        spans = [
            {"entity_group": "account_number", "start": 18, "end": 36, "score": 0.999, "word": "x"},
            {"entity_group": "account_number", "start": 36, "end": 37, "score": 1.000, "word": "y"},
        ]
        out = _merge_adjacent(spans)
        assert len(out) == 1
        assert out[0]["start"] == 18 and out[0]["end"] == 37
        assert out[0]["score"] == pytest.approx(0.999)  # min of the two

    def test_does_not_merge_with_gap(self):
        spans = [
            {"entity_group": "account_number", "start": 21, "end": 24, "score": 0.8, "word": "a"},
            {"entity_group": "account_number", "start": 25, "end": 28, "score": 0.6, "word": "b"},
        ]
        out = _merge_adjacent(spans)
        assert len(out) == 2

    def test_does_not_merge_different_categories(self):
        spans = [
            {"entity_group": "private_email", "start": 0, "end": 5, "score": 0.9, "word": "a"},
            {"entity_group": "private_phone", "start": 5, "end": 10, "score": 0.9, "word": "b"},
        ]
        out = _merge_adjacent(spans)
        assert len(out) == 2

    def test_empty_input(self):
        assert _merge_adjacent([]) == []


class TestToProjectSpans:

    def test_drops_address_person_url_date_secret(self):
        text = "irrelevant"
        spans = [
            {"entity_group": "private_address", "start": 0, "end": 5, "score": 1.0},
            {"entity_group": "private_person", "start": 0, "end": 5, "score": 1.0},
            {"entity_group": "private_url", "start": 0, "end": 5, "score": 1.0},
            {"entity_group": "private_date", "start": 0, "end": 5, "score": 1.0},
            {"entity_group": "secret", "start": 0, "end": 5, "score": 1.0},
        ]
        assert _to_project_spans(spans, text) == []

    def test_keeps_email_and_phone(self):
        text = "Email me at john@example.com or call 415-555-0142 today."
        spans = [
            {"entity_group": "private_email", "start": 12, "end": 28, "score": 0.99},
            {"entity_group": "private_phone", "start": 37, "end": 49, "score": 0.99},
        ]
        out = _to_project_spans(spans, text)
        assert out == [
            {"value": "john@example.com", "start": 12, "end": 28, "type": "email"},
            {"value": "415-555-0142", "start": 37, "end": 49, "type": "phone_number"},
        ]

    def test_account_number_validates_credit_card(self):
        text = "card 4111-1111-1111-1111 here"
        spans = [
            {"entity_group": "account_number", "start": 5, "end": 24, "score": 1.0},
        ]
        out = _to_project_spans(spans, text)
        assert out == [{
            "value": "4111-1111-1111-1111",
            "start": 5, "end": 24, "type": "credit_card_number",
        }]

    def test_account_number_validates_ssn(self):
        text = "ssn 123-45-6789 ok"
        spans = [
            {"entity_group": "account_number", "start": 4, "end": 15, "score": 1.0},
        ]
        out = _to_project_spans(spans, text)
        assert out[0]["type"] == "ssn"
        assert out[0]["value"] == "123-45-6789"

    def test_account_number_unvalidated_keeps_span_as_account_number(self):
        # 16 digits but fails Luhn — span is preserved with the model's
        # coarse class as the type rather than being dropped.
        text = "tracking 1234567890123456 was delivered"
        spans = [
            {"entity_group": "account_number", "start": 9, "end": 25, "score": 1.0},
        ]
        out = _to_project_spans(spans, text)
        assert out == [{
            "value": "1234567890123456",
            "start": 9, "end": 25, "type": "account_number",
        }]

    def test_account_number_non_ascii_keeps_span_as_account_number(self):
        # Unicode subscripts (from fuzzy attacks) bypass int()-based
        # validators; the span survives with the coarse class instead.
        bad = "1234₀₁₂₃567890123456"
        text = f"id {bad} ok"
        spans = [
            {"entity_group": "account_number",
             "start": 3, "end": 3 + len(bad), "score": 1.0},
        ]
        out = _to_project_spans(spans, text)
        assert out[0]["type"] == "account_number"
        assert out[0]["value"] == bad


class TestPrivacyFilterPiiDetector:

    def test_returns_filtered_spans(self, mocker):
        mock_pipe = mocker.MagicMock()
        mock_pipe.return_value = [
            {"entity_group": "private_email", "start": 12, "end": 28, "score": 0.99, "word": "x"},
            {"entity_group": "private_address", "start": 30, "end": 40, "score": 0.99, "word": "y"},
        ]
        mocker.patch(
            "detectors.privacy_filter.detector._get_pipeline",
            return_value=mock_pipe,
        )
        text = "Email me at john@example.com 221B Baker St"
        out = privacy_filter_pii_detector(text)
        # Address dropped, email kept and value sliced from `text`.
        assert out == [{
            "value": "john@example.com",
            "start": 12, "end": 28, "type": "email",
        }]

    def test_empty_text_short_circuits(self, mocker):
        mock_pipe = mocker.MagicMock()
        mocker.patch(
            "detectors.privacy_filter.detector._get_pipeline",
            return_value=mock_pipe,
        )
        assert privacy_filter_pii_detector("") == []
        mock_pipe.assert_not_called()

    def test_merges_split_credit_card(self, mocker):
        # Pipeline emits a CC as two adjacent spans — merger fuses them
        # so Luhn validation sees the full 16 digits.
        mock_pipe = mocker.MagicMock()
        # text[11:29] = "4111-1111-1111-111", text[29:30] = "1"
        mock_pipe.return_value = [
            {"entity_group": "account_number", "start": 11, "end": 29, "score": 0.99, "word": "x"},
            {"entity_group": "account_number", "start": 29, "end": 30, "score": 1.00, "word": "y"},
        ]
        mocker.patch(
            "detectors.privacy_filter.detector._get_pipeline",
            return_value=mock_pipe,
        )
        text = "My card is 4111-1111-1111-1111, please charge."
        out = privacy_filter_pii_detector(text)
        assert len(out) == 1
        assert out[0]["type"] == "credit_card_number"
        assert out[0]["value"] == "4111-1111-1111-1111"


class TestPrivacyFilterPiiDetectorBatch:

    def test_preserves_input_order_and_handles_none(self, mocker):
        mock_pipe = mocker.MagicMock()
        mock_pipe.return_value = [
            [{"entity_group": "private_email", "start": 0, "end": 16, "score": 0.99, "word": "x"}],
            [],
        ]
        mocker.patch(
            "detectors.privacy_filter.detector._get_pipeline",
            return_value=mock_pipe,
        )
        out = privacy_filter_pii_detector_batch([
            "john@example.com sent the report",
            None,
            "hello world",
        ])
        assert len(out) == 3
        # First text: email kept; second: None -> []; third: no spans.
        assert len(out[0]) == 1 and out[0][0]["type"] == "email"
        assert out[1] == []
        assert out[2] == []

    def test_all_none_returns_empty_lists(self, mocker):
        mock_pipe = mocker.MagicMock()
        mocker.patch(
            "detectors.privacy_filter.detector._get_pipeline",
            return_value=mock_pipe,
        )
        out = privacy_filter_pii_detector_batch([None, None])
        assert out == [[], []]
        mock_pipe.assert_not_called()


@_run_hf
class TestPrivacyFilterRealModel:
    """End-to-end integration with the real openai/privacy-filter model.

    First run downloads model weights from HF Hub (~hundreds of MB) and
    is slow; subsequent runs hit the local cache.
    """

    def test_detects_email(self):
        text = "Please contact me at john.doe@example.com about the report."
        spans = privacy_filter_pii_detector(text)
        emails = [s for s in spans if s["type"] == "email"]
        assert emails, f"expected an email span, got {spans}"
        assert "john.doe@example.com" in emails[0]["value"]
        s = emails[0]
        assert text[s["start"]:s["end"]] == s["value"]

    def test_detects_credit_card_via_account_number(self):
        # Luhn-valid Visa test number. The phrasing matters — the model
        # only fires on strong cues like "credit card number is …";
        # casual phrasings ("charge my card …") are missed.
        text = "My credit card number is 4111 1111 1111 1111."
        spans = privacy_filter_pii_detector(text)
        ccs = [s for s in spans if s["type"] == "credit_card_number"]
        assert ccs, f"expected a credit_card_number span, got {spans}"
        # Merged span should cover all 16 digits and pass Luhn cleanup.
        assert ccs[0]["value"].replace("-", "").replace(" ", "") == "4111111111111111"

    def test_drops_non_pii_text(self):
        text = "The weather forecast looks clear for the weekend."
        spans = privacy_filter_pii_detector(text)
        assert spans == []

    def test_batch_preserves_order_and_filters(self):
        texts = [
            "Email me at jane@example.com please.",
            "",
            "Just a neutral sentence with no PII.",
        ]
        out = privacy_filter_pii_detector_batch(texts)
        assert len(out) == 3
        assert any(s["type"] == "email" for s in out[0])
        assert out[1] == []
        assert out[2] == []
