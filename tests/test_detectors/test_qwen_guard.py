import pytest
import torch

from detectors.guards.qwen_guard import (
    _build_spans_from_pii_tokens,
    _find_content_token_range,
    _parse_gen_output,
    classify_pii_gen,
    classify_pii_gen_batch,
    detect_pii_stream,
    qwen_guard_gen_pii_detector,
    qwen_guard_stream_pii_detector,
)


PII_SPAN = {"value": None, "start": None, "end": None, "type": "pii"}


# ── _parse_gen_output ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content, expected",
    [
        ("Safety: Safe\nCategories: None", False),
        ("Safety: Unsafe\nCategories: PII", True),
        ("Safety: Controversial\nCategories: PII", True),
        ("Safety: Unsafe\nCategories: Violent", False),
        ("Safety: Unsafe\nCategories: Violent, PII", True),
        ("Safety: Safe\nCategories: PII", False),
        ("", False),
        ("random text", False),
    ],
)
def test_parse_gen_output(content, expected):
    assert _parse_gen_output(content) is expected


# ── _find_content_token_range ──────────────────────────────────────────


def test_find_content_token_range():
    mock_tokenizer = type("T", (), {
        "convert_tokens_to_ids": lambda self, t: {
            "<|im_start|>": 100, "user": 200, "<|im_end|>": 300,
        }[t],
    })()
    # <|im_start|> user \n content1 content2 <|im_end|>
    token_ids = [100, 200, 50, 10, 20, 300]
    start, end = _find_content_token_range(token_ids, mock_tokenizer)
    assert start == 3
    assert end == 5


# ── _build_spans_from_pii_tokens ──────────────────────────────────────


class TestBuildSpansFromPiiTokens:

    @pytest.fixture()
    def mock_tokenizer(self, mocker):
        tokenizer = mocker.MagicMock()
        return tokenizer

    def test_single_pii_span(self, mock_tokenizer):
        text = "my ssn is 123-45-6789"
        # Simulate tokens: ["my", " ssn", " is", " 123", "-45", "-6789"]
        mock_tokenizer.return_value = {
            "input_ids": [10, 20, 30, 40, 50, 60],
            "offset_mapping": [(0, 2), (2, 6), (6, 9), (9, 13), (13, 17), (17, 21)],
        }
        pii_flags = [False, False, False, True, True, True]
        token_ids = [99, 98, 97, 10, 20, 30, 40, 50, 60, 300]
        content_start = 3

        spans = _build_spans_from_pii_tokens(
            pii_flags, token_ids, content_start, mock_tokenizer, text,
        )
        assert len(spans) == 1
        assert spans[0]["value"] == " 123-45-6789"
        assert spans[0]["start"] == 9
        assert spans[0]["end"] == 21
        assert spans[0]["type"] == "pii"

    def test_no_pii_flags(self, mock_tokenizer):
        mock_tokenizer.return_value = {
            "input_ids": [10, 20],
            "offset_mapping": [(0, 2), (2, 5)],
        }
        spans = _build_spans_from_pii_tokens(
            [False, False], [99, 98, 10, 20], 2, mock_tokenizer, "hello",
        )
        assert spans == []

    def test_multiple_pii_spans(self, mock_tokenizer):
        text = "ssn 123 and email a@b.c"
        mock_tokenizer.return_value = {
            "input_ids": [10, 20, 30, 40, 50],
            "offset_mapping": [(0, 3), (3, 7), (7, 12), (12, 18), (18, 23)],
        }
        pii_flags = [False, True, False, False, True]
        token_ids = [99, 10, 20, 30, 40, 50]
        content_start = 1

        spans = _build_spans_from_pii_tokens(
            pii_flags, token_ids, content_start, mock_tokenizer, text,
        )
        assert len(spans) == 2
        assert spans[0]["value"] == " 123"
        assert spans[1]["value"] == "a@b.c"

    def test_alignment_mismatch_falls_back(self, mock_tokenizer):
        mock_tokenizer.return_value = {
            "input_ids": [10, 20],
            "offset_mapping": [(0, 2), (2, 5)],
        }
        # Mismatched: template has [99, 98] but raw text has [10, 20]
        spans = _build_spans_from_pii_tokens(
            [True, False], [99, 98], 0, mock_tokenizer, "hello",
        )
        assert spans == [PII_SPAN]

    def test_alignment_mismatch_no_pii(self, mock_tokenizer):
        mock_tokenizer.return_value = {
            "input_ids": [10, 20],
            "offset_mapping": [(0, 2), (2, 5)],
        }
        spans = _build_spans_from_pii_tokens(
            [False, False], [99, 98], 0, mock_tokenizer, "hello",
        )
        assert spans == []


# ── classify_pii_gen (mocked) ─────────────────────────────────────────


@pytest.fixture()
def mock_qwen_gen(mocker):
    mock_tokenizer = mocker.MagicMock()
    mock_model = mocker.MagicMock()
    mock_model.device = torch.device("cpu")

    mocker.patch(
        "detectors.guards.qwen_guard._get_gen_model",
        return_value=(mock_tokenizer, mock_model),
    )
    return mock_tokenizer, mock_model


def _setup_gen_generate(mock_tokenizer, mock_model, decoded_output: str):
    input_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer.apply_chat_template.return_value = input_ids
    mock_tokenizer.pad_token_id = 0
    type(mock_tokenizer).input_ids = input_ids

    output_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.generate.return_value = output_tensor
    mock_tokenizer.decode.return_value = decoded_output


class TestClassifyPiiGen:

    def test_safe(self, mock_qwen_gen):
        tokenizer, model = mock_qwen_gen
        _setup_gen_generate(tokenizer, model, "Safety: Safe\nCategories: None")
        assert classify_pii_gen("hello", "qwen-guard-gen-4b") is False

    def test_unsafe_pii(self, mock_qwen_gen):
        tokenizer, model = mock_qwen_gen
        _setup_gen_generate(tokenizer, model, "Safety: Unsafe\nCategories: PII")
        assert classify_pii_gen("my ssn", "qwen-guard-gen-4b") is True

    def test_unsafe_non_pii(self, mock_qwen_gen):
        tokenizer, model = mock_qwen_gen
        _setup_gen_generate(tokenizer, model, "Safety: Unsafe\nCategories: Violent")
        assert classify_pii_gen("violence", "qwen-guard-gen-4b") is False


# ── classify_pii_gen_batch (mocked) ───────────────────────────────────


class TestClassifyPiiGenBatch:

    def test_batch_mixed(self, mock_qwen_gen):
        tokenizer, model = mock_qwen_gen
        input_ids = torch.tensor([[1, 2, 3]])
        tokenizer.apply_chat_template.return_value = input_ids
        tokenizer.pad_token_id = 0

        output_tensor = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]])
        model.generate.return_value = output_tensor
        tokenizer.decode.side_effect = [
            "Safety: Unsafe\nCategories: PII",
            "Safety: Safe\nCategories: None",
        ]

        results = classify_pii_gen_batch(["pii text", "safe text"], "qwen-guard-gen-4b")
        assert results == [True, False]

    def test_batch_all_safe(self, mock_qwen_gen):
        tokenizer, model = mock_qwen_gen
        input_ids = torch.tensor([[1, 2, 3]])
        tokenizer.apply_chat_template.return_value = input_ids
        tokenizer.pad_token_id = 0

        output_tensor = torch.tensor([[1, 2, 3, 4, 5]])
        model.generate.return_value = output_tensor
        tokenizer.decode.return_value = "Safety: Safe\nCategories: None"

        results = classify_pii_gen_batch(["safe"], "qwen-guard-gen-4b")
        assert results == [False]


# ── detect_pii_stream (mocked) ────────────────────────────────────────


@pytest.fixture()
def mock_qwen_stream(mocker):
    mock_tokenizer = mocker.MagicMock()
    mock_model = mocker.MagicMock()

    mocker.patch(
        "detectors.guards.qwen_guard._get_stream_model",
        return_value=(mock_tokenizer, mock_model),
    )
    return mock_tokenizer, mock_model


class TestDetectPiiStream:

    def test_no_pii_returns_empty(self, mock_qwen_stream):
        tokenizer, model = mock_qwen_stream

        # Token IDs: <|im_start|> user \n hello <|im_end|>
        token_ids = torch.tensor([100, 200, 50, 10, 300])
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.return_value = type("E", (), {"input_ids": token_ids.unsqueeze(0)})()
        tokenizer.convert_tokens_to_ids.side_effect = lambda t: {
            "<|im_start|>": 100, "user": 200, "<|im_end|>": 300,
        }[t]

        # All safe
        model.stream_moderate_from_ids.return_value = (
            {"risk_level": ["Safe"], "category": ["None"]}, "state",
        )

        result = detect_pii_stream("hello", "qwen-guard-stream-4b")
        assert result == []
        model.close_stream.assert_called_once()

    def test_pii_detected_builds_spans(self, mock_qwen_stream, mocker):
        tokenizer, model = mock_qwen_stream

        # Token IDs: <|im_start|> user \n my ssn <|im_end|>
        token_ids = torch.tensor([100, 200, 50, 10, 20, 300])
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.return_value = type("E", (), {"input_ids": token_ids.unsqueeze(0)})()
        tokenizer.convert_tokens_to_ids.side_effect = lambda t: {
            "<|im_start|>": 100, "user": 200, "<|im_end|>": 300,
        }[t]

        # First content token safe, second unsafe PII
        call_count = [0]

        def stream_moderate(tok, role, stream_state):
            call_count[0] += 1
            if call_count[0] <= 1:
                # Template tokens or first content token
                return {"risk_level": ["Safe"], "category": ["None"]}, "state"
            if call_count[0] == 2:
                return {"risk_level": ["Safe"], "category": ["None"]}, "state"
            return {"risk_level": ["Unsafe"], "category": ["PII"]}, "state"

        model.stream_moderate_from_ids.side_effect = stream_moderate

        # Mock _build_spans_from_pii_tokens
        mock_build = mocker.patch(
            "detectors.guards.qwen_guard._build_spans_from_pii_tokens",
            return_value=[{"value": "ssn", "start": 3, "end": 6, "type": "pii"}],
        )

        result = detect_pii_stream("my ssn", "qwen-guard-stream-4b")
        assert len(result) == 1
        assert result[0]["value"] == "ssn"
        mock_build.assert_called_once()


# ── Public API ────────────────────────────────────────────────────────


class TestQwenGuardGenPiiDetector:

    def test_none_returns_empty(self, mock_qwen_gen):
        assert qwen_guard_gen_pii_detector(None) == []

    def test_safe_returns_empty(self, mock_qwen_gen):
        tokenizer, model = mock_qwen_gen
        _setup_gen_generate(tokenizer, model, "Safety: Safe\nCategories: None")
        assert qwen_guard_gen_pii_detector("hello") == []

    def test_pii_returns_span(self, mock_qwen_gen):
        tokenizer, model = mock_qwen_gen
        _setup_gen_generate(tokenizer, model, "Safety: Unsafe\nCategories: PII")
        assert qwen_guard_gen_pii_detector("my ssn") == [PII_SPAN]


class TestQwenGuardStreamPiiDetector:

    def test_none_returns_empty(self, mock_qwen_stream):
        assert qwen_guard_stream_pii_detector(None) == []

    def test_delegates_to_detect_pii_stream(self, mocker):
        mock_detect = mocker.patch(
            "detectors.guards.qwen_guard.detect_pii_stream",
            return_value=[{"value": "123", "start": 0, "end": 3, "type": "pii"}],
        )
        result = qwen_guard_stream_pii_detector("my ssn 123", "qwen-guard-stream-0.6b")
        assert len(result) == 1
        mock_detect.assert_called_once_with("my ssn 123", "qwen-guard-stream-0.6b")


# ── Span integrity tests ────────────────────────────────────────────────


class TestBuildSpansIntegrity:
    """Validate that built spans have correct char offsets matching the text."""

    @pytest.fixture()
    def mock_tokenizer(self, mocker):
        return mocker.MagicMock()

    def test_span_value_matches_text_slice(self, mock_tokenizer):
        text = "call me at 555-123-4567 please"
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5, 6],
            "offset_mapping": [
                (0, 4), (4, 7), (7, 11), (11, 15), (15, 23), (23, 30),
            ],
        }
        # Tokens 3-4 are PII (the phone number region)
        pii_flags = [False, False, False, True, True, False]
        token_ids = [99, 98, 97, 1, 2, 3, 4, 5, 6, 300]
        content_start = 3

        spans = _build_spans_from_pii_tokens(
            pii_flags, token_ids, content_start, mock_tokenizer, text,
        )
        assert len(spans) == 1
        span = spans[0]
        # The span value must equal the text slice at [start:end]
        assert span["value"] == text[span["start"]:span["end"]]
        assert span["start"] == 11
        assert span["end"] == 23

    def test_multiple_spans_no_overlap(self, mock_tokenizer):
        text = "ssn 123-45-6789 email a@b.com"
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4],
            "offset_mapping": [(0, 4), (4, 16), (16, 22), (22, 29)],
        }
        pii_flags = [False, True, False, True]
        token_ids = [99, 1, 2, 3, 4]
        content_start = 1

        spans = _build_spans_from_pii_tokens(
            pii_flags, token_ids, content_start, mock_tokenizer, text,
        )
        assert len(spans) == 2
        # Verify no overlap
        assert spans[0]["end"] <= spans[1]["start"]
        # Verify values match text slices
        for span in spans:
            assert span["value"] == text[span["start"]:span["end"]]

    def test_all_tokens_pii_single_span(self, mock_tokenizer):
        text = "4111-1111-1111-1111"
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "offset_mapping": [(0, 4), (4, 14), (14, 19)],
        }
        pii_flags = [True, True, True]
        token_ids = [99, 1, 2, 3]
        content_start = 1

        spans = _build_spans_from_pii_tokens(
            pii_flags, token_ids, content_start, mock_tokenizer, text,
        )
        assert len(spans) == 1
        assert spans[0]["value"] == text
        assert spans[0]["start"] == 0
        assert spans[0]["end"] == len(text)

    def test_consecutive_pii_merged(self, mock_tokenizer):
        text = "my card 4111222233334444"
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "offset_mapping": [(0, 2), (2, 7), (7, 11), (11, 19), (19, 23)],
        }
        # Tokens 2-4 are consecutive PII
        pii_flags = [False, False, True, True, True]
        token_ids = [99, 1, 2, 3, 4, 5]
        content_start = 1

        spans = _build_spans_from_pii_tokens(
            pii_flags, token_ids, content_start, mock_tokenizer, text,
        )
        # Consecutive PII tokens should produce a single merged span
        assert len(spans) == 1
        assert spans[0]["value"] == text[7:23]

    def test_span_type_always_pii(self, mock_tokenizer):
        text = "test"
        mock_tokenizer.return_value = {
            "input_ids": [1],
            "offset_mapping": [(0, 4)],
        }
        pii_flags = [True]
        token_ids = [99, 1]
        content_start = 1

        spans = _build_spans_from_pii_tokens(
            pii_flags, token_ids, content_start, mock_tokenizer, text,
        )
        assert all(s["type"] == "pii" for s in spans)
