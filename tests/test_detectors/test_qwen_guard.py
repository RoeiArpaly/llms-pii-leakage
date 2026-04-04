import pytest
import torch

from detectors.guards.qwen_guard import (
    _has_pii,
    classify_pii,
    classify_pii_batch,
    qwen_guard_pii_detector,
)


PII_SPAN = {"value": None, "start": None, "end": None, "type": "pii"}


# ── _has_pii ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "content, expected",
    [
        ("Safety: Safe\nCategories: None", False),
        ("Safety: Unsafe\nCategories: PII", True),
        ("Safety: Controversial\nCategories: PII", True),
        ("Safety: Unsafe\nCategories: Violent", False),
        ("Safety: Unsafe\nCategories: Violent, PII", True),
        ("", False),
        ("random text", False),
    ],
)
def test_has_pii(content, expected):
    assert _has_pii(content) is expected


# ── classify_pii (mocked) ───────────────────────────────────────────


@pytest.fixture()
def mock_qwen(mocker):
    mock_tokenizer = mocker.MagicMock()
    mock_model = mocker.MagicMock()
    mock_model.device = torch.device("cpu")

    mocker.patch(
        "detectors.guards.qwen_guard._get_model",
        return_value=(mock_tokenizer, mock_model),
    )
    return mock_tokenizer, mock_model


def _setup_generate(mock_tokenizer, mock_model, decoded_output: str):
    mocker_inputs = mock_tokenizer.return_value
    mocker_inputs.to.return_value = mocker_inputs
    mocker_inputs.input_ids = torch.tensor([[1, 2, 3]])

    output_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.generate.return_value = output_tensor
    mock_tokenizer.decode.return_value = decoded_output


class TestClassifyPii:

    def test_safe(self, mock_qwen):
        tokenizer, model = mock_qwen
        _setup_generate(tokenizer, model, "Safety: Safe\nCategories: None")
        assert classify_pii("hello", "qwen-guard-4b") is False

    def test_unsafe_pii(self, mock_qwen):
        tokenizer, model = mock_qwen
        _setup_generate(tokenizer, model, "Safety: Unsafe\nCategories: PII")
        assert classify_pii("my ssn", "qwen-guard-4b") is True

    def test_unsafe_non_pii(self, mock_qwen):
        tokenizer, model = mock_qwen
        _setup_generate(
            tokenizer, model, "Safety: Unsafe\nCategories: Violent",
        )
        assert classify_pii("violence", "qwen-guard-4b") is False


# ── classify_pii_batch (mocked) ─────────────────────────────────────


class TestClassifyPiiBatch:

    def test_batch_mixed(self, mock_qwen):
        tokenizer, model = mock_qwen
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.pad_token_id = 0
        tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3]])

        output_tensor = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]])
        model.generate.return_value = output_tensor
        tokenizer.decode.side_effect = [
            "Safety: Unsafe\nCategories: PII",
            "Safety: Safe\nCategories: None",
        ]

        results = classify_pii_batch(
            ["pii text", "safe text"], "qwen-guard-4b",
        )
        assert results == [True, False]

    def test_batch_all_safe(self, mock_qwen):
        tokenizer, model = mock_qwen
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.pad_token_id = 0
        tokenizer.return_value.input_ids = torch.tensor([[1, 2, 3]])

        output_tensor = torch.tensor([[1, 2, 3, 4, 5]])
        model.generate.return_value = output_tensor
        tokenizer.decode.return_value = "Safety: Safe\nCategories: None"

        results = classify_pii_batch(["safe"], "qwen-guard-4b")
        assert results == [False]


# ── Public API ──────────────────────────────────────────────────────


class TestQwenGuardPiiDetector:

    def test_none_returns_empty(self, mock_qwen):
        assert qwen_guard_pii_detector(None) == []

    def test_safe_returns_empty(self, mock_qwen):
        tokenizer, model = mock_qwen
        _setup_generate(tokenizer, model, "Safety: Safe\nCategories: None")
        assert qwen_guard_pii_detector("hello") == []

    def test_pii_returns_span(self, mock_qwen):
        tokenizer, model = mock_qwen
        _setup_generate(
            tokenizer, model, "Safety: Unsafe\nCategories: PII",
        )
        assert qwen_guard_pii_detector("my ssn") == [PII_SPAN]
