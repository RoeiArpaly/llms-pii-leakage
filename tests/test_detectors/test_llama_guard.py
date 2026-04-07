import pytest
import torch

from detectors.guards.llama_guard import (
    LLAMA_GUARD_MODELS,
    classify_pii,
    classify_pii_batch,
    llama_guard_pii_detector,
)


PII_SPAN = {"value": None, "start": None, "end": None, "type": "pii"}


@pytest.fixture()
def mock_llama_guard(mocker):
    """Mock _get_model to return a fake tokenizer + model."""
    mock_tokenizer = mocker.MagicMock()
    mock_model = mocker.MagicMock()
    mock_model.device = torch.device("cpu")

    mocker.patch(
        "detectors.guards.llama_guard._get_model",
        return_value=(mock_tokenizer, mock_model),
    )
    return mock_tokenizer, mock_model


def _setup_generate(mock_tokenizer, mock_model, decoded_output: str):
    """Configure mocks so classify_pii returns based on decoded_output."""
    input_ids = torch.tensor([[1, 2, 3]])
    mock_tokenizer.apply_chat_template.return_value = {
        "input_ids": input_ids,
    }
    mock_tokenizer.pad_token_id = 0

    output_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.generate.return_value = output_tensor
    mock_tokenizer.decode.return_value = decoded_output


class TestLlamaGuardModels:

    def test_model_mapping_has_1b_and_8b(self):
        assert "llama-guard-3-1b" in LLAMA_GUARD_MODELS
        assert "llama-guard-3-8b" in LLAMA_GUARD_MODELS


class TestClassifyPii:

    def test_safe_returns_false(self, mock_llama_guard):
        tokenizer, model = mock_llama_guard
        _setup_generate(tokenizer, model, "safe")
        assert classify_pii("hello world") is False

    def test_unsafe_s7_returns_true(self, mock_llama_guard):
        tokenizer, model = mock_llama_guard
        _setup_generate(tokenizer, model, "unsafe\nS7")
        assert classify_pii("my ssn is 123-45-6789") is True

    def test_unsafe_other_category_returns_false(self, mock_llama_guard):
        tokenizer, model = mock_llama_guard
        _setup_generate(tokenizer, model, "unsafe\nS1")
        assert classify_pii("violent content") is False

    def test_unsafe_multiple_categories_with_s7(self, mock_llama_guard):
        tokenizer, model = mock_llama_guard
        _setup_generate(tokenizer, model, "unsafe\nS1,S7")
        assert classify_pii("violent with pii") is True

    def test_accepts_model_name_kwarg(self, mock_llama_guard):
        tokenizer, model = mock_llama_guard
        _setup_generate(tokenizer, model, "unsafe\nS7")
        assert classify_pii("pii text", model_name="llama-guard-3-8b") is True


class TestClassifyPiiBatch:

    def test_batch_mixed_results(self, mock_llama_guard):
        tokenizer, model = mock_llama_guard
        input_ids = torch.tensor([[1, 2, 3]])
        tokenizer.apply_chat_template.return_value = {
            "input_ids": input_ids,
        }
        tokenizer.pad_token_id = 0

        # Generate returns 2 outputs
        output_tensor = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]])
        model.generate.return_value = output_tensor
        tokenizer.decode.side_effect = ["unsafe\nS7", "safe"]

        results = classify_pii_batch(["pii text", "safe text"])
        assert results == [True, False]


class TestLlamaGuardPiiDetector:

    def test_none_returns_empty(self, mock_llama_guard):
        assert llama_guard_pii_detector(None) == []

    def test_safe_returns_empty(self, mock_llama_guard):
        tokenizer, model = mock_llama_guard
        _setup_generate(tokenizer, model, "safe")
        assert llama_guard_pii_detector("hello") == []

    def test_pii_detected_returns_span(self, mock_llama_guard):
        tokenizer, model = mock_llama_guard
        _setup_generate(tokenizer, model, "unsafe\nS7")
        assert llama_guard_pii_detector("my ssn is 123-45-6789") == [PII_SPAN]

    def test_accepts_model_name(self, mock_llama_guard):
        tokenizer, model = mock_llama_guard
        _setup_generate(tokenizer, model, "unsafe\nS7")
        result = llama_guard_pii_detector("pii text", model_name="llama-guard-3-8b")
        assert result == [PII_SPAN]
