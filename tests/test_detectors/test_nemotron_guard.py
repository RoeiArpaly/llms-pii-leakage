import pytest
import torch

from detectors.guards.nemotron_guard import (
    classify_pii,
    classify_pii_batch,
    nemotron_pii_detector,
)


PII_SPAN = {"value": None, "start": None, "end": None, "type": "pii"}


@pytest.fixture()
def mock_nemotron(mocker):
    """Mock _get_model to return a fake processor + model."""
    mock_processor = mocker.MagicMock()
    mock_model = mocker.MagicMock()
    mock_model.device = torch.device("cpu")

    mocker.patch(
        "detectors.guards.nemotron_guard._get_model",
        return_value=(mock_processor, mock_model),
    )
    return mock_processor, mock_model


def _setup_generate(mock_processor, mock_model, decoded_output: str):
    input_ids = torch.tensor([[1, 2, 3]])
    mock_processor.apply_chat_template.return_value = {"input_ids": input_ids}
    mock_processor.tokenizer.pad_token_id = 0

    output_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.generate.return_value = output_tensor
    mock_processor.decode.return_value = decoded_output


class TestClassifyPii:

    def test_unharmful_returns_false(self, mock_nemotron):
        processor, model = mock_nemotron
        _setup_generate(processor, model, "Prompt harm: unharmful")
        assert classify_pii("hello world") is False

    def test_harmful_returns_true(self, mock_nemotron):
        processor, model = mock_nemotron
        _setup_generate(processor, model, "Prompt harm: harmful")
        assert classify_pii("my ssn is 123-45-6789") is True

    def test_unparseable_returns_false(self, mock_nemotron):
        processor, model = mock_nemotron
        _setup_generate(processor, model, "gibberish output")
        assert classify_pii("some text") is False

    def test_case_insensitive_match(self, mock_nemotron):
        processor, model = mock_nemotron
        _setup_generate(processor, model, "Prompt Harm: Harmful")
        assert classify_pii("pii text") is True


class TestClassifyPiiBatch:

    def test_batch_mixed_results(self, mock_nemotron):
        processor, model = mock_nemotron
        input_ids = torch.tensor([[1, 2, 3]])
        processor.apply_chat_template.return_value = {"input_ids": input_ids}
        processor.tokenizer.pad_token_id = 0

        output_tensor = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]])
        model.generate.return_value = output_tensor
        processor.decode.side_effect = [
            "Prompt harm: harmful",
            "Prompt harm: unharmful",
        ]

        results = classify_pii_batch(["pii text", "safe text"])
        assert results == [True, False]


class TestNemotronPiiDetector:

    def test_none_returns_empty(self, mock_nemotron):
        assert nemotron_pii_detector(None) == []

    def test_safe_returns_empty(self, mock_nemotron):
        processor, model = mock_nemotron
        _setup_generate(processor, model, "Prompt harm: unharmful")
        assert nemotron_pii_detector("hello") == []

    def test_pii_detected_returns_span(self, mock_nemotron):
        processor, model = mock_nemotron
        _setup_generate(processor, model, "Prompt harm: harmful")
        assert nemotron_pii_detector("my ssn is 123-45-6789") == [PII_SPAN]
