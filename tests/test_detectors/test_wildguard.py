import pytest
import torch

from detectors.guards.wildguard import (
    _parse_result,
    classify_pii,
    classify_pii_batch,
    wildguard_pii_detector,
)


PII_SPAN = {"value": None, "start": None, "end": None, "type": "pii"}


@pytest.fixture()
def mock_wildguard(mocker):
    mock_tokenizer = mocker.MagicMock()
    mock_model = mocker.MagicMock()
    mock_model.device = torch.device("cpu")

    mocker.patch(
        "detectors.guards.wildguard._get_model",
        return_value=(mock_tokenizer, mock_model),
    )
    return mock_tokenizer, mock_model


def _setup_generate(mock_tokenizer, mock_model, decoded_output: str):
    input_ids = torch.tensor([[1, 2, 3]])
    mock_encoding = mock_tokenizer.return_value
    mock_encoding.__getitem__ = lambda self, k: {"input_ids": input_ids}[k]
    mock_encoding.to.return_value = mock_encoding

    output_tensor = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.generate.return_value = output_tensor
    mock_tokenizer.decode.return_value = decoded_output


# ── _parse_result ────────────────────────────────────────────────────


@pytest.mark.parametrize("text, expected", [
    ("Harmful request: yes\nResponse refusal: n/a", True),
    ("Harmful request: no\nResponse refusal: n/a", False),
    ("gibberish", False),
    # SentencePiece garbled output
    ("H arm ful ▁request : ▁yes <0x0A> R response", True),
    ("H arm ful ▁request : ▁no <0x0A> R response", False),
])
def test_parse_result(text, expected):
    assert _parse_result(text) is expected


# ── classify_pii ─────────────────────────────────────────────────────


class TestClassifyPii:

    def test_safe_returns_false(self, mock_wildguard):
        tokenizer, model = mock_wildguard
        _setup_generate(tokenizer, model, "Harmful request: no")
        assert classify_pii("hello world") is False

    def test_harmful_returns_true(self, mock_wildguard):
        tokenizer, model = mock_wildguard
        _setup_generate(tokenizer, model, "Harmful request: yes")
        assert classify_pii("my ssn is 123-45-6789") is True

    def test_unparseable_returns_false(self, mock_wildguard):
        tokenizer, model = mock_wildguard
        _setup_generate(tokenizer, model, "gibberish output")
        assert classify_pii("some text") is False


# ── classify_pii_batch ───────────────────────────────────────────────


class TestClassifyPiiBatch:

    def test_batch_mixed_results(self, mock_wildguard):
        tokenizer, model = mock_wildguard

        input_ids = torch.tensor([[1, 2, 3]])
        mock_encoding = tokenizer.return_value
        mock_encoding.__getitem__ = lambda self, k: {"input_ids": input_ids}[k]
        mock_encoding.to.return_value = mock_encoding

        output_tensor = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 6, 7]])
        model.generate.return_value = output_tensor
        tokenizer.decode.side_effect = [
            "Harmful request: yes",
            "Harmful request: no",
        ]

        results = classify_pii_batch(["pii text", "safe text"])
        assert results == [True, False]


# ── wildguard_pii_detector ───────────────────────────────────────────


class TestWildguardPiiDetector:

    def test_none_returns_empty(self, mock_wildguard):
        assert wildguard_pii_detector(None) == []

    def test_safe_returns_empty(self, mock_wildguard):
        tokenizer, model = mock_wildguard
        _setup_generate(tokenizer, model, "Harmful request: no")
        assert wildguard_pii_detector("hello") == []

    def test_pii_detected_returns_span(self, mock_wildguard):
        tokenizer, model = mock_wildguard
        _setup_generate(tokenizer, model, "Harmful request: yes")
        assert wildguard_pii_detector("my ssn is 123-45-6789") == [PII_SPAN]
