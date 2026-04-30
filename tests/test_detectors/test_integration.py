"""Integration tests that load real models and verify PII detection.

These tests are slow (model loading) but verify end-to-end
correctness without mocks. Prints input/output for debugging.
Run with: uv run pytest tests/test_detectors/test_integration.py -s

Models are loaded one at a time and unloaded after each test class
to avoid accumulating GPU/RAM usage.
"""
import gc

import pytest
import torch

from constants import PII_ENTITIES

STANDARD_TYPES = set(PII_ENTITIES.values())

PII_SAMPLES = {
    "ssn": "My SSN is 219-09-9999.",
    "credit_card_number": "My credit card number is 4111-1111-1111-1111.",
    "iban": "Please use my personal IBAN GB10WODC31116901210401 for the wire.",
    "phone_number": "My personal phone number is 457-492-0782.",
    "email": "Contact john.snow@gmail.com for details.",
}

SAFE_TEXTS = [
    "What are the best practices for code review?",
    "How do I reverse a linked list in Python?",
    "Explain the difference between TCP and UDP.",
    "What is the capital of France?",
]


def _flush_ram():
    """Force garbage collection and free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _log(detector: str, pii_type: str, text: str, result):
    """Print input/output for each detection call."""
    detected = len(result) > 0 if isinstance(result, list) else result
    mark = "✓" if detected else "✗"
    print(f"\n  {mark} [{detector}] {pii_type}")
    print(f"    Input:  {text}")
    print(f"    Output: {result}")


# ── Presidio ──────────────────────────────────────────


class TestPresidioIntegration:

    @pytest.mark.parametrize("pii_type", PII_SAMPLES)
    def test_detects_pii(self, pii_type):
        from detectors.presidio import presidio_pii_analyzer
        text = PII_SAMPLES[pii_type]
        result = presidio_pii_analyzer(text)
        _log("presidio", pii_type, text, result)
        assert len(result) > 0, f"Presidio missed {pii_type}"

    @pytest.mark.parametrize("safe_text", SAFE_TEXTS)
    def test_safe_text(self, safe_text):
        from detectors.presidio import presidio_pii_analyzer
        result = presidio_pii_analyzer(safe_text)
        _log("presidio", "safe", safe_text, result)
        assert result == []

    @pytest.mark.parametrize("pii_type", PII_SAMPLES)
    def test_types_are_standard(self, pii_type):
        from detectors.presidio import presidio_pii_analyzer
        result = presidio_pii_analyzer(PII_SAMPLES[pii_type])
        for span in result:
            assert span["type"] in STANDARD_TYPES, (
                f"Non-standard type: {span['type']}"
            )


# ── Presidio Fuzzy ────────────────────────────────────


class TestPresidioFuzzyIntegration:

    @pytest.mark.parametrize("pii_type", PII_SAMPLES)
    def test_detects_pii(self, pii_type):
        from detectors.presidio import (
            get_fuzzy_recognizers,
            presidio_pii_analyzer,
        )
        text = PII_SAMPLES[pii_type]
        result = presidio_pii_analyzer(
            text, recognizers=get_fuzzy_recognizers(),
        )
        _log("presidio-fuzzy", pii_type, text, result)
        assert len(result) > 0, f"Presidio-fuzzy missed {pii_type}"

    @pytest.mark.parametrize("safe_text", SAFE_TEXTS)
    def test_safe_text(self, safe_text):
        from detectors.presidio import (
            get_fuzzy_recognizers,
            presidio_pii_analyzer,
        )
        result = presidio_pii_analyzer(
            safe_text, recognizers=get_fuzzy_recognizers(),
        )
        _log("presidio-fuzzy", "safe", safe_text, result)
        assert result == []


# ── GLiNER ────────────────────────────────────────────


class TestGlinerIntegration:

    @pytest.fixture(autouse=True, scope="class")
    def _cleanup_model(self):
        yield
        from detectors.gliner.detector import _model_cache
        _model_cache.clear()
        _flush_ram()

    @pytest.mark.parametrize("model_name", ["gliner", "gliner-nvidia"])
    @pytest.mark.parametrize("pii_type", PII_SAMPLES)
    def test_detects_pii(self, model_name, pii_type):
        from detectors.gliner import gliner_pii_detector
        text = PII_SAMPLES[pii_type]
        result = gliner_pii_detector(text, model_name=model_name)
        _log(model_name, pii_type, text, result)
        assert len(result) > 0, f"{model_name} missed {pii_type}"

    @pytest.mark.parametrize("model_name", ["gliner", "gliner-nvidia"])
    @pytest.mark.parametrize("safe_text", SAFE_TEXTS)
    def test_safe_text(self, model_name, safe_text):
        from detectors.gliner import gliner_pii_detector
        result = gliner_pii_detector(safe_text, model_name=model_name)
        _log(model_name, "safe", safe_text, result)
        assert result == []

    @pytest.mark.parametrize("model_name", ["gliner", "gliner-nvidia"])
    @pytest.mark.parametrize("pii_type", PII_SAMPLES)
    def test_types_are_standard(self, model_name, pii_type):
        from detectors.gliner import gliner_pii_detector
        result = gliner_pii_detector(
            PII_SAMPLES[pii_type], model_name=model_name,
        )
        for span in result:
            assert span["type"] in STANDARD_TYPES, (
                f"{model_name} returned non-standard type '{span['type']}'"
            )

    @pytest.mark.parametrize("model_name", ["gliner", "gliner-nvidia"])
    def test_batch(self, model_name):
        from detectors.gliner import gliner_pii_detector_batch
        texts = list(PII_SAMPLES.values())
        results = gliner_pii_detector_batch(texts, model_name=model_name)
        assert len(results) == len(texts)
        for i, r in enumerate(results):
            assert len(r) > 0, f"{model_name} batch missed index {i}"


# ── Llama Guard ───────────────────────────────────────


class TestLlamaGuardIntegration:

    @pytest.fixture(autouse=True, scope="class")
    def _cleanup_model(self):
        yield
        from detectors.guards.utils import _model_cache
        for key in list(_model_cache):
            if key.startswith("llama-guard"):
                _model_cache.pop(key)
        _flush_ram()

    @pytest.mark.parametrize("model_name", [
        "llama-guard-3-1b", "llama-guard-3-8b",
    ])
    @pytest.mark.parametrize("pii_type", PII_SAMPLES)
    def test_detects_pii(self, model_name, pii_type):
        from detectors.guards.llama_guard import llama_guard_pii_detector
        text = PII_SAMPLES[pii_type]
        result = llama_guard_pii_detector(text, model_name=model_name)
        _log(model_name, pii_type, text, result)
        assert len(result) > 0, f"{model_name} missed {pii_type}"
        for span in result:
            assert span["value"] is None
            assert span["start"] is None
            assert span["end"] is None
            assert span["type"] == "pii"

    @pytest.mark.parametrize("model_name", [
        "llama-guard-3-1b", "llama-guard-3-8b",
    ])
    @pytest.mark.parametrize("safe_text", SAFE_TEXTS)
    def test_safe_text(self, model_name, safe_text):
        from detectors.guards.llama_guard import llama_guard_pii_detector
        result = llama_guard_pii_detector(safe_text, model_name=model_name)
        _log(model_name, "safe", safe_text, result)
        assert result == []

    @pytest.mark.parametrize("model_name", [
        "llama-guard-3-1b", "llama-guard-3-8b",
    ])
    def test_batch(self, model_name):
        import pandas as pd

        from detectors.guards.llama_guard import (
            llama_guard_pii_detector_batch,
        )
        texts = list(PII_SAMPLES.values())[:2]
        results = llama_guard_pii_detector_batch(
            pd.Series(texts), model_name=model_name,
        )
        _log(model_name, "batch", str(texts), results.tolist())
        assert len(results) == len(texts)
        for i, spans in enumerate(results):
            assert len(spans) > 0, f"{model_name} batch missed index {i}"
            assert spans[0]["type"] == "pii"
            assert spans[0]["value"] is None

    @pytest.mark.parametrize("model_name", [
        "llama-guard-3-1b", "llama-guard-3-8b",
    ])
    def test_none_returns_empty(self, model_name):
        from detectors.guards.llama_guard import llama_guard_pii_detector
        assert llama_guard_pii_detector(None, model_name=model_name) == []


# ── Nemotron ──────────────────────────────────────────


class TestNemotronIntegration:

    @pytest.fixture(autouse=True, scope="class")
    def _cleanup_model(self):
        yield
        from detectors.guards.nemotron_guard import _cache
        _cache.clear()
        _flush_ram()

    @pytest.mark.parametrize("pii_type", PII_SAMPLES)
    def test_detects_pii(self, pii_type):
        from detectors.guards.nemotron_guard import nemotron_pii_detector
        text = PII_SAMPLES[pii_type]
        result = nemotron_pii_detector(text)
        _log("nemotron-4b", pii_type, text, result)
        assert len(result) > 0, f"Nemotron missed {pii_type}"
        for span in result:
            assert span["value"] is None
            assert span["type"] == "pii"

    @pytest.mark.parametrize("safe_text", SAFE_TEXTS)
    def test_safe_text(self, safe_text):
        from detectors.guards.nemotron_guard import nemotron_pii_detector
        result = nemotron_pii_detector(safe_text)
        _log("nemotron-4b", "safe", safe_text, result)
        assert result == []

    def test_batch(self):
        from detectors.guards.nemotron_guard import classify_pii_batch
        texts = list(PII_SAMPLES.values())[:2]
        results = classify_pii_batch(texts)
        _log("nemotron-4b", "batch", str(texts), results)
        assert len(results) == len(texts)
        assert all(isinstance(r, bool) for r in results)


# ── WildGuard ─────────────────────────────────────────


class TestWildGuardIntegration:

    @pytest.fixture(autouse=True, scope="class")
    def _cleanup_model(self):
        yield
        from detectors.guards.utils import _model_cache
        _model_cache.pop("wildguard-7b", None)
        _flush_ram()

    @pytest.mark.parametrize("pii_type", PII_SAMPLES)
    def test_detects_pii(self, pii_type):
        from detectors.guards.wildguard import wildguard_pii_detector
        text = PII_SAMPLES[pii_type]
        result = wildguard_pii_detector(text)
        _log("wildguard-7b", pii_type, text, result)
        assert len(result) > 0, f"WildGuard missed {pii_type}"
        for span in result:
            assert span["value"] is None
            assert span["type"] == "pii"

    @pytest.mark.parametrize("safe_text", SAFE_TEXTS)
    def test_safe_text(self, safe_text):
        from detectors.guards.wildguard import wildguard_pii_detector
        result = wildguard_pii_detector(safe_text)
        _log("wildguard-7b", "safe", safe_text, result)
        assert result == []

    def test_batch(self):
        from detectors.guards.wildguard import classify_pii_batch
        texts = list(PII_SAMPLES.values())[:2]
        results = classify_pii_batch(texts)
        _log("wildguard-7b", "batch", str(texts), results)
        assert len(results) == len(texts)
        assert all(isinstance(r, bool) for r in results)


# ── Qwen Guard ───────────────────────────────────────


class TestQwenGuardIntegration:

    @pytest.fixture(autouse=True, scope="class")
    def _cleanup_model(self):
        yield
        from detectors.guards.qwen_guard import _model_cache
        _model_cache.clear()
        _flush_ram()

    @pytest.mark.parametrize("model_name", [
        "qwen-guard-0.6b", "qwen-guard-4b",
    ])
    @pytest.mark.parametrize("pii_type", PII_SAMPLES)
    def test_detects_pii(self, model_name, pii_type):
        from detectors.guards.qwen_guard import qwen_guard_pii_detector
        text = PII_SAMPLES[pii_type]
        result = qwen_guard_pii_detector(text, model_name=model_name)
        _log(model_name, pii_type, text, result)
        assert len(result) > 0, f"{model_name} missed {pii_type}"
        for span in result:
            assert span["value"] is None
            assert span["type"] == "pii"

    @pytest.mark.parametrize("model_name", [
        "qwen-guard-0.6b", "qwen-guard-4b",
    ])
    @pytest.mark.parametrize("safe_text", SAFE_TEXTS)
    def test_safe_text(self, model_name, safe_text):
        from detectors.guards.qwen_guard import qwen_guard_pii_detector
        result = qwen_guard_pii_detector(
            safe_text, model_name=model_name,
        )
        _log(model_name, "safe", safe_text, result)
        assert result == []

    @pytest.mark.parametrize("model_name", [
        "qwen-guard-0.6b", "qwen-guard-4b",
    ])
    def test_batch(self, model_name):
        from detectors.guards.qwen_guard import classify_pii_batch
        texts = list(PII_SAMPLES.values())[:2]
        results = classify_pii_batch(texts, model_name=model_name)
        _log(model_name, "batch", str(texts), results)
        assert len(results) == len(texts)
        assert all(isinstance(r, bool) for r in results)


# ── LLM (GPT-4o-mini via mock) ───────────────────────

class TestLlmDetectorIntegration:

    @pytest.mark.parametrize("pii_type", PII_SAMPLES)
    def test_detects_pii(self, pii_type):
        from detectors.llm import llm_pii_detector
        text = PII_SAMPLES[pii_type]
        result = llm_pii_detector(text)
        _log("gpt-4o-mini", pii_type, text, result)
        assert isinstance(result, dict)
        assert isinstance(result["spans"], list)
        assert result["result"]["pii_detected"] is True, (
            f"LLM missed {pii_type}"
        )
        for span in result["spans"]:
            assert "value" in span
            assert "start" in span
            assert "end" in span
            assert "type" in span

    @pytest.mark.parametrize("safe_text", SAFE_TEXTS)
    def test_safe_text(self, safe_text):
        from detectors.llm import llm_pii_detector
        result = llm_pii_detector(safe_text)
        _log("gpt-4o-mini", "safe", safe_text, result)
        assert result["spans"] == []

    def test_none_returns_empty_spans(self):
        from detectors.llm import llm_pii_detector
        result = llm_pii_detector(None)
        assert isinstance(result, dict)
        assert result["spans"] == []
