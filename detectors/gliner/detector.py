"""GLiNER-based PII detector using transformer NER models.

Supports multiple GLiNER variants (urchade/gliner_multi_pii-v1, nvidia/gliner-PII)
with native PyTorch inference and optional quantization. Provides both single-text
and batch detection with span-level output.
"""
import logging
import os
import platform
import warnings

# Silence HF/torch logs and tqdm bars before importing them.
for _name in ("transformers", "huggingface_hub", "accelerate", "torch", "gliner"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from huggingface_hub.utils import disable_progress_bars  # noqa: E402
disable_progress_bars()

import torch  # noqa: E402

from gliner import GLiNER  # noqa: E402
from constants import (
    GLINER_INVALID_VALUES,
    PII_ENTITIES,
)
from logger import logger

GLINER_MODELS = {
    "gliner": "urchade/gliner_multi_pii-v1",
    "gliner-nvidia": "nvidia/gliner-PII",
}

# nvidia/gliner-PII uses different label names — map them to our standard types
_NVIDIA_LABEL_MAP = {
    "credit_card_number": "credit_card_number",
    "iban_code": "iban",
    "SSN": "ssn",
    "social_security_number": "ssn",
    "phone_number": "phone_number",
    "email": "email",
    "email_address": "email",
}

# Labels to request from each model
_MODEL_LABELS = list(PII_ENTITIES.values())

_model_cache: dict = {}


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_gliner_model(model_name: str = "gliner"):
    if model_name not in _model_cache:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        model_id = GLINER_MODELS[model_name]
        device = _get_device()

        kwargs = {"max_length": 4096}
        if device == "cuda":
            kwargs["quantize"] = "fp16"
        elif device == "cpu":
            kwargs["quantize"] = "int8"

        # torch.compile gives ~1.4x speedup on Linux (not supported on macOS)
        if platform.system() == "Linux":
            kwargs["compile_torch_model"] = True

        logger.info(f"Loading {model_name} ({model_id}) on {device}")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*resume_download.*")
            model = GLiNER.from_pretrained(model_id, **kwargs)
        model.to(device)
        _model_cache[model_name] = model
    return _model_cache[model_name]


def _filter_spans(spans: list[dict], label_map: dict = None) -> list[dict]:
    results = []
    for span in spans:
        span["value"] = span.pop("text")
        raw_label = span.pop("label")
        span["type"] = label_map[raw_label] if label_map else raw_label
        if all(val not in span["value"].lower() for val in GLINER_INVALID_VALUES):
            results.append(span)
    return results


def _get_label_map(model_name: str) -> dict | None:
    return _NVIDIA_LABEL_MAP if model_name == "gliner-nvidia" else None


@torch.inference_mode()
def gliner_pii_detector(
    text: str, threshold: float = 0.5, model_name: str = "gliner",
) -> list:
    model = get_gliner_model(model_name)
    spans = model.predict_entities(text=text, labels=_MODEL_LABELS, threshold=threshold)
    return _filter_spans(spans, _get_label_map(model_name))


@torch.inference_mode()
def gliner_pii_detector_batch(
    texts: list[str], threshold: float = 0.5, model_name: str = "gliner",
) -> list[list[dict]]:
    model = get_gliner_model(model_name)
    label_map = _get_label_map(model_name)
    batch_spans = model.inference(
        texts=list(texts), labels=_MODEL_LABELS, threshold=threshold,
    )
    return [_filter_spans(spans, label_map) for spans in batch_spans]
