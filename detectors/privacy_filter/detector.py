"""OpenAI privacy-filter as a span-level PII detector.

Wraps the canonical token-classification pipeline from
https://huggingface.co/openai/privacy-filter and filters its 8 native
categories down to the project's PII types (CREDIT_CARD, IBAN_CODE,
US_SSN, PHONE_NUMBER, EMAIL_ADDRESS). The model's ``account_number``
category is split downstream via Luhn / Mod-97 / SSN-format checks
since it covers all three numeric ID types; everything outside the
five project categories (addresses, persons, URLs, dates, secrets)
is dropped.
"""
import logging
import os
import warnings

for _name in ("transformers", "huggingface_hub", "accelerate", "torch"):
    logging.getLogger(_name).setLevel(logging.ERROR)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from huggingface_hub.utils import disable_progress_bars  # noqa: E402
disable_progress_bars()

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

from data_generation.pii_validators import (  # noqa: E402
    is_valid_credit_card,
    is_valid_iban,
)
from detectors.validators import _is_valid_ssn  # noqa: E402
from logger import logger  # noqa: E402


PRIVACY_FILTER_MODELS = {
    "openai-privacy-filter": "openai/privacy-filter",
}

# Direct mappings from privacy-filter categories to project PII types.
# `account_number` is intentionally absent — it covers credit card /
# IBAN / SSN, so it must be disambiguated via checksum/regex.
_DIRECT_MAP = {
    "private_email": "email",
    "private_phone": "phone_number",
}

_model_cache: dict = {}


def _get_pipeline(model_name: str = "openai-privacy-filter"):
    if model_name not in _model_cache:
        model_id = PRIVACY_FILTER_MODELS[model_name]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            clf = pipeline(
                task="token-classification",
                model=model_id,
                device_map="auto",
                aggregation_strategy="simple",
            )
        _model_cache[model_name] = clf
        logger.info(f"Loaded {model_name} ({model_id})")
    return _model_cache[model_name]


def _classify_account_number(value: str) -> str | None:
    """Disambiguate a privacy-filter ``account_number`` span.

    Returns the project PII type if the value passes a credit-card
    Luhn check, an IBAN Mod-97 check, or a structurally valid SSN
    range check; otherwise None (the span is dropped).
    """
    cleaned = value.strip()
    # CC/IBAN/SSN are ASCII-only formats. Reject Unicode look-alikes
    # (e.g. subscript digits ₀-₉ from fuzzy attacks) before they reach
    # the validators — str.isdigit() accepts them but int() rejects them.
    if not cleaned.isascii():
        return None
    if is_valid_credit_card(cleaned):
        return "credit_card_number"
    if is_valid_iban(cleaned):
        return "iban"
    if _is_valid_ssn(cleaned):
        return "ssn"
    return None


def _merge_adjacent(raw_spans: list[dict]) -> list[dict]:
    """Merge consecutive same-category spans with no gap between them.

    The privacy-filter pipeline sometimes emits a long entity as a
    B-...E- pair followed by an S- token, producing two adjacent
    spans (e.g. a 16-digit credit card split at offset 18 and 36).
    Strict adjacency (prev.end == cur.start) is required so unrelated
    runs separated by punctuation aren't fused.
    """
    merged: list[dict] = []
    for s in raw_spans:
        if (
            merged
            and merged[-1]["entity_group"] == s["entity_group"]
            and merged[-1]["end"] == s["start"]
        ):
            prev = merged[-1]
            prev["end"] = s["end"]
            prev["score"] = min(prev["score"], s["score"])
        else:
            merged.append(dict(s))
    return merged


def _to_project_spans(merged: list[dict], text: str) -> list[dict]:
    """Map merged privacy-filter spans into the project's PII span schema.

    Drops categories outside the project scope (addresses, persons,
    URLs, dates, secrets). ``account_number`` is split into CC / IBAN /
    SSN by checksum validators and otherwise kept as ``account_number``.
    """
    spans: list[dict] = []
    for s in merged:
        start, end = s["start"], s["end"]
        value = text[start:end]
        cat = s["entity_group"]

        if cat == "account_number":
            # Validators disambiguate into one of the three numeric
            # project types (CC / IBAN / SSN). When none match, keep
            # the span with the model's coarse class as the type.
            pii_type = _classify_account_number(value) or "account_number"
        elif cat in _DIRECT_MAP:
            pii_type = _DIRECT_MAP[cat]
        else:
            continue

        spans.append({
            "value": value,
            "start": start,
            "end": end,
            "type": pii_type,
        })
    return spans


@torch.inference_mode()
def privacy_filter_pii_detector(
    text: str, model_name: str = "openai-privacy-filter",
) -> list:
    if not text:
        return []
    classifier = _get_pipeline(model_name)
    raw = classifier(text)
    return _to_project_spans(_merge_adjacent(raw), text)


@torch.inference_mode()
def privacy_filter_pii_detector_batch(
    texts: list[str], model_name: str = "openai-privacy-filter",
) -> list[list[dict]]:
    classifier = _get_pipeline(model_name)
    out: list[list[dict]] = [[] for _ in texts]
    valid = [(i, t) for i, t in enumerate(texts) if t]
    if not valid:
        return out

    valid_texts = [t for _, t in valid]
    raw_batches = classifier(valid_texts)
    if isinstance(raw_batches, dict):
        raw_batches = [raw_batches]

    for (idx, text), raw in zip(valid, raw_batches):
        out[idx] = _to_project_spans(_merge_adjacent(raw), text)
    return out
