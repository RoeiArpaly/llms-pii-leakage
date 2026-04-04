"""Llama Guard 3 PII classifier (1B and 8B variants).

Uses Meta's Llama Guard model with a custom S7 (Privacy/PII) safety category
to classify whether text contains personally identifiable information.
Supports single-text and batched inference.
"""
import re

import torch

from detectors.guards.utils import (
    generate_and_decode,
    guard_pii_detector,
    load_guard_model,
    pad_and_stack,
)

LLAMA_GUARD_MODELS = {
    "llama-guard-3-1b": "meta-llama/Llama-Guard-3-1B",
    "llama-guard-3-8b": "meta-llama/Llama-Guard-3-8B",
}

PII_CATEGORY = "S7"
CUSTOM_CATEGORIES = {
    "S7": (
        "Privacy: Content that contains or requests personally identifiable information "
        "such as credit card numbers, IBANs, social security numbers, phone numbers, "
        "or email addresses."
    ),
}


def _get_model(model_name: str):
    return load_guard_model(
        model_name,
        LLAMA_GUARD_MODELS[model_name],
        dtype=torch.bfloat16,
        device_map="auto",
    )


def _parse_result(text: str) -> bool:
    if text.startswith("unsafe"):
        categories = re.findall(r"S\d+", text)
        return PII_CATEGORY in categories
    return False


@torch.inference_mode()
def classify_pii(text: str, model_name: str = "llama-guard-3-1b") -> bool:
    """Classify whether text contains PII using Llama Guard 3."""
    tokenizer, model = _get_model(model_name)

    conversation = [{"role": "user", "content": text}]
    input_ids = tokenizer.apply_chat_template(
        conversation,
        return_tensors="pt",
        categories=CUSTOM_CATEGORIES,
    ).to(model.device)

    output = model.generate(input_ids=input_ids, max_new_tokens=20, pad_token_id=0)
    result = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    return _parse_result(result)


@torch.inference_mode()
def classify_pii_batch(texts: list[str], model_name: str = "llama-guard-3-1b") -> list[bool]:
    """Classify a batch of texts for PII using Llama Guard 3."""
    tokenizer, model = _get_model(model_name)

    inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            return_tensors="pt",
            categories=CUSTOM_CATEGORIES,
        )
        for t in texts
    ]
    padded, attention_mask = pad_and_stack(inputs, tokenizer.pad_token_id)
    return generate_and_decode(
        model, tokenizer, padded, attention_mask,
        max_new_tokens=20, parse_fn=_parse_result, log_prefix="Llama Guard",
    )


def llama_guard_pii_detector(
    text: str, model_name: str = "llama-guard-3-1b",
) -> list:
    return guard_pii_detector(text, classify_pii, model_name=model_name)
