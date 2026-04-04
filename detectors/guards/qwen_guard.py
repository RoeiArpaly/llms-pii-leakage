"""Qwen Guard PII detector.

Uses Qwen3Guard-Gen models to classify whether text contains PII.
Based on the simple inference pattern from qwen_easy_guard.py.
"""
import logging
import warnings

for _name in ("transformers", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from detectors.guards.utils import (
    generate_and_decode,
    guard_pii_detector,
    pad_and_stack,
)
from logger import logger

QWEN_GUARD_MODELS = {
    "qwen-guard-0.6b": "Qwen/Qwen3Guard-Gen-0.6B",
    "qwen-guard-4b": "Qwen/Qwen3Guard-Gen-4B",
}

_model_cache: dict = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        model_id = QWEN_GUARD_MODELS[model_name]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype="auto", device_map="auto",
            )
        _model_cache[model_name] = (tokenizer, model)
    return _model_cache[model_name]


def _has_pii(content: str) -> bool:
    return "PII" in content


@torch.inference_mode()
def classify_pii(text: str, model_name: str = "qwen-guard-4b") -> bool:
    """Classify whether text contains PII using Qwen3Guard-Gen."""
    tokenizer, model = _get_model(model_name)

    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    torch.manual_seed(42)
    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    output_ids = generated_ids[0][inputs.input_ids.shape[-1]:]
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    logger.debug(f"Qwen Guard result: {content}")
    return _has_pii(content)


@torch.inference_mode()
def classify_pii_batch(
    texts: list[str], model_name: str = "qwen-guard-4b",
) -> list[bool]:
    """Classify a batch of texts for PII using Qwen3Guard-Gen."""
    tokenizer, model = _get_model(model_name)

    all_inputs = [
        tokenizer(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": t}], tokenize=False,
            ),
            return_tensors="pt",
        ).input_ids[0]
        for t in texts
    ]
    padded, attention_mask = pad_and_stack(all_inputs, tokenizer.pad_token_id)
    return generate_and_decode(
        model, tokenizer, padded, attention_mask,
        max_new_tokens=128, parse_fn=_has_pii,
        log_prefix="Qwen Guard",
    )


def qwen_guard_pii_detector(text: str, model_name: str = "qwen-guard-4b") -> list:
    return guard_pii_detector(text, classify_pii, model_name=model_name)
