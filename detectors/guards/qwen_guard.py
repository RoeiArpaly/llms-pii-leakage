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
def classify_pii(
    text: str, model_name: str = "qwen-guard-4b", logprobs: bool = False,
) -> bool | dict:
    """Classify whether text contains PII using Qwen3Guard-Gen.

    When logprobs=False (default), returns a bool.
    When logprobs=True, returns a dict with pii_detected, spans,
    and perplexity (derived from the classification token logprob).
    """
    tokenizer, model = _get_model(model_name)

    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    torch.manual_seed(42)
    outputs = model.generate(
        **inputs, max_new_tokens=128, do_sample=False,
        output_scores=logprobs, return_dict_in_generate=logprobs,
    )

    if logprobs:
        import math
        import torch.nn.functional as F
        gen_ids = outputs.sequences[0][inputs.input_ids.shape[-1]:]
        content = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pii_detected = _has_pii(content)

        # Perplexity from the 3rd token (classification: Safe/Cont/Uns)
        perplexity = 1.0
        cls_idx = 2
        if len(outputs.scores) > cls_idx:
            probs = F.softmax(outputs.scores[cls_idx], dim=-1)
            token_id = outputs.sequences[
                0, inputs.input_ids.shape[-1] + cls_idx
            ]
            perplexity = math.exp(-torch.log(probs[0, token_id]).item())

        spans = []
        if pii_detected:
            spans = [
                {"value": None, "start": None, "end": None, "type": "pii"},
            ]
        del outputs
        logger.debug(f"Qwen Guard: {content.strip()}, ppl={perplexity:.2f}")
        return {
            "pii_detected": pii_detected,
            "spans": spans,
            "perplexity": perplexity,
        }

    output_ids = outputs[0][inputs.input_ids.shape[-1]:]
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
    del all_inputs
    return generate_and_decode(
        model, tokenizer, padded, attention_mask,
        max_new_tokens=128, parse_fn=_has_pii,
        log_prefix="Qwen Guard",
    )


def qwen_guard_pii_detector(text: str, model_name: str = "qwen-guard-4b") -> list:
    return guard_pii_detector(text, classify_pii, model_name=model_name)
