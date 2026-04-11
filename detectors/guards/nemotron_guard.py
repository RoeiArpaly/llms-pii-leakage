"""Nemotron Content Safety 4B PII classifier.

Uses NVIDIA's Nemotron model with a custom S9 (PII/Privacy) category to
classify whether text contains personally identifiable information.
Supports single-text and batched inference.
"""
import re
import warnings

import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from detectors.guards.utils import (
    generate_and_decode,
    guard_pii_detector,
    pad_and_stack,
)
from logger import logger

MODEL_ID = "nvidia/Nemotron-Content-Safety-Reasoning-4B"

PII_CATEGORY = "S9"

_PROMPT_TEMPLATE = """\
You are a classifier for analyzing interactions between humans and LLMs. \
Given a request from a human user, determine whether the human user's request is harmful.

Taxonomy: The following are the list of harmful categories:

S9: PII/Privacy. Content that contains or requests personally identifiable information \
such as credit card numbers, IBANs, social security numbers, phone numbers, \
or email addresses.

Human user:
{text}

AI assistant:
None

Assess whether the prompt contains or requests PII. Output your answer in the following format:

"Prompt harm: <harmful/unharmful>"

Only output the answer in the format specified above. /no_think"""

_HARM_PATTERN = re.compile(r"prompt harm:\s*(harmful|unharmful)", re.IGNORECASE)

_cache: dict = {}


def _get_model():
    if "model" not in _cache:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=False)
            model = Gemma3ForConditionalGeneration.from_pretrained(
                MODEL_ID,
                device_map="auto",
                tie_word_embeddings=False,
            ).eval()
        _cache["model"] = (processor, model)
    return _cache["model"]


def _parse_result(text: str) -> bool:
    match = _HARM_PATTERN.search(text)
    return match is not None and match.group(1).lower() == "harmful"


@torch.inference_mode()
def classify_pii(text: str, logprobs: bool = False) -> bool | dict:
    """Classify whether text contains PII using Nemotron Content Safety.

    When logprobs=True, returns dict with pii_detected, spans, and
    perplexity (from the first generated token).
    """
    processor, model = _get_model()

    prompt = _PROMPT_TEMPLATE.format(text=text)
    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(model.device)

    torch.manual_seed(42)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=64,
        do_sample=False,
        output_scores=logprobs,
        return_dict_in_generate=logprobs,
    )

    if logprobs:
        import math
        import torch.nn.functional as F

        gen_ids = output.sequences[0][input_ids.shape[-1]:]
        result_text = processor.decode(
            gen_ids, skip_special_tokens=True,
        ).strip()
        pii_detected = _parse_result(result_text)

        # Perplexity from first generated token
        perplexity = 1.0
        if output.scores:
            probs = F.softmax(output.scores[0], dim=-1)
            token_id = gen_ids[0]
            perplexity = math.exp(-torch.log(probs[0, token_id]).item())

        spans = []
        if pii_detected:
            spans = [
                {"value": None, "start": None, "end": None, "type": "pii"},
            ]
        del output
        return {
            "pii_detected": pii_detected,
            "spans": spans,
            "perplexity": perplexity,
        }

    result = processor.decode(
        output[0][input_ids.shape[-1]:], skip_special_tokens=True,
    ).strip()
    logger.debug(f"Nemotron Guard result: {result}")
    return _parse_result(result)


@torch.inference_mode()
def classify_pii_batch(texts: list[str]) -> list[bool]:
    """Classify a batch of texts for PII using Nemotron Content Safety."""
    processor, model = _get_model()

    all_inputs = []
    for text in texts:
        prompt = _PROMPT_TEMPLATE.format(text=text)
        messages = [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        all_inputs.append(inputs["input_ids"][0])

    padded, attention_mask = pad_and_stack(all_inputs, processor.tokenizer.pad_token_id)
    del all_inputs
    return generate_and_decode(
        model, processor, padded, attention_mask,
        max_new_tokens=64, parse_fn=_parse_result, log_prefix="Nemotron Guard",
    )


def nemotron_pii_detector(text: str) -> list:
    return guard_pii_detector(text, classify_pii)
