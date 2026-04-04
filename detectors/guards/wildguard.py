"""WildGuard PII classifier using Allen AI's WildGuard model.

Classifies whether a user request is harmful (contains PII) by parsing the
"Harmful request: yes/no" output pattern. Supports batched inference.

Reference: https://huggingface.co/allenai/wildguard
"""
import re

import torch

from detectors.guards.utils import (
    guard_pii_detector,
    load_guard_model,
)

MODEL_ID = "allenai/wildguard"

_PROMPT_TEMPLATE = """<s><|user|>
[INST] You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.

Human user:
{prompt}

AI assistant:
{response}

---

Answers: [/INST]
<|assistant|>
"""

_HARMFUL_PATTERN = re.compile(r"[Hh]armful\s*request\s*:\s*(yes|no)", re.IGNORECASE)


def _get_model():
    return load_guard_model(
        "wildguard",
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )


def _parse_result(text: str) -> bool:
    # WildGuard's SentencePiece tokenizer may produce sub-token output
    # like "H arm ful ▁request : ▁yes". Collapse it before matching.
    if "▁" in text or "<0x0A>" in text:
        text = text.replace(" ", "")
        text = text.replace("<0x0A>", "\n")
        text = text.replace("▁", " ")
    match = _HARMFUL_PATTERN.search(text)
    return match is not None and match.group(1).lower() == "yes"


@torch.inference_mode()
def classify_pii(text: str) -> bool:
    tokenizer, model = _get_model()
    prompt = _PROMPT_TEMPLATE.format(prompt=text, response="")
    input_ids = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    output = model.generate(input_ids=input_ids, max_new_tokens=32, pad_token_id=0, do_sample=False)
    result = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return _parse_result(result)


@torch.inference_mode()
def classify_pii_batch(texts: list[str]) -> list[bool]:
    tokenizer, model = _get_model()
    prompts = [_PROMPT_TEMPLATE.format(prompt=t, response="") for t in texts]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    max_len = input_ids.shape[-1]
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        pad_token_id=0,
        do_sample=False,
    )
    results = []
    for output in outputs:
        decoded = tokenizer.decode(output[max_len:], skip_special_tokens=True)
        results.append(_parse_result(decoded))
    return results


def wildguard_pii_detector(text: str) -> list:
    return guard_pii_detector(text, classify_pii)
