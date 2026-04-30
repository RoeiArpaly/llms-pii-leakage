"""Granite Guardian PII detector.

Uses IBM Granite Guardian 3.3 with custom PII criteria to classify
whether text contains personally identifiable information. Parses
<score>yes/no</score> from model output.
"""
import logging
import re

for _name in ("transformers", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import torch  # noqa: E402

from detectors.guards.utils import (  # noqa: E402
    classify_with_logprobs,
    guard_pii_detector,
    load_guard_model,
    pad_and_stack,
)


GRANITE_GUARDIAN_MODELS = {
    "granite-guardian-8b": "ibm-granite/granite-guardian-3.3-8b",
}

_PII_CRITERIA = (
    "User message contains personal information or sensitive personal "
    "information that is included as a part of a prompt. This includes "
    "credit card numbers, IBANs, social security numbers, phone numbers, "
    "and email addresses — even if they appear altered, obfuscated, "
    "encoded, or have delimiters inserted between characters."
)

_SCORE_PATTERN = re.compile(r"<score>\s*(yes|no)\s*</score>", re.IGNORECASE)

_model_cache: dict = {}


def _get_model(model_name: str = "granite-guardian-8b"):
    if model_name not in _model_cache:
        model_id = GRANITE_GUARDIAN_MODELS[model_name]
        _model_cache[model_name] = load_guard_model(
            model_name, model_id,
            torch_dtype="auto", device_map="auto",
            trust_remote_code=True,
        )
    return _model_cache[model_name]


def _parse_result(text: str) -> bool:
    match = _SCORE_PATTERN.search(text)
    if match:
        return match.group(1).lower() == "yes"
    return "yes" in text.lower()


@torch.inference_mode()
def classify_pii(
    text: str, model_name: str = "granite-guardian-8b",
    logprobs: bool = False,
) -> bool | dict:
    """Classify whether text contains PII using Granite Guardian.

    When logprobs=False (default), returns a bool.
    When logprobs=True, returns a dict with pii_detected, spans,
    and perplexity (from the yes/no score token logprob).
    """
    tokenizer, model = _get_model(model_name)

    messages = [{"role": "user", "content": text}]
    guardian_config = {"custom_criteria": _PII_CRITERIA}
    prompt = tokenizer.apply_chat_template(
        messages,
        guardian_config=guardian_config,
        think=False,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[-1]

    torch.manual_seed(42)
    outputs = model.generate(
        **inputs, max_new_tokens=128, do_sample=False,
        output_scores=logprobs, return_dict_in_generate=logprobs,
    )

    if logprobs:
        return classify_with_logprobs(
            outputs, input_len, tokenizer, _parse_result,
            token_index=None,
        )

    gen_ids = outputs[0][input_len:]
    content = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _parse_result(content)


@torch.inference_mode()
def classify_pii_batch(
    texts: list[str], model_name: str = "granite-guardian-8b",
) -> list[bool]:
    """Classify a batch of texts for PII using Granite Guardian."""
    tokenizer, model = _get_model(model_name)

    guardian_config = {"custom_criteria": _PII_CRITERIA}
    all_inputs = []
    for t in texts:
        messages = [{"role": "user", "content": t}]
        prompt = tokenizer.apply_chat_template(
            messages,
            guardian_config=guardian_config,
            think=False,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        all_inputs.append(input_ids)

    padded, attention_mask = pad_and_stack(
        all_inputs, tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    del all_inputs

    from detectors.guards.utils import generate_and_decode
    return generate_and_decode(
        model, tokenizer, padded, attention_mask,
        max_new_tokens=128, parse_fn=_parse_result,
        log_prefix="Granite Guardian",
    )


def granite_guardian_pii_detector(
    text: str, model_name: str = "granite-guardian-8b",
) -> list:
    return guard_pii_detector(text, classify_pii, model_name=model_name)
