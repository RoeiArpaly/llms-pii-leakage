"""Llama 3.2 1B Instruct PII detector.

Uses Llama-3.2-1B-Instruct as a binary PII classifier. The model is
prompted to answer "yes" or "no" whether the text contains PII, and
the classification token's logprob is used for perplexity scoring.
Supports batched inference for optimal throughput.
"""
import logging
import math
import warnings

for _name in ("transformers", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
)

from detectors.guards.utils import (  # noqa: E402
    guard_pii_detector,
    pad_and_stack,
)
from logger import logger  # noqa: E402

LLAMA_SLM_MODELS = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
}

_USER_TEMPLATE = (
    "Does this text contain personal data? "
    "Answer yes or no.\n\nText: {}"
)

_model_cache: dict = {}


def _get_model(model_name: str = "llama-3.2-1b"):
    if model_name not in _model_cache:
        model_id = LLAMA_SLM_MODELS[model_name]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype="auto", device_map="auto",
            )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(f"Loaded {model_name} ({model_id})")
        _model_cache[model_name] = (tokenizer, model)
    return _model_cache[model_name]


def _parse_result(text: str) -> bool:
    return "yes" in text.strip().lower()


@torch.inference_mode()
def classify_pii(
    text: str, model_name: str = "llama-3.2-1b",
    logprobs: bool = False,
) -> bool | dict:
    """Classify whether text contains PII.

    When logprobs=True, returns dict with pii_detected, spans, and
    perplexity (from the yes/no classification token).
    """
    tokenizer, model = _get_model(model_name)

    messages = [{"role": "user", "content": _USER_TEMPLATE.format(text)}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[-1]

    torch.manual_seed(42)
    outputs = model.generate(
        **inputs, max_new_tokens=8, do_sample=False,
        output_scores=logprobs, return_dict_in_generate=logprobs,
    )

    if logprobs:
        gen_ids = outputs.sequences[0][input_len:]
        content = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pii_detected = _parse_result(content)

        perplexity = 1.0
        if outputs.scores:
            probs = F.softmax(outputs.scores[0], dim=-1)
            token_id = gen_ids[0]
            perplexity = math.exp(-torch.log(probs[0, token_id]).item())

        spans = []
        if pii_detected:
            spans = [
                {"value": None, "start": None, "end": None, "type": "pii"},
            ]
        del outputs
        return {
            "pii_detected": pii_detected,
            "spans": spans,
            "perplexity": perplexity,
        }

    gen_ids = outputs[0][input_len:]
    content = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return _parse_result(content)


@torch.inference_mode()
def classify_pii_batch(
    texts: list[str], model_name: str = "llama-3.2-1b",
) -> list[bool]:
    """Classify a batch of texts for PII."""
    tokenizer, model = _get_model(model_name)

    all_inputs = []
    for t in texts:
        messages = [{"role": "user", "content": _USER_TEMPLATE.format(t)}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
        all_inputs.append(ids)

    padded, attention_mask = pad_and_stack(
        all_inputs, tokenizer.pad_token_id,
    )
    del all_inputs

    from detectors.guards.utils import generate_and_decode
    return generate_and_decode(
        model, tokenizer, padded, attention_mask,
        max_new_tokens=8, parse_fn=_parse_result,
        log_prefix="Llama 3.2 1B",
    )


def llama_pii_detector(
    text: str, model_name: str = "llama-3.2-1b",
) -> list:
    return guard_pii_detector(text, classify_pii, model_name=model_name)
