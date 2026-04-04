"""Qwen Guard PII detectors (Gen and Stream variants).

Gen models classify text as safe/unsafe at the message level. Stream models
provide per-token safety classification, enabling span-level PII detection
with character offset mapping.
"""
import logging
import re
import warnings

for _name in ("transformers", "huggingface_hub"):
    logging.getLogger(_name).setLevel(logging.ERROR)

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from detectors.guards.utils import (
    generate_and_decode,
    guard_pii_detector,
    pad_and_stack,
)
from logger import logger

QWEN_GUARD_GEN_MODELS = {
    "qwen-guard-gen-0.6b": "Qwen/Qwen3Guard-Gen-0.6B",
    "qwen-guard-gen-4b": "Qwen/Qwen3Guard-Gen-4B",
}

QWEN_GUARD_STREAM_MODELS = {
    "qwen-guard-stream-0.6b": "Qwen/Qwen3Guard-Stream-0.6B",
    "qwen-guard-stream-4b": "Qwen/Qwen3Guard-Stream-4B",
}

PII_CATEGORY = "PII"

_SAFETY_PATTERN = re.compile(r"Safety: (Safe|Unsafe|Controversial)")
_CATEGORY_PATTERN = re.compile(
    r"(Violent|Non-violent Illegal Acts|Sexual Content or Sexual Acts|PII|"
    r"Suicide & Self-Harm|Unethical Acts|Politically Sensitive Topics|"
    r"Copyright Violation|Jailbreak|None)"
)

_model_cache: dict = {}


def _prepare_rope(config):
    """Apply RoPE validation, handling the API change across transformers versions."""
    config.tie_word_embeddings = False
    if hasattr(config, "validate_rope"):
        config.validate_rope()
    elif hasattr(config, "standardize_rope_params"):
        config.standardize_rope_params()


def _get_gen_model(model_name: str):
    key = f"gen:{model_name}"
    if key not in _model_cache:
        model_id = QWEN_GUARD_GEN_MODELS[model_name]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            config = AutoConfig.from_pretrained(model_id)
            _prepare_rope(config)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                config=config,
                dtype="auto",
                device_map="auto",
            )
        _model_cache[key] = (tokenizer, model)
    return _model_cache[key]


def _get_stream_model(model_name: str):
    key = f"stream:{model_name}"
    if key not in _model_cache:
        model_id = QWEN_GUARD_STREAM_MODELS[model_name]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            _prepare_rope(config)
            if not hasattr(config, "pad_token_id") or config.pad_token_id is None:
                config.pad_token_id = tokenizer.pad_token_id or 0
            model = AutoModel.from_pretrained(
                model_id,
                config=config,
                device_map="auto",
                dtype=torch.bfloat16,
                trust_remote_code=True,
            ).eval()
        _model_cache[key] = (tokenizer, model)
    return _model_cache[key]


def _parse_gen_output(content: str) -> bool:
    """Check if Qwen Guard Gen output flags PII."""
    safety_match = _SAFETY_PATTERN.search(content)
    if safety_match and safety_match.group(1) in ("Unsafe", "Controversial"):
        categories = _CATEGORY_PATTERN.findall(content)
        return PII_CATEGORY in categories
    return False


def _find_content_token_range(token_ids_list: list[int], tokenizer) -> tuple[int, int]:
    """Find the start and end indices of user content tokens within the chat template."""
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    user_id = tokenizer.convert_tokens_to_ids("user")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # Find last <|im_start|>user pattern
    last_start = next(
        i for i in range(len(token_ids_list) - 1, -1, -1)
        if token_ids_list[i:i + 2] == [im_start_id, user_id]
    )
    # Content starts after <|im_start|>user\n (3 tokens: im_start, user, \n)
    content_start = last_start + 3
    # Content ends at the next <|im_end|>
    content_end = next(
        i for i in range(content_start, len(token_ids_list))
        if token_ids_list[i] == im_end_id
    )
    return content_start, content_end


def _build_spans_from_pii_tokens(
    pii_flags: list[bool],
    token_ids: list[int],
    content_start: int,
    tokenizer,
    text: str,
) -> list[dict]:
    """Group consecutive PII-flagged content tokens into spans with char offsets."""
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offset_mapping = encoding["offset_mapping"]
    content_token_ids = encoding["input_ids"]

    template_content_ids = token_ids[content_start:content_start + len(content_token_ids)]

    if template_content_ids != content_token_ids:
        logger.warning("Token alignment mismatch between chat template and raw text")
        if any(pii_flags):
            return [{"value": None, "start": None, "end": None, "type": "pii"}]
        return []

    spans = []
    i = 0
    while i < len(pii_flags):
        if pii_flags[i]:
            j = i
            while j < len(pii_flags) and pii_flags[j]:
                j += 1
            char_start = offset_mapping[i][0]
            char_end = offset_mapping[j - 1][1]
            spans.append({
                "value": text[char_start:char_end],
                "start": char_start,
                "end": char_end,
                "type": "pii",
            })
            i = j
        else:
            i += 1
    return spans


@torch.inference_mode()
def classify_pii_gen(text: str, model_name: str) -> bool:
    """Classify whether text contains PII using Qwen3Guard-Gen."""
    tokenizer, model = _get_gen_model(model_name)

    messages = [{"role": "user", "content": text}]
    model_inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", tokenize=True,
    ).to(model.device)

    generated_ids = model.generate(model_inputs, max_new_tokens=128)
    output_ids = generated_ids[0][model_inputs.shape[-1]:]
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    logger.debug(f"Qwen Guard Gen result: {content}")

    return _parse_gen_output(content)


@torch.inference_mode()
def classify_pii_gen_batch(texts: list[str], model_name: str) -> list[bool]:
    """Classify a batch of texts for PII using Qwen3Guard-Gen."""
    tokenizer, model = _get_gen_model(model_name)

    all_inputs = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": t}], return_tensors="pt", tokenize=True,
        )
        for t in texts
    ]
    padded, attention_mask = pad_and_stack(all_inputs, tokenizer.pad_token_id)
    return generate_and_decode(
        model, tokenizer, padded, attention_mask,
        max_new_tokens=128, parse_fn=_parse_gen_output, log_prefix="Qwen Guard Gen",
    )


@torch.inference_mode()
def detect_pii_stream(text: str, model_name: str) -> list[dict]:
    """Detect PII spans using Qwen3Guard-Stream per-token classification."""
    tokenizer, model = _get_stream_model(model_name)

    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    model_inputs = tokenizer(prompt, return_tensors="pt")
    token_ids = model_inputs.input_ids[0]
    token_ids_list = token_ids.tolist()

    content_start, content_end = _find_content_token_range(token_ids_list, tokenizer)
    n_content_tokens = content_end - content_start

    stream_state = None
    pii_flags = []

    if content_start > 0:
        _, stream_state = model.stream_moderate_from_ids(
            token_ids[:content_start], role="user", stream_state=None,
        )

    for i in range(content_start, content_end):
        result, stream_state = model.stream_moderate_from_ids(
            token_ids[i], role="user", stream_state=stream_state,
        )
        risk_level = result["risk_level"][-1]
        category = result.get("category", ["None"])[-1]
        is_pii = risk_level in ("Unsafe", "Controversial") and PII_CATEGORY in category
        pii_flags.append(is_pii)

    model.close_stream(stream_state)
    logger.debug(f"Qwen Guard Stream PII flags: {sum(pii_flags)}/{n_content_tokens} tokens")

    if not any(pii_flags):
        return []

    return _build_spans_from_pii_tokens(
        pii_flags, token_ids_list, content_start, tokenizer, text,
    )


def qwen_guard_gen_pii_detector(text: str, model_name: str = "qwen-guard-gen-4b") -> list:
    return guard_pii_detector(text, classify_pii_gen, model_name=model_name)


def qwen_guard_stream_pii_detector(
    text: str, model_name: str = "qwen-guard-stream-4b",
) -> list:
    if text is None:
        return []
    return detect_pii_stream(text, model_name)
