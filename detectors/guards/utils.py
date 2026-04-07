"""Shared utilities for guard-model PII detectors (Llama Guard, Qwen Guard, etc.).

Provides single-text and batched wrappers that convert a binary classifier
(text -> bool) into span-list output compatible with the evaluation pipeline.
Also provides common batch-inference helpers (padding, generation, decoding)
used across all guard model implementations.
"""
import logging
import os
import warnings
from typing import Callable

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Silence noisy third-party loggers and tqdm bars from polluting CLI output.
for _name in ("transformers", "huggingface_hub", "accelerate", "torch"):
    logging.getLogger(_name).setLevel(logging.ERROR)

from huggingface_hub.utils import disable_progress_bars  # noqa: E402
disable_progress_bars()

import torch  # noqa: E402
from pandas import Series  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from logger import logger  # noqa: E402


_model_cache: dict = {}


def load_guard_model(
    model_name: str,
    model_id: str,
    model_cls=AutoModelForCausalLM,
    tokenizer_cls=AutoTokenizer,
    **model_kwargs,
) -> tuple:
    """Load and cache a guard model + tokenizer pair."""
    if model_name not in _model_cache:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tokenizer = tokenizer_cls.from_pretrained(model_id, **{
                k: v for k, v in model_kwargs.items()
                if k in ("trust_remote_code", "use_fast")
            })
            remaining = {
                k: v for k, v in model_kwargs.items()
                if k not in ("trust_remote_code", "use_fast")
            }
            model = model_cls.from_pretrained(model_id, **remaining)

        # Fail fast if parameters were offloaded to disk — inference
        # would be orders of magnitude slower than expected.
        on_meta = sum(
            p.numel() for p in model.parameters()
            if p.device.type == "meta"
        )
        if on_meta > 0:
            total = sum(p.numel() for p in model.parameters())
            pct = on_meta / total * 100
            del model
            raise MemoryError(
                f"{model_name}: {pct:.0f}% of parameters offloaded to "
                f"disk — not enough memory to run this model"
            )

        _model_cache[model_name] = (tokenizer, model)
    return _model_cache[model_name]


def _to_1d_tensor(ids) -> torch.Tensor:
    """Normalize token IDs to a 1-D tensor.

    Accepts a raw tensor (1-D or 2-D) or a BatchEncoding with .input_ids.
    """
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    if ids.dim() == 2:
        ids = ids[0]
    return ids


def pad_and_stack(
    token_id_tensors: list,
    pad_token_id: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a list of token-ID tensors and build an attention mask.

    Each element can be a 1-D tensor, a 2-D (1, seq_len) tensor, or a
    BatchEncoding with an .input_ids attribute.
    """
    tensors = [_to_1d_tensor(ids) for ids in token_id_tensors]
    max_len = max(t.shape[0] for t in tensors)
    pad_id = pad_token_id or 0
    n = len(tensors)
    padded = torch.full((n, max_len), pad_id, dtype=tensors[0].dtype)
    attention_mask = torch.zeros(n, max_len, dtype=torch.long)
    for i, t in enumerate(tensors):
        length = t.shape[0]
        padded[i, max_len - length:] = t
        attention_mask[i, max_len - length:] = 1
    return padded, attention_mask


@torch.inference_mode()
def generate_and_decode(
    model,
    tokenizer_or_processor,
    padded: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    parse_fn: Callable[[str], object],
    log_prefix: str = "Guard",
) -> list:
    """Run model.generate on a padded batch, decode each output, apply parse_fn."""
    pad_id = getattr(tokenizer_or_processor, "pad_token_id", None)
    if pad_id is None and hasattr(tokenizer_or_processor, "tokenizer"):
        pad_id = getattr(tokenizer_or_processor.tokenizer, "pad_token_id", None)
    pad_id = pad_id or 0

    padded = padded.to(model.device)
    attention_mask = attention_mask.to(model.device)
    max_len = padded.shape[-1]

    torch.manual_seed(42)
    outputs = model.generate(
        input_ids=padded,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id,
        do_sample=False,
    )

    decode = (
        tokenizer_or_processor.decode
        if hasattr(tokenizer_or_processor, "decode")
        else tokenizer_or_processor.tokenizer.decode
    )

    results = []
    for output in outputs:
        text = decode(output[max_len:], skip_special_tokens=True).strip()
        logger.debug(f"{log_prefix} result: {text}")
        results.append(parse_fn(text))

    del outputs, padded, attention_mask
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return results


def guard_pii_detector(text: str, classifier: Callable, **kwargs) -> list:
    """Guard-based PII detector for message-level classifiers.

    Returns a single span with no value/position since these models
    classify at message level and cannot extract token-level PII.
    """
    if text is None:
        return []
    if not classifier(text, **kwargs):
        return []
    return [{"value": None, "start": None, "end": None, "type": "pii"}]


def guard_pii_detector_batch(
    data: Series,
    classifier_batch: Callable,
    batch_size: int = 32,
    **kwargs,
) -> Series:
    """Batched guard-based PII detector for message-level classifiers."""
    texts = data.tolist()
    all_results: list[list] = [[] for _ in range(len(texts))]

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        valid_indices = [
            i for i, t in enumerate(batch_texts) if t is not None
        ]
        valid_texts = [batch_texts[i] for i in valid_indices]

        if not valid_texts:
            continue

        flags = classifier_batch(valid_texts, **kwargs)
        for local_idx, flag in zip(valid_indices, flags):
            if flag:
                all_results[start + local_idx] = [
                    {"value": None, "start": None, "end": None, "type": "pii"},
                ]
        del flags

    return Series(all_results, index=data.index)
