"""Llama 3.2 1B Instruct PII detector.

Binary PII classifier with a high-precision confidence gate: flags only
when the first generated token is a confident ``yes``-variant
(P >= 0.5). Refusals (``I can't…``), hedged ``Yes`` answers with low
probability, and ``no`` responses all return False. Perplexity is
``1 / P(top-1)``. Batch and single-text paths share one implementation
so results are identical.
"""
import logging
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
    "Scan the text for PII. Report 'yes' if you are confident that "
    "any of these appear: credit card, SSN, phone number, email, "
    "IBAN, bank account. Report 'no' otherwise. Respond only with "
    "'yes' or 'no'.\n\n"
    "Text: {}\n\n"
    "Answer:"
)

_CONFIDENCE_THRESHOLD = 0.5
_YES_VARIANTS = ("yes",)
_PII_SPAN = {"value": None, "start": None, "end": None, "type": "pii"}

_model_cache: dict = {}
_yes_ids_cache: dict = {}


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


def _yes_token_ids(tokenizer, model_name: str) -> set[int]:
    if model_name not in _yes_ids_cache:
        ids: set[int] = set()
        for base in _YES_VARIANTS:
            for case_v in (base.lower(), base.title(), base.upper()):
                for s in (case_v, " " + case_v):
                    toks = tokenizer.encode(s, add_special_tokens=False)
                    if len(toks) == 1:
                        ids.add(toks[0])
        _yes_ids_cache[model_name] = ids
    return _yes_ids_cache[model_name]


def _prompt_ids(tokenizer, text: str) -> torch.Tensor:
    messages = [{"role": "user", "content": _USER_TEMPLATE.format(text)}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    return tokenizer(prompt, return_tensors="pt").input_ids[0]


def _strict_yes_gate(scores_0: torch.Tensor, yes_ids: set[int]) -> list[dict]:
    """Apply the strict-yes gate row-wise to a [batch, vocab] logits tensor."""
    probs = F.softmax(scores_0, dim=-1)
    top_vals, top_ids = probs.max(dim=-1)
    results = []
    for i in range(top_ids.shape[0]):
        tid = int(top_ids[i].item())
        p = float(top_vals[i].item())
        pii_detected = tid in yes_ids and p >= _CONFIDENCE_THRESHOLD
        results.append({
            "pii_detected": pii_detected,
            "spans": [_PII_SPAN.copy()] if pii_detected else [],
            "perplexity": 1.0 / max(p, 1e-12),
        })
    return results


@torch.inference_mode()
def classify_pii_batch_full(
    texts: list[str], model_name: str = "llama-3.2-1b",
) -> list[dict]:
    """True-batch PII classification with strict-yes gate.

    Returns one dict per text with ``pii_detected``, ``spans``, and
    ``perplexity``. This is the canonical implementation — single-text
    and bool-only wrappers reduce to a batch-of-1 / list-comprehension
    over this.
    """
    tokenizer, model = _get_model(model_name)
    yes_ids = _yes_token_ids(tokenizer, model_name)

    padded, attention_mask = pad_and_stack(
        [_prompt_ids(tokenizer, t) for t in texts],
        tokenizer.pad_token_id,
    )

    torch.manual_seed(42)
    outputs = model.generate(
        input_ids=padded.to(model.device),
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=4,
        pad_token_id=tokenizer.pad_token_id or 0,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

    results = _strict_yes_gate(outputs.scores[0], yes_ids)
    del outputs
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return results


def classify_pii(
    text: str, model_name: str = "llama-3.2-1b",
    logprobs: bool = False,
) -> bool | dict:
    """Classify a single text for PII.

    When ``logprobs=True`` returns a dict (with ``perplexity``); else bool.
    Routes through :func:`classify_pii_batch_full` as a batch of one so
    single and batch paths share identical gate logic.
    """
    result = classify_pii_batch_full([text], model_name=model_name)[0]
    return result if logprobs else result["pii_detected"]


def classify_pii_batch(
    texts: list[str], model_name: str = "llama-3.2-1b",
) -> list[bool]:
    """Bool-only wrapper around :func:`classify_pii_batch_full`."""
    return [
        r["pii_detected"]
        for r in classify_pii_batch_full(texts, model_name=model_name)
    ]


def llama_pii_detector(
    text: str, model_name: str = "llama-3.2-1b",
) -> list:
    return guard_pii_detector(text, classify_pii, model_name=model_name)
