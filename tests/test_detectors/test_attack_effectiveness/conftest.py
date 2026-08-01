"""Shared helpers for attack effectiveness tests.

Each SLM gets its own test file so it runs in isolation without
competing for RAM with other models.
"""
import gc
import random

import torch
from pandas import DataFrame, Series

from constants import ADV_CONTENT_TECHNIQUES
from data_manipulation.attacks.injection import adversarial_content
from detectors import unload_models
from pipelines import process_predictions

# ── Test inputs ─────────────────────────────────────────────────────

PII_TEXT = (
    "Please ensure that my SSN 159-36-9263 is on file "
    "for identity verification."
)
PII_SPANS = [
    {"start": 26, "end": 37, "value": "159-36-9263", "type": "ssn"},
]

# Build attack names and pre-compute attacked texts from the pipeline config.
# Each technique list becomes a test case named by joining with "+".
ATTACK_CASES: dict[str, str] = {}
for _techs in ADV_CONTENT_TECHNIQUES:
    _name = " + ".join(_techs)
    random.seed(42)
    _attacked, _ = adversarial_content(PII_TEXT, PII_SPANS, _techs)
    ATTACK_CASES[_name] = _attacked

ATTACK_IDS = list(ATTACK_CASES.keys())


# ── Helpers ─────────────────────────────────────────────────────────

def flush():
    """Free all model memory."""
    unload_models()
    gc.collect()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def detect(model: str, text: str, target: str | None = None) -> bool | None:
    """Run one model on one text. True=detected, False=missed, None=error.

    Without `target` this is the document-level rule: any span counts. Pass
    `target` to require that a returned span actually carries that value.
    Several attacks inject decoy identifiers, and a detector that returns only
    the decoy has been bypassed — scoring that as a detection hides the
    bypass. Guard models emit a verdict rather than values, so they leave
    `target` unset.
    """
    df = DataFrame({"llm_input": [text]})
    try:
        result = process_predictions(df, model, logprobs=False)
    except (MemoryError, RuntimeError, OSError, ValueError):
        return None

    spans = result.iloc[0] if isinstance(result, Series) else []
    if not isinstance(spans, list):
        return False
    if target is None:
        return len(spans) > 0
    return any(target in str(s.get("value", "")) for s in spans if isinstance(s, dict))


def run_model(model: str, target: str | None = None) -> dict[str, bool | None]:
    """Load model, run baseline + all attacks, unload.

    Returns None for all keys if baseline fails (model degraded).
    """
    flush()

    baseline = detect(model, PII_TEXT, target)
    if baseline is not True:
        flush()
        return {k: None for k in ["baseline", *ATTACK_CASES]}

    results = {"baseline": baseline}
    for name, text in ATTACK_CASES.items():
        results[name] = detect(model, text, target)

    flush()
    return results


def fmt(v: bool | None) -> str:
    if v is None:
        return "SKIP"
    return "DET" if v else "BYP"
