"""Attack effectiveness against lightweight detectors (Presidio, GLiNER).

These are lightweight — all models can run in the same process.
"""
import pytest

from .conftest import ATTACK_IDS, PII_SPANS, fmt, run_model

# Immunity to content-level attacks follows from *how* a detector reads, not
# from it being lightweight. Regex matches the value itself, which content
# attacks never touch, so Presidio is immune by construction. Transformer NER
# reads the surrounding context, and `supportive_context` rewrites exactly the
# cue words it relies on — so a bypass there is the expected finding, not a
# regression. Both families stay in the summary table; only the pattern-based
# ones carry the immunity assertion.
PATTERN_MODELS = [
    "presidio",
    "presidio-fuzzy",
]
CONTEXTUAL_MODELS = [
    "gliner",
    "gliner-nvidia",
]
MODELS = PATTERN_MODELS + CONTEXTUAL_MODELS

TARGET = PII_SPANS[0]["value"]

_CACHE: dict[str, dict] = {}


def _get(model):
    if model not in _CACHE:
        # Score against the target value, not "any span": pi_few_shot_safe
        # injects decoy identifiers that a detector can return while missing
        # the real PII entirely, which would otherwise read as a detection.
        _CACHE[model] = run_model(model, target=TARGET)
    return _CACHE[model]


class TestBaseline:

    @pytest.mark.parametrize("model", MODELS)
    def test_detects_clean_pii(self, model):
        r = _get(model)["baseline"]
        if r is None:
            pytest.skip(f"{model}: failed to load")
        assert r is True, f"{model} missed PII in clean text"


class TestImmunity:
    """Content-level attacks must not bypass pattern-based detectors.

    Scoped to PATTERN_MODELS deliberately — see the note at the top of the
    module for why contextual NER is excluded rather than expected to pass.
    """

    @pytest.mark.parametrize("attack", ATTACK_IDS)
    @pytest.mark.parametrize("model", PATTERN_MODELS)
    def test_still_detects(self, model, attack):
        r = _get(model).get(attack)
        if r is None:
            pytest.skip(f"{model}: failed to load")
        assert r is True, f"{model} bypassed by {attack}"


class TestSummary:

    def test_print_table(self):
        header = f"{'Model':30s}  {'Base':>4s}"
        for a in ATTACK_IDS:
            header += f"  {a[:6]:>6s}"
        print("\n" + header)
        print("-" * len(header))
        for model in MODELS:
            r = _get(model)
            row = f"{model:30s}  {fmt(r.get('baseline')):>4s}"
            for a in ATTACK_IDS:
                row += f"  {fmt(r.get(a)):>6s}"
            print(row)
