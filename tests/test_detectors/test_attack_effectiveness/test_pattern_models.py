"""Attack effectiveness against pattern-based detectors (Presidio, GLiNER).

These are lightweight — all models can run in the same process.
"""
import pytest

from .conftest import ATTACK_IDS, fmt, run_model

MODELS = [
    "presidio", "presidio-defend",
    "presidio-fuzzy", "presidio-fuzzy-defend",
    "gliner", "gliner-defend",
    "gliner-nvidia", "gliner-nvidia-defend",
]

_CACHE: dict[str, dict] = {}


def _get(model):
    if model not in _CACHE:
        _CACHE[model] = run_model(model)
    return _CACHE[model]


class TestBaseline:

    @pytest.mark.parametrize("model", MODELS)
    def test_detects_clean_pii(self, model):
        r = _get(model)["baseline"]
        if r is None:
            pytest.skip(f"{model}: failed to load")
        assert r is True, f"{model} missed PII in clean text"


class TestImmunity:
    """Content-level attacks should NOT bypass pattern-based detectors."""

    @pytest.mark.parametrize("attack", ATTACK_IDS)
    @pytest.mark.parametrize("model", MODELS)
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
