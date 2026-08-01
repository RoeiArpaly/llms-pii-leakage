"""Attack effectiveness against Qwen Guard 4B.

Run independently to avoid OOM:
    uv run pytest tests/test_detectors/test_attack_effectiveness/test_qwen_guard_4b.py -s -v
"""
import pytest

from .conftest import ATTACK_IDS, fmt, run_model

MODELS = ["qwen-guard-4b"]

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


class TestAttacks:

    @pytest.mark.parametrize("attack", ATTACK_IDS)
    @pytest.mark.parametrize("model", MODELS)
    def test_bypass_status(self, model, attack):
        r = _get(model).get(attack)
        if r is None:
            pytest.skip(f"{model}: failed to load")
        status = "DETECTED" if r else "BYPASSED"
        print(f"\n  {attack} + {model}: {status}")


class TestSummary:

    def test_print_table(self):
        header = f"{'Model':35s}  {'Base':>4s}"
        for a in ATTACK_IDS:
            header += f"  {a[:6]:>6s}"
        print("\n" + header)
        print("-" * len(header))
        for model in MODELS:
            r = _get(model)
            row = f"{model:35s}  {fmt(r.get('baseline')):>4s}"
            for a in ATTACK_IDS:
                row += f"  {fmt(r.get(a)):>6s}"
            print(row)
