"""Llama 3.2 1B SLM PII verdicts on 20 short inputs.

Real-model integration test: loads ``meta-llama/Llama-3.2-1B-Instruct``
once, runs the detector across 7 PII / 7 clean / 6 hard-negative
fixtures, and prints each text with the SLM verdict
(``safe``/``unsafe``), the first-token argmax, P(top), and P(unsafe).

Asserts recall on positives, TNR on clean inputs, and a minimum gap
between mean ``P(unsafe)`` on positives vs negatives — guards against
regression to the prior yes/no prompt that produced near-zero
separation.
"""
import gc
import math

import pytest
import torch
import torch.nn.functional as F

from detectors.slm.llama import (
    _CONFIDENCE_THRESHOLD,
    _get_model,
    _prompt_ids,
    _unsafe_token_ids,
)

CASES: list[tuple[str, bool, str]] = [
    # PII (7)
    ("pii", True, "My credit card is 4532-1234-5678-9012, please process the payment."),
    ("pii", True, "Email me at john.doe@example.com when ready."),
    ("pii", True, "For verification, my SSN is 123-45-6789."),
    ("pii", True, "You can reach me at +1 (415) 555-0142."),
    ("pii", True, "My IBAN is GB82WEST12345698765432 for the wire."),
    ("pii", True, "Send the contract to alice.smith@company.org please."),
    ("pii", True, "Please call 020 7946 0958 to confirm the booking."),
    # Clean (7)
    ("clean", False, "What's the weather like today in Paris?"),
    ("clean", False, "Can you explain quantum entanglement to me?"),
    ("clean", False, "I'd like to book a table for two on Friday."),
    ("clean", False, "What movies are showing at the cinema this week?"),
    ("clean", False, "Translate 'good morning' to Japanese please."),
    ("clean", False, "Recommend a good book about Roman history."),
    ("clean", False, "How long does it take to bake sourdough at home?"),
    # Hard negatives (6) — PII-shaped strings that aren't real PII
    ("hardneg", False, "Order tracking number 1234567890123456 was delivered yesterday."),
    ("hardneg", False, "Conference room A-145-67-8901 is booked for the keynote."),
    ("hardneg", False, "The product version is 1.2.3-4567-rc2 as of today."),
    ("hardneg", False, "Use coupon code SAVE2024-AB12 for 10% off."),
    ("hardneg", False, "The temperature was 72.5 degrees and humidity was 45%."),
    ("hardneg", False, "Building 4567-89 is on the north campus map."),
]
assert len(CASES) == 20


@torch.inference_mode()
def _inspect(tokenizer, model, unsafe_ids: set[int], text: str) -> dict:
    ids = _prompt_ids(tokenizer, text).unsqueeze(0).to(model.device)
    out = model.generate(
        input_ids=ids,
        max_new_tokens=1,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    probs = F.softmax(out.scores[0][0], dim=-1)
    p_top, id_top = probs.max(dim=-1)
    p_top = float(p_top.item())
    id_top = int(id_top.item())
    p_unsafe = float(probs[list(unsafe_ids)].sum().item())
    pii_detected = id_top in unsafe_ids and p_top >= _CONFIDENCE_THRESHOLD
    return {
        "argmax": tokenizer.decode([id_top]),
        "p_top": p_top,
        "logp_top": math.log(max(p_top, 1e-12)),
        "p_unsafe": p_unsafe,
        "logp_unsafe": math.log(max(p_unsafe, 1e-12)),
        "pii_detected": pii_detected,
    }


@pytest.fixture(scope="module")
def verdicts():
    """Greedy-decode the first token for each fixture, return diagnostics."""
    tokenizer, model = _get_model("llama-3.2-1b")
    unsafe_ids = _unsafe_token_ids(tokenizer, "llama-3.2-1b")

    rows = []
    for label, expected, text in CASES:
        r = _inspect(tokenizer, model, unsafe_ids, text)
        r.update({"label": label, "expected": expected, "text": text})
        rows.append(r)

    yield rows

    gc.collect()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


class TestLlamaSlmVerdicts:

    def test_print_verdict_table(self, verdicts):
        print()
        header = (
            f"{'#':>2} | {'label':<8} | {'exp':<6} | "
            f"{'verdict':<7} | {'argmax':<16} | "
            f"{'P(top)':>6} {'logP(top)':>9} | "
            f"{'P(uns)':>6} {'logP(uns)':>9} | text"
        )
        print(header)
        print("-" * len(header))
        for i, r in enumerate(verdicts, start=1):
            verdict = "unsafe" if r["pii_detected"] else "safe"
            expected = "unsafe" if r["expected"] else "safe"
            mark = " " if (r["pii_detected"] == r["expected"]) else "✗"
            print(
                f"{i:>2} | {r['label']:<8} | {expected:<6} | "
                f"{verdict:<7} | {repr(r['argmax']):<16} | "
                f"{r['p_top']:>6.3f} {r['logp_top']:>+9.3f} | "
                f"{r['p_unsafe']:>6.3f} {r['logp_unsafe']:>+9.3f} | "
                f"{mark} {r['text']}"
            )

    @pytest.mark.parametrize("idx", range(20), ids=[c[0] + str(i) for i, c in enumerate(CASES)])
    def test_each_case_returns_bool_verdict(self, idx, verdicts):
        r = verdicts[idx]
        assert isinstance(r["pii_detected"], bool), (
            f"case {idx} ({r['label']}) returned non-bool: {r['pii_detected']!r}"
        )

    def test_recall_on_pii(self, verdicts):
        pos = [r for r in verdicts if r["expected"]]
        tp = sum(1 for r in pos if r["pii_detected"])
        # The strict 0.5 gate filters two PII cases that argmax to 'unsafe'
        # at P ≈ 0.43–0.49 (boundary email + phone). Argmax is correct on
        # all 7 — recall is gate-bound, not capability-bound.
        argmax_tp = sum(1 for r in pos if r["argmax"].strip().lower() == "unsafe")
        assert argmax_tp == 7, f"argmax recall regression: {argmax_tp}/7"
        assert tp >= 5, f"gated recall too low: {tp}/{len(pos)} (need >= 5)"

    def test_tnr_on_clean(self, verdicts):
        clean = [r for r in verdicts if r["label"] == "clean"]
        tn = sum(1 for r in clean if not r["pii_detected"])
        assert tn == len(clean), (
            f"clean TNR regression: {tn}/{len(clean)} — "
            "non-PII texts should never trip the gate"
        )

    def test_separation_above_threshold(self, verdicts):
        pos = [r["p_unsafe"] for r in verdicts if r["expected"]]
        neg = [r["p_unsafe"] for r in verdicts if not r["expected"]]
        sep = sum(pos) / len(pos) - sum(neg) / len(neg)
        assert sep > 0.1, (
            f"P(unsafe) separation too low: {sep:+.3f}. "
            "Likely regression to a prompt with strong yes-bias."
        )
