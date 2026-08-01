"""Attack loop: runs the adaptive attacker against the base Presidio detector
for a given PII value, tracking per-iteration results and stopping when the
detector is evaded.
"""
import yaml

from importlib.resources import files

from detectors.presidio import presidio_pii_analyzer

from data_manipulation.attacks.neural_prompt_to_prompt.adaptive_attacks.attacker import (
    AdaptiveAttacker,
)
from data_manipulation.attacks.neural_prompt_to_prompt.adaptive_attacks.const import (
    ADVERSARIAL_ATTACK_README_PATHS,
)


PROMPTS_PATH = files("data_manipulation.attacks.neural_prompt_to_prompt").joinpath("prompts.yaml")

with PROMPTS_PATH.open("r") as f:
    PROMPTS = yaml.safe_load(f)


def _init_attacker(pii: str, pii_type: str, attacker_awareness: str) -> AdaptiveAttacker:
    attacker = AdaptiveAttacker(
        pii=pii,
        pii_type=pii_type,
        system_prompt=PROMPTS["iterative_rewrite"],
    )

    if attacker_awareness == "attacks":
        attacker.append_readme_content_as_context(ADVERSARIAL_ATTACK_README_PATHS)

    return attacker


def _naive_detector(text: str) -> dict:
    return {"detected": int(len(presidio_pii_analyzer(text=text)) > 0)}


DETECTORS = {
    "naive": _naive_detector,
}


def run_attack(
    attack_id: int,
    pii: str,
    pii_type: str,
    max_iters: int,
    attacker_awareness: str,
):
    attacker = _init_attacker(pii, pii_type, attacker_awareness)

    done = {name: False for name in DETECTORS}
    trace = []

    for i in range(max_iters):
        text = attacker.craft_input()
        if not attacker.validate_pii_exists(adversarial_text=text):
            attacker.update_memory(text=text, verdict=-1)
            continue

        verdicts = []

        for detector_name, detector_fn in DETECTORS.items():
            if done[detector_name]:
                continue

            result = detector_fn(text)
            detected = result["detected"]
            verdicts.append(detected)

            trace.append({
                "attack_id": attack_id,
                "iteration": i + 1,
                "input": text,
                "pii": pii,
                "pii_type": pii_type,
                "attacker_awareness": attacker_awareness,
                "detector": detector_name,
                "detected": detected,
            })

            if detected == 0:
                done[detector_name] = True

        attacker.update_memory(
            text=text,
            verdict=min(verdicts) if verdicts else -1,
        )

        if all(done.values()):
            break

    return trace
