import yaml

from importlib.resources import files

from config import Config
from detectors.presidio_detector import presidio_pii_analyzer
from pii_shield import guard

from attacker import AdaptiveAttacker
from const import (
    ADVERSARIAL_ATTACK_README_PATHS,
    ADVERSARIAL_DEFENSE_README_PATHS,
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

    if attacker_awareness in {"attacks", "attacks_and_defenses"}:
        attacker.append_readme_content_as_context(ADVERSARIAL_ATTACK_README_PATHS)

    if attacker_awareness in {"defenses", "attacks_and_defenses"}:
        attacker.append_readme_content_as_context(ADVERSARIAL_DEFENSE_README_PATHS)
        attacker.system_prompt += (
            "\n\nThe detector applies a cascade of rule-based, ML-based (NER), "
            "LLM-based, and perplexity-based detection mechanisms."
        )

    return attacker


def _naive_detector(text: str) -> dict:
    return {"detected": int(len(presidio_pii_analyzer(text=text)) > 0)}


def _pii_shield_detector(text: str) -> dict:
    result = guard(text=text, perplexity_threshold=Config.PERPLEXITY_THRESHOLD)
    return {
        "detected": int(result["detected"]),
        "pii_shield_detector": result.get("detector", None),
        "pii_shield_perplexity": result.get("perplexity", None),
    }


DETECTORS = {
    "naive": _naive_detector,
    "pii_shield": _pii_shield_detector,
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
                "pii_shield_detector": result.get("pii_shield_detector", None),
                "pii_shield_perplexity": result.get("pii_shield_perplexity", None),
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
