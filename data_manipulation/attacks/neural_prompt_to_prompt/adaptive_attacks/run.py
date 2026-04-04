"""CLI runner for the adaptive attack experiment.

Generates random PII values and runs the adaptive attack loop across all
attacker awareness levels, writing per-attack traces to a CSV file.
"""
import contextlib
import os

from random import choice

from constants import PII_ENTITIES
from data_generation.pii_generator import get_faker
from utils import csv_batch_writer

from data_manipulation.attacks.neural_prompt_to_prompt.adaptive_attacks.loop import run_attack


class Config:
    NUMBER_OF_ATTACKS: int = 100
    MAX_ITERS_PER_ATTACK: int = 20
    ATTACKER_AWARENESS: list = ["none", "attacks", "defenses", "attacks_and_defenses"]
    FILENAME: str = "adaptive_attack_results_01.csv"


def main():
    with contextlib.redirect_stdout(open(os.devnull, "w")), \
            contextlib.redirect_stderr(open(os.devnull, "w")):
        faker = get_faker()

    for attacker_awareness in Config.ATTACKER_AWARENESS:
        for attack_id in range(1, Config.NUMBER_OF_ATTACKS + 1):

            pii_type = choice(list(PII_ENTITIES.values()))
            faker._sentence_templates = [f"{{{{{pii_type}}}}}"]
            with contextlib.redirect_stdout(open(os.devnull, "w")), \
                    contextlib.redirect_stderr(open(os.devnull, "w")):
                samples = faker.generate_new_fake_sentences(num_samples=1)
            pii = samples[0].full_text

            trace = run_attack(
                attack_id=attack_id,
                pii=pii,
                pii_type=pii_type,
                max_iters=Config.MAX_ITERS_PER_ATTACK,
                attacker_awareness=attacker_awareness,
            )

            csv_batch_writer(
                batch=trace,
                filename=Config.FILENAME,
            )


if __name__ == "__main__":
    main()
