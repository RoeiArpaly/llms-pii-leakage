import json

from random import choice

from constants import PII_ENTITIES
from data_generation.pii_generator import get_data_generator
from utils import csv_batch_writer

from loop import run_attack


class Config:
    NUMBER_OF_ATTACKS: int = 100
    MAX_ITERS_PER_ATTACK: int = 20
    ATTACKER_AWARENESS: list = ["none", "attacks", "defenses", "attacks_and_defenses"]
    FILENAME: str = "adaptive_attack_results_01.csv"


def main():
    data_generator = get_data_generator()

    for attacker_awareness in Config.ATTACKER_AWARENESS:
        for attack_id in range(1, Config.NUMBER_OF_ATTACKS + 1):

            pii_type = choice(list(PII_ENTITIES.values()))
            fake_pii = data_generator.generate_fake_data(
                templates=["{{" + pii_type + "}}"],
                n_samples=1,
            )
            pii = json.loads(list(fake_pii)[0].toJSON())["fake"]

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
