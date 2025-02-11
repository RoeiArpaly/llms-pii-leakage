from config import Config
from logger import logger
from pipelines import (
    generate_baseline_dataset,
    generate_fuzzy_dataset,
    generate_fuzzy_adv_dataset,
    pii_detector_presidio,
    pii_detector_llm,
)


if __name__ == "__main__":
    generate_baseline_dataset(n_samples=Config.NUMBER_OF_SAMPLES, pii_proba=Config.PII_PROBABILITY)
    generate_fuzzy_dataset()
    generate_fuzzy_adv_dataset()

    pii_detector_presidio()
    pii_detector_llm()

    logger.info("SUCCESS")
