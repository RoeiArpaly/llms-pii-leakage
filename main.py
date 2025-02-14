from config import Config
from logger import logger
from pipelines import (
    generate_baseline_dataset,
    generate_fuzzy_dataset,
    generate_fuzzy_adv_dataset,
    pii_detector_presidio,
    pii_detector_llm,
)
from evaluation.reports import evaluate_and_save_datasets


if __name__ == "__main__":
    generate_baseline_dataset(n_samples=Config.NUMBER_OF_SAMPLES, pii_proba=Config.PII_PROBABILITY)
    generate_fuzzy_dataset()
    generate_fuzzy_adv_dataset()

    pii_detector_presidio()
    pii_detector_presidio(nlp=True)
    pii_detector_llm()

    evaluate_and_save_datasets(
        datasets=["baseline", "fuzzy", "fuzzy_adv"],
        models=["presidio", "presidio_nlp", "gpt-4o-mini"],
    )

    logger.info("SUCCESS")
