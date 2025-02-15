from config import Config
from logger import logger
from pipelines import (
    generate_baseline_dataset,
    generate_fuzzy_dataset,
    generate_fuzzy_adv_dataset,
    pii_detection_pipeline,
)
from evaluation.reports import evaluate_and_save_datasets


if __name__ == "__main__":
    generate_baseline_dataset(n_samples=Config.NUMBER_OF_SAMPLES, pii_proba=Config.PII_PROBABILITY)
    generate_fuzzy_dataset()
    generate_fuzzy_adv_dataset()
    pii_detection_pipeline(models=Config.MODELS)
    evaluate_and_save_datasets(models=Config.MODELS)

    logger.info("SUCCESS")
