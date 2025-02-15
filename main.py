from config import Config
from constants import DATASETS
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

    for model in Config.MODELS:
        pii_detection_pipeline(model=model)

    evaluate_and_save_datasets(datasets=DATASETS, models=Config.MODELS)

    logger.info("SUCCESS")
