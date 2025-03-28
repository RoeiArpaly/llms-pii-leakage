import random

from pandas import (
    concat,
    DataFrame,
    read_csv,
    Series,
)

from constants import (
    ADV_CONTENT_TECHNIQUES,
    BASELINE_DATASET_COLS,
    DATASETS,
    FUZZY_TECHNIQUES,
    FUZZY_DATASET_COLS,
    FUZZY_ADV_DATASET_COLS,
)
from data_generation.pii_generator import presidio_inject_pii
from data_generation.llm_input_generator import generate_llm_input
from data_manipulation.rule_based import (
    adversarial_content,
    pii_fuzzer,
)
from data_manipulation.defenses.fuzzy_match import (
    fuzzy_pii_analyzer,
    fuzzy_recognizers_setup,
)
from data_manipulation.defenses.preprocess import defensive_preprocess
from detectors.gliner_detector import gliner_pii_detector
from detectors.llm_detector import llm_pii_detector
from detectors.presidio_detector import presidio_pii_analyzer
from evaluation import spans_set
from logger import logger
from utils import (
    cast_to_json,
    infer_json,
    parallel_apply,
)


def generate_baseline_dataset(n_samples: int, pii_proba: float, save_every_n: int = 100):
    results = []
    for i in range(n_samples):
        logger.info(f"Generating LLM input sample {i + 1}/{n_samples}")
        contains_pii = random.random() < pii_proba
        llm_input = generate_llm_input(contains_pii=contains_pii)
        fake_record = presidio_inject_pii(text=llm_input) if contains_pii else {"spans": []}

        results.append({
            "llm_input": fake_record["text"] if contains_pii else llm_input,
            "pii_spans": fake_record["spans"] if contains_pii else [],
        })
        if (i + 1) % save_every_n == 0 or i + 1 == n_samples:
            data = DataFrame(results)
            if i + 1 == n_samples:
                data = data.sort_values(
                    by="pii_spans",
                    key=lambda spans: spans.str.len().astype(bool),
                    ascending=False,
                )
                data["llm_input_defend"] = data["llm_input"].apply(defensive_preprocess)
                data = data[BASELINE_DATASET_COLS]
                data = data.reset_index(drop=True)
            data.index.name = "uid"
            data.apply(cast_to_json).to_csv(path_or_buf="datasets/baseline_dataset.csv", index=True)
            logger.info(f"LLM input generation results saved at sample {i + 1}")
    logger.info("LLM input generation completed successfully")


def generate_fuzzy_dataset():
    data = read_csv("datasets/baseline_dataset.csv").apply(infer_json)
    data = data[data["pii_spans"].apply(len) > 0].copy().reset_index(drop=True)
    data = data.rename(columns={"uid": "input_id"})
    datasets = []
    for technique in FUZZY_TECHNIQUES:
        logger.info(f"Generating fuzzy content for technique: {technique}")
        _data = data.copy()
        _data["fuzzy_techniques"] = _data.apply(lambda _: technique, axis=1)
        _data[["llm_input", "pii_spans"]] = _data.apply(
            lambda row: pii_fuzzer(
                llm_input=row["llm_input"],
                spans=row["pii_spans"],
                chosen_techniques=row["fuzzy_techniques"],
            ),
            axis=1,
            result_type="expand",
        )
        datasets.append(_data.copy())
    data = concat(datasets, ignore_index=True)
    data["llm_input_defend"] = data["llm_input"].apply(defensive_preprocess)
    data = data[FUZZY_DATASET_COLS]
    data.index.name = "uid"
    data.apply(cast_to_json).to_csv(path_or_buf="datasets/fuzzy_dataset.csv", index=True)


def generate_fuzzy_adv_dataset():
    data = read_csv("datasets/fuzzy_dataset.csv").apply(infer_json)
    data = data.drop(columns=["uid"])
    datasets = []
    for technique in ADV_CONTENT_TECHNIQUES:
        logger.info(f"Generating adversarial content for technique: {technique}")
        _data = data.copy()
        _data["adv_content_techniques"] = _data.apply(lambda _: technique, axis=1)
        _data[["llm_input", "pii_spans"]] = _data.apply(
            lambda row: adversarial_content(
                llm_input=row["llm_input"],
                spans=row["pii_spans"],
                chosen_techniques=row["adv_content_techniques"],
            ),
            axis=1,
            result_type="expand",
        )
        datasets.append(_data.copy())
    data = concat(datasets, ignore_index=True)
    data["llm_input_defend"] = data["llm_input"].apply(defensive_preprocess)
    data = data[FUZZY_ADV_DATASET_COLS]
    data.index.name = "uid"
    data.apply(cast_to_json).to_csv(path_or_buf="datasets/fuzzy_adv_dataset.csv", index=True)


def process_predictions(data: DataFrame, model: str, dataset: str) -> Series:
    """Apply the appropriate PII detection model to the dataset."""
    logger.info(f"Detecting PII with {model} for {dataset} dataset")

    if model == "presidio":
        prediction = data["llm_input"].apply(presidio_pii_analyzer)
    elif model == "presidio-defend":
        prediction = data["llm_input_defend"].apply(presidio_pii_analyzer)
    elif model == "presidio-fuzzy":
        recognizers = fuzzy_recognizers_setup()
        prediction = data["llm_input"].apply(fuzzy_pii_analyzer, recognizers=recognizers)
    elif model == "presidio-fuzzy-defend":
        recognizers = fuzzy_recognizers_setup()
        prediction = data["llm_input_defend"].apply(fuzzy_pii_analyzer, recognizers=recognizers)
    elif model == "gliner":
        prediction = data["llm_input"].apply(gliner_pii_detector)
    elif model == "gliner-defend":
        prediction = data["llm_input_defend"].apply(gliner_pii_detector)
    elif model == "gpt-4o-mini":
        prediction = parallel_apply(func=llm_pii_detector, series=data["llm_input"])
    elif model == "gpt-4o-mini-defend":
        prediction = parallel_apply(func=llm_pii_detector, series=data["llm_input_defend"])
    else:
        raise ValueError(f"Model {model} is not supported")
    return prediction


def pii_detection_pipeline(models: list[str]):
    """Runs PII detection using multiple models and aggregates results."""
    for dataset in DATASETS:
        data = read_csv(f"datasets/{dataset}_dataset.csv").apply(infer_json)
        ensemble_predictions = DataFrame()
        for model in models:
            if model == "ensemble":
                continue
            data["prediction"] = process_predictions(data=data, model=model, dataset=dataset)
            ensemble_predictions[model] = data["prediction"]
            path = f"datasets/predictions/{dataset}_{model}.csv"
            data.apply(cast_to_json).to_csv(path, index=False)

        ensemble_predictions = ensemble_predictions.apply(spans_set, axis=1)
        data["prediction"] = ensemble_predictions
        path = f"datasets/predictions/{dataset}_ensemble.csv"
        data.apply(cast_to_json).to_csv(path, index=False)
