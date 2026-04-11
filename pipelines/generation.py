"""Dataset generation stages: baseline, fuzzy (PII-level), adversarial (content-level)."""
import random

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path

from pandas import (
    concat,
    DataFrame,
    read_csv,
)

from config import Config
from constants import (
    ADV_CONTENT_TECHNIQUES,
    DATASET_COLS,
    FUZZY_TECHNIQUES,
)
from data_generation.llm_input_generator import (
    generate_hard_negative,
    generate_llm_input,
)
from data_generation.pii_generator import presidio_inject_pii
from data_manipulation.attacks.injection import (
    adversarial_content,
    pii_fuzzer,
)
from data_manipulation.attacks.neural_prompt_to_prompt.llm import llm_pii_fuzzer
from logger import logger
from utils import (
    cast_to_json,
    infer_json,
)

DATASET_PATH = Path("datasets/dataset.csv")


def _load_dataset() -> DataFrame:
    return read_csv(DATASET_PATH).apply(infer_json)


def _save_dataset(df: DataFrame):
    df = df[DATASET_COLS]
    df.apply(cast_to_json).to_csv(DATASET_PATH, index=False)


def generate_baseline_dataset(
    n_samples: int, pii_proba: float, save_every_n: int = 100,
):
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline_cats = ["negative", "hard_negative", "positive"]

    if DATASET_PATH.exists():
        existing = _load_dataset()
        baseline = existing[existing["category"].isin(baseline_cats)]
        logger.info(f"Found existing dataset with {len(baseline)} baseline samples.")
        n_samples = max(0, n_samples - len(baseline))
    else:
        existing = DataFrame()

    from pipelines.cli import Spinner
    spinner = Spinner(f"Generating sample 0/{n_samples}")
    spinner.start()
    results = []
    for i in range(n_samples):
        contains_pii = random.random() < pii_proba
        if contains_pii:
            llm_input = generate_llm_input(contains_pii=True)
            fake_record = presidio_inject_pii(text=llm_input)
            results.append({
                "category": "positive",
                "llm_input": fake_record["text"],
                "pii_spans": fake_record["spans"],
            })
        elif random.random() <= 0.1:
            llm_input = generate_hard_negative()
            results.append({
                "category": "hard_negative",
                "llm_input": llm_input,
                "pii_spans": [],
            })
        else:
            llm_input = generate_llm_input(contains_pii=False)
            results.append({
                "category": "negative",
                "llm_input": llm_input,
                "pii_spans": [],
            })

        spinner.update(f"Generating sample {i + 1}/{n_samples}")

        should_save = (i + 1) % save_every_n == 0 or i + 1 == n_samples
        if should_save:
            partial = DataFrame(results)
            if i + 1 == n_samples:
                partial = partial.sort_values(
                    by="pii_spans",
                    key=lambda spans: spans.str.len().astype(bool),
                    ascending=False,
                ).reset_index(drop=True)
                partial["uid"] = range(len(partial))
                partial["input_id"] = partial["uid"]
                partial["attack_target"] = None

            if not existing.empty:
                data = concat(objs=[existing, partial], ignore_index=True)
            else:
                data = partial

            data.apply(cast_to_json).to_csv(DATASET_PATH, index=False)
    spinner.stop()
    logger.info("LLM input generation completed successfully")


def generate_fuzzy_dataset():
    dataset = _load_dataset()
    baseline_pii = dataset[
        (dataset["category"] == "positive")
        & (dataset["uid"] == dataset["input_id"])
    ].copy()
    next_uid = dataset["uid"].max() + 1

    datasets = []
    for technique in FUZZY_TECHNIQUES:
        _data = baseline_pii.copy()
        _data["input_id"] = _data["uid"]
        _data["category"] = "positive"
        is_baseline = technique == ["baseline"]
        _data["attack_target"] = _data.apply(
            lambda _, t=technique: None if is_baseline else {"pii": t}, axis=1,
        )
        _data[["llm_input", "pii_spans"]] = _data.apply(
            lambda row: pii_fuzzer(
                llm_input=row["llm_input"],
                spans=row["pii_spans"],
                chosen_techniques=technique,
            ),
            axis=1,
            result_type="expand",
        )
        datasets.append(_data)

    fuzzy = concat(datasets, ignore_index=True)
    fuzzy["uid"] = range(next_uid, next_uid + len(fuzzy))

    combined = concat([dataset, fuzzy], ignore_index=True)
    _save_dataset(combined)
    logger.info("Fuzzy dataset generation completed successfully")


def generate_fuzzy_adv_dataset(max_workers: int = 8):
    dataset = _load_dataset()
    fuzzy = dataset[
        (dataset["category"] == "positive")
        & (dataset["uid"] != dataset["input_id"])
    ]
    next_uid = dataset["uid"].max() + 1

    # Pre-filter to rows that have PII attacks (avoid per-technique filtering)
    fuzzy_with_pii = fuzzy[fuzzy["attack_target"].apply(
        lambda t: isinstance(t, dict) and len(t.get("pii", [])) > 0,
    )]

    # Build all content-attack rows in a single pass over the data
    rows = []
    for _, row in fuzzy_with_pii.iterrows():
        pii_techniques = row["attack_target"]["pii"]
        for technique in ADV_CONTENT_TECHNIQUES:
            text, spans = adversarial_content(
                llm_input=row["llm_input"],
                spans=row["pii_spans"],
                chosen_techniques=technique,
            )
            rows.append({
                "uid": 0,
                "input_id": row["input_id"],
                "category": "positive",
                "attack_target": {"pii": pii_techniques, "context": technique},
                "llm_input": text,
                "pii_spans": spans,
            })

    # Neural Prompt-to-Prompt (LLM-based — parallelized)
    # Requires a real LLM; skip when MOCK_LLM is enabled.
    if not Config.MOCK_LLM:
        baseline_pii = dataset[
            (dataset["category"] == "positive")
            & (dataset["uid"] == dataset["input_id"])
        ]
        technique = ["neural_prompt_to_prompt"]

        def _neural_fuzz(row):
            text, spans = llm_pii_fuzzer(
                llm_input=row["llm_input"],
                spans=row["pii_spans"],
                few_shots=False,
            )
            return {
                "uid": 0,
                "input_id": row["uid"],
                "category": "positive",
                "attack_target": {"pii": technique, "context": technique},
                "llm_input": text,
                "pii_spans": spans,
            }

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_neural_fuzz, row)
                for _, row in baseline_pii.iterrows()
            ]
            for future in as_completed(futures):
                rows.append(future.result())

    adv = DataFrame(rows)
    adv["uid"] = range(next_uid, next_uid + len(adv))

    combined = concat([dataset, adv], ignore_index=True)
    _save_dataset(combined)
    logger.info("Adversarial dataset generation completed successfully")
