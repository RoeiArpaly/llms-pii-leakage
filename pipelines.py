"""Pipeline functions for dataset generation, PII detection, and prediction I/O.

Provides three dataset-generation stages (baseline, fuzzy, fuzzy+adversarial),
a multi-model PII detection pipeline with ensemble aggregation, and CSV
serialization helpers.
"""
import random
import time
from pathlib import Path

from pandas import (
    concat,
    DataFrame,
    json_normalize,
    read_csv,
    Series,
)
from constants import (
    ADV_CONTENT_TECHNIQUES,
    DATASET_COLS,
    FUZZY_TECHNIQUES,
)
from data_generation.pii_generator import presidio_inject_pii
from data_generation.llm_input_generator import (
    generate_hard_negative,
    generate_llm_input,
)
from data_manipulation.attacks.injection import (
    adversarial_content,
    pii_fuzzer,
)
from data_manipulation.attacks.neural_prompt_to_prompt.llm import llm_pii_fuzzer
from data_manipulation.defenses.preprocess import (
    defensive_preprocess,
    light_defensive_preprocess,
)
from detectors import unload_models
from detectors.gliner import GLINER_MODELS, gliner_pii_detector_batch
from detectors.guards import (
    LLAMA_GUARD_MODELS,
    QWEN_GUARD_MODELS,
    guard_pii_detector_batch,
    llama_guard_classify_pii_batch,
    nemotron_classify_pii_batch,
    qwen_guard_classify_pii_batch,
    wildguard_classify_pii_batch,
)
from detectors.llm import llm_pii_detector
from detectors.presidio import (
    get_fuzzy_recognizers,
    presidio_pii_analyzer,
)
from logger import logger
from utils import (
    cast_to_json,
    infer_json,
    parallel_apply,
)


DATASET_PATH = Path("datasets/dataset.csv")
PREDICTIONS_PATH = Path("datasets/predictions.csv")
PREDICTIONS_DIR = Path("datasets/predictions")


def _load_dataset() -> DataFrame:
    return read_csv(DATASET_PATH).apply(infer_json)


def _save_dataset(df: DataFrame):
    df = df[DATASET_COLS]
    df.apply(cast_to_json).to_csv(DATASET_PATH, index=False)


def generate_baseline_dataset(n_samples: int, pii_proba: float, save_every_n: int = 100):
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    baseline_cats = ["negative", "hard_negative", "positive"]

    if DATASET_PATH.exists():
        existing = _load_dataset()
        baseline = existing[existing["category"].isin(baseline_cats)]
        logger.info(f"Found existing dataset with {len(baseline)} baseline samples.")
        n_samples = max(0, n_samples - len(baseline))
    else:
        existing = DataFrame()

    from cli import Spinner
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


def generate_fuzzy_adv_dataset():
    dataset = _load_dataset()
    # Fuzzy-derived rows: positive category with input_id != uid
    fuzzy = dataset[
        (dataset["category"] == "positive")
        & (dataset["uid"] != dataset["input_id"])
    ].copy()
    next_uid = dataset["uid"].max() + 1

    datasets = []
    for technique in ADV_CONTENT_TECHNIQUES:
        _data = fuzzy.copy()
        _data["category"] = "positive"
        _data["attack_target"] = _data["attack_target"].apply(
            lambda target: {
                "pii": (target if isinstance(target, dict) else {}).get("pii", []),
                "context": technique,
            },
        )
        # Drop rows where pii is empty — context attacks require a pii target
        _data = _data[_data["attack_target"].apply(
            lambda t: len(t.get("pii", [])) > 0,
        )]
        _data[["llm_input", "pii_spans"]] = _data.apply(
            lambda row: adversarial_content(
                llm_input=row["llm_input"],
                spans=row["pii_spans"],
                chosen_techniques=technique,
            ),
            axis=1,
            result_type="expand",
        )
        datasets.append(_data)

    # Neural Prompt-to-Prompt
    technique = ["neural_prompt_to_prompt"]
    baseline_pii = dataset[
        (dataset["category"] == "positive")
        & (dataset["uid"] == dataset["input_id"])
    ].copy()
    _data = baseline_pii.copy()
    _data["input_id"] = _data["uid"]
    _data[["llm_input", "pii_spans"]] = _data.apply(
        lambda row: llm_pii_fuzzer(
            llm_input=row["llm_input"],
            spans=row["pii_spans"],
            few_shots=False,
        ),
        axis=1,
        result_type="expand",
    )
    _data["attack_target"] = _data.apply(
        lambda _: {"pii": technique, "context": technique}, axis=1,
    )
    _data["category"] = "positive"
    datasets.append(_data)

    adv = concat(datasets, ignore_index=True)
    adv["uid"] = range(next_uid, next_uid + len(adv))

    combined = concat([dataset, adv], ignore_index=True)
    _save_dataset(combined)
    logger.info("Adversarial dataset generation completed successfully")


_DETECTOR_DISPATCH = {
    "presidio": lambda data, **_: data.apply(presidio_pii_analyzer),
    "presidio-fuzzy": lambda data, **_: data.apply(
        presidio_pii_analyzer, recognizers=get_fuzzy_recognizers(),
    ),
    **{
        name: (lambda data, _name=name, **_: Series(
            gliner_pii_detector_batch(data.tolist(), model_name=_name), index=data.index,
        ))
        for name in GLINER_MODELS
    },
    "gpt-4o-mini": lambda data, logprobs=False: parallel_apply(
        func=llm_pii_detector, series=data, logprobs=logprobs,
    ),
    **{
        name: (lambda data, _name=name, **_: guard_pii_detector_batch(
            data, llama_guard_classify_pii_batch, model_name=_name,
        ))
        for name in LLAMA_GUARD_MODELS
    },
    "nemotron-content-safety-4b": lambda data, **_: guard_pii_detector_batch(
        data, nemotron_classify_pii_batch,
    ),
    "wildguard-7b": lambda data, **_: guard_pii_detector_batch(
        data, wildguard_classify_pii_batch,
    ),
    **{
        name: (lambda data, _name=name, **_: guard_pii_detector_batch(
            data, qwen_guard_classify_pii_batch, model_name=_name,
        ))
        for name in QWEN_GUARD_MODELS
    },
    "pii-shield": lambda data, **_: data.apply(_pii_shield_detect),
}


def _pii_shield_detect(text):
    """Run PII Shield cascade on a single text."""
    from pii_shield import guard
    from detectors.guards.qwen_guard import classify_pii as qwen_classify
    result = guard(text, slm_fn=qwen_classify, slm_name="qwen-guard-0.6b")
    if result["detected"]:
        return result.get("spans", [
            {"value": None, "start": None, "end": None, "type": "pii"},
        ])
    return []


_SPAN_KEY_FIELDS = ("value", "start", "end", "type")


def _deduplicate_spans(spans: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for span in spans:
        key = tuple(span.get(f) for f in _SPAN_KEY_FIELDS)
        if key not in seen:
            seen.add(key)
            unique.append({f: span[f] for f in _SPAN_KEY_FIELDS if f in span})
    return unique


# Models that are LLM/SLM-based classifiers — they understand natural language
# and are harmed by aggressive text normalization.
_SLM_MODELS = {
    *LLAMA_GUARD_MODELS,
    *QWEN_GUARD_MODELS,
    "nemotron-content-safety-4b",
    "wildguard-7b",
    "gpt-4o-mini",
}


def process_predictions(
    data: DataFrame, model: str, logprobs: bool,
) -> Series:
    defend = model.endswith("-defend")
    base_model = model.removesuffix("-defend")

    if base_model not in _DETECTOR_DISPATCH:
        raise ValueError(f"Model {model} is not supported")

    input_col = data["llm_input"].copy()
    if defend:
        preprocess_fn = (
            light_defensive_preprocess
            if base_model in _SLM_MODELS
            else defensive_preprocess
        )
        input_col = input_col.apply(preprocess_fn)
    return _DETECTOR_DISPATCH[base_model](input_col, logprobs=logprobs)


def _model_predictions_path(model: str) -> Path:
    """Per-model predictions file: datasets/predictions/<model>.csv"""
    return PREDICTIONS_DIR / f"{model}.csv"


def _append_model_predictions(model: str, rows: list[dict]):
    """Append prediction rows to the per-model CSV file."""
    if not rows:
        return
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = _model_predictions_path(model)
    batch_df = DataFrame(rows)
    write_header = not path.exists() or path.stat().st_size == 0
    batch_df.apply(cast_to_json).to_csv(
        path, mode="a", index=False, header=write_header,
    )


def _read_model_uids(model: str) -> set:
    """Read UIDs already on disk for a model.

    Checks the per-model CSV first, then falls back to predictions.csv.
    """
    path = _model_predictions_path(model)
    if path.exists():
        try:
            df = read_csv(path, usecols=["uid"])
            return set(df["uid"].tolist())
        except Exception:
            return set()
    if PREDICTIONS_PATH.exists():
        try:
            df = read_csv(PREDICTIONS_PATH, usecols=["uid", "model"])
            return set(df.loc[df["model"] == model, "uid"].tolist())
        except Exception:
            return set()
    return set()


def _aggregate_predictions():
    """Upsert all per-model CSVs into predictions.csv.

    Reads the existing predictions.csv (if any), replaces models that have
    a per-model file with the file's content, and keeps models that don't
    have a per-model file unchanged.
    """
    if not PREDICTIONS_DIR.exists():
        return

    model_files = list(PREDICTIONS_DIR.glob("*.csv"))
    if not model_files:
        return

    # Read existing predictions (models without per-model files stay).
    existing = read_csv(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else DataFrame()

    # Build set of models being upserted from per-model files.
    new_parts = []
    upserted_models = set()
    for path in model_files:
        try:
            part = read_csv(path)
            new_parts.append(part)
            upserted_models.add(path.stem)
        except Exception:
            logger.warning(f"Failed to read {path}, skipping")

    # Keep rows from existing predictions for models NOT being upserted.
    if not existing.empty and upserted_models:
        kept = existing[~existing["model"].isin(upserted_models)]
    else:
        kept = DataFrame()

    merged = concat([kept] + new_parts, ignore_index=True) if new_parts else kept
    if not merged.empty:
        merged.to_csv(PREDICTIONS_PATH, index=False)
        logger.info(
            f"Upserted {len(upserted_models)} models into {PREDICTIONS_PATH} "
            f"({len(merged)} total rows)"
        )


def _cleanup_prediction_parts():
    """Remove per-model CSV files after successful aggregation."""
    if PREDICTIONS_DIR.exists():
        for f in PREDICTIONS_DIR.glob("*.csv"):
            f.unlink()
        PREDICTIONS_DIR.rmdir()
        logger.info("Cleaned up per-model prediction files")


def _verify_predictions(
    det: dict,
    checkpoint,
    expected_uids: set,
):
    """Verify per-model prediction files against checkpoint state.

    Reads each completed model's CSV and checks UIDs against the expected
    set. Models with missing UIDs get their checkpoint reset so the pipeline
    resumes only the missing rows.
    """
    completed = det.get("completed", [])
    if not completed:
        return

    # Use presidio UIDs as reference if its file exists, else dataset UIDs.
    presidio_uids = _read_model_uids("presidio")
    reference_uids = presidio_uids if presidio_uids else expected_uids

    incomplete = []
    for model in list(completed):
        model_uids = _read_model_uids(model)
        missing = reference_uids - model_uids
        if not missing:
            continue
        completed.remove(model)
        # Seed checkpoint with existing UIDs so pipeline resumes the gap.
        checkpoint.data["detection"]["in_progress"] = {
            "model": model,
            "processed_uids": sorted(model_uids),
        }
        incomplete.append((model, len(model_uids), len(reference_uids), len(missing)))

    if not incomplete:
        return

    for model, have, total, n_missing in incomplete:
        logger.info(
            f"Verification: {model} has {have}/{total} UIDs, "
            f"{n_missing} missing — will resume"
        )

    checkpoint.data["detection"]["completed"] = completed
    checkpoint._save()


def pii_detection_pipeline(
    models: list[str],
    logprobs: bool = False,
    checkpoint=None,
):
    from cli import (
        Spinner,
        print_model_done,
        print_model_skip,
        print_model_skipped,
        print_model_start,
    )

    dataset = _load_dataset()
    total_rows = len(dataset)
    skipped_models = set()
    prev_base = None

    expected_uids = set(dataset["uid"].tolist())

    if checkpoint:
        det = checkpoint.get_detection_state()
        skipped_models = set(det["skipped"].keys())
        _verify_predictions(det, checkpoint, expected_uids)

    # Upsert any existing per-model files into predictions.csv on startup.
    _aggregate_predictions()

    for idx, model in enumerate(models):
        base_model = model.removesuffix("-defend")

        # Skip completed models
        if checkpoint and checkpoint.is_done(model):
            print_model_skip(model)
            prev_base = base_model
            continue

        # Skip models marked as failed
        if model in skipped_models:
            prev_base = base_model
            continue

        # Free memory when switching to a different base model.
        if prev_base is not None and base_model != prev_base:
            unload_models()
        prev_base = base_model

        # Check for partially processed model (batch-level resume).
        # Read UIDs already on disk for this model's file.
        done_uids = _read_model_uids(model)

        remaining = dataset[~dataset["uid"].isin(done_uids)] if done_uids else dataset
        n_remaining = len(remaining)

        if n_remaining == 0:
            if checkpoint:
                checkpoint.complete_model(model)
            print_model_done(model, 0)
            continue

        if done_uids:
            print_model_start(model, f"{n_remaining} remaining of {total_rows}")
        else:
            print_model_start(model, total_rows)

        if checkpoint:
            checkpoint.start_model(model)
            if done_uids:
                checkpoint.save_batch(model, len(done_uids))

        spinner = Spinner(f"{model}  0/{n_remaining} rows")
        spinner.start()
        start_time = time.time()
        processed = 0
        batch_size = 32
        failed = False

        for batch_start in range(0, n_remaining, batch_size):
            batch_df = remaining.iloc[batch_start:batch_start + batch_size]

            try:
                preds = process_predictions(
                    data=batch_df, model=model, logprobs=logprobs,
                )
            except Exception as e:
                spinner.stop()
                reason = str(e).split("\n")[0][:80]
                logger.warning(f"Skipping {model}: {e}")
                skipped_models.update([base_model, f"{base_model}-defend"])
                if checkpoint:
                    checkpoint.skip_model(model, reason)
                    paired = (
                        f"{base_model}-defend"
                        if not model.endswith("-defend") else base_model
                    )
                    checkpoint.skip_model(paired, f"paired with {model}")
                unload_models()
                print_model_skipped(model, reason)
                failed = True
                break

            if isinstance(preds, list):
                preds = json_normalize(preds)
                pred_spans = preds["spans"]
                perplexity = preds["perplexity"]
            else:
                pred_spans = preds
                perplexity = Series(
                    [None] * len(preds), index=preds.index,
                )

            rows = [
                {
                    "uid": uid,
                    "model": model,
                    "prediction": pred,
                    "perplexity": perp,
                }
                for uid, pred, perp in zip(
                    batch_df["uid"], pred_spans, perplexity,
                )
            ]
            _append_model_predictions(model, rows)
            n_batch = len(rows)
            del preds, pred_spans, perplexity, rows, batch_df

            if checkpoint:
                checkpoint.save_batch(model, n_batch)
            processed += n_batch
            spinner.update(f"{model}  {processed}/{n_remaining} rows")

        if failed:
            continue
        spinner.stop()

        elapsed = round(time.time() - start_time, 1)
        if checkpoint:
            checkpoint.complete_model(model)
        _aggregate_predictions()
        print_model_done(model, elapsed)

        # Proactively unload if the next model uses a different base,
        # so memory is freed immediately instead of at next iteration.
        next_base = None
        for upcoming in models[idx + 1:]:
            if upcoming not in skipped_models:
                next_base = upcoming.removesuffix("-defend")
                break
        if next_base is not None and next_base != base_model:
            unload_models()

    # Final cleanup.
    unload_models()

    # Final upsert.
    _aggregate_predictions()

    # Only clean up per-model files when ALL configured models are done.
    from config import Config as _Cfg
    all_configured = set(_Cfg.MODELS)
    if checkpoint and all_configured <= (
        set(checkpoint.get_detection_state()["completed"])
        | set(checkpoint.get_detection_state()["skipped"])
    ):
        _cleanup_prediction_parts()
    logger.info(f"Predictions saved to {PREDICTIONS_PATH}")
