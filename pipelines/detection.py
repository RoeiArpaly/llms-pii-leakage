"""PII detection pipeline: detector dispatch, prediction I/O, batch orchestration."""
import time
from pathlib import Path

from pandas import concat, DataFrame, json_normalize, read_csv, Series

from data_manipulation.defenses.preprocess import (
    defensive_preprocess,
    is_suspicious,
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
from detectors.presidio import get_fuzzy_recognizers, presidio_pii_analyzer
from logger import logger
from pipelines.generation import DATASET_PATH
from utils import cast_to_json, infer_json, parallel_apply

PREDICTIONS_PATH = Path("datasets/predictions.csv")
PREDICTIONS_DIR = Path("datasets/predictions")


# ── Detector dispatch table ─────────────────────────────────────────

_DETECTOR_DISPATCH = {
    "presidio": lambda data, **_: data.apply(presidio_pii_analyzer),
    "presidio-fuzzy": lambda data, **_: data.apply(
        presidio_pii_analyzer, recognizers=get_fuzzy_recognizers(),
    ),
    **{
        name: (lambda data, _name=name, **_: Series(
            gliner_pii_detector_batch(data.tolist(), model_name=_name),
            index=data.index,
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

_SLM_MODELS = {
    *LLAMA_GUARD_MODELS,
    *QWEN_GUARD_MODELS,
    "nemotron-content-safety-4b",
    "wildguard-7b",
    "gpt-4o-mini",
}


def _pii_shield_detect(text):
    from config import Config
    from detectors.guards.qwen_guard import classify_pii
    from functools import partial
    from pii_shield import guard
    result = guard(
        text,
        perplexity_threshold=Config.PERPLEXITY_THRESHOLD,
        slm_detector=partial(
            classify_pii, model_name="qwen-guard-0.6b", logprobs=True,
        ),
    )
    if result["detected"]:
        return result.get("spans", [
            {"value": None, "start": None, "end": None, "type": "pii"},
        ])
    return []


# ── Prediction processing ───────────────────────────────────────────

def process_predictions(
    data: DataFrame, model: str, logprobs: bool,
) -> Series:
    defend = model.endswith("-defend")
    base_model = model.removesuffix("-defend")

    if base_model not in _DETECTOR_DISPATCH:
        raise ValueError(f"Model {model} is not supported")

    input_col = data["llm_input"].copy()
    if defend:
        if base_model in _SLM_MODELS:
            # SLMs: light normalization by default. If character anomaly
            # detection flags the input as suspicious (homoglyphs, emoji,
            # zero-width chars), apply full normalization without sandwich.
            input_col = input_col.apply(
                lambda t: (
                    defensive_preprocess(t, include_sandwich=False)
                    if is_suspicious(t)
                    else light_defensive_preprocess(t)
                ),
            )
        elif base_model in GLINER_MODELS:
            # NER: full normalization but no sandwich (noise dilutes NER signal)
            input_col = input_col.apply(
                lambda t: defensive_preprocess(t, include_sandwich=False),
            )
        else:
            # Presidio: full normalization + sandwich
            input_col = input_col.apply(defensive_preprocess)
    return _DETECTOR_DISPATCH[base_model](input_col, logprobs=logprobs)


# ── Prediction I/O ──────────────────────────────────────────────────

def _model_predictions_path(model: str) -> Path:
    return PREDICTIONS_DIR / f"{model}.csv"


def _append_model_predictions(model: str, rows: list[dict]):
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
    path = _model_predictions_path(model)
    if path.exists():
        try:
            return set(read_csv(path, usecols=["uid"])["uid"].tolist())
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
    if not PREDICTIONS_DIR.exists():
        return
    model_files = list(PREDICTIONS_DIR.glob("*.csv"))
    if not model_files:
        return

    existing = read_csv(PREDICTIONS_PATH) if PREDICTIONS_PATH.exists() else DataFrame()

    new_parts = []
    upserted_models = set()
    for path in model_files:
        try:
            new_parts.append(read_csv(path))
            upserted_models.add(path.stem)
        except Exception:
            logger.warning(f"Failed to read {path}, skipping")

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
    if PREDICTIONS_DIR.exists():
        for f in PREDICTIONS_DIR.glob("*.csv"):
            f.unlink()
        PREDICTIONS_DIR.rmdir()
        logger.info("Cleaned up per-model prediction files")


def _verify_predictions(det, checkpoint, expected_uids):
    completed = det.get("completed", [])
    if not completed:
        return

    presidio_uids = _read_model_uids("presidio")
    reference_uids = presidio_uids if presidio_uids else expected_uids

    incomplete = []
    for model in list(completed):
        model_uids = _read_model_uids(model)
        missing = reference_uids - model_uids
        if not missing:
            continue
        completed.remove(model)
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


# ── Main pipeline ───────────────────────────────────────────────────

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

    dataset = read_csv(DATASET_PATH).apply(infer_json)
    total_rows = len(dataset)
    skipped_models = set()
    prev_base = None

    if checkpoint:
        det = checkpoint.get_detection_state()
        skipped_models = set(det["skipped"].keys())
        _verify_predictions(det, checkpoint, set(dataset["uid"].tolist()))

    _aggregate_predictions()

    for idx, model in enumerate(models):
        base_model = model.removesuffix("-defend")

        if checkpoint and checkpoint.is_done(model):
            print_model_skip(model)
            prev_base = base_model
            continue

        if model in skipped_models:
            prev_base = base_model
            continue

        if prev_base is not None and base_model != prev_base:
            unload_models()
        prev_base = base_model

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
                perplexity = Series([None] * len(preds), index=preds.index)

            rows = [
                {
                    "uid": uid, "model": model,
                    "prediction": pred, "perplexity": perp,
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

        next_base = None
        for upcoming in models[idx + 1:]:
            if upcoming not in skipped_models:
                next_base = upcoming.removesuffix("-defend")
                break
        if next_base is not None and next_base != base_model:
            unload_models()

    unload_models()
    _aggregate_predictions()

    from config import Config as _Cfg
    all_configured = set(_Cfg.MODELS)
    if checkpoint and all_configured <= (
        set(checkpoint.get_detection_state()["completed"])
        | set(checkpoint.get_detection_state()["skipped"])
    ):
        _cleanup_prediction_parts()
    logger.info(f"Predictions saved to {PREDICTIONS_PATH}")
