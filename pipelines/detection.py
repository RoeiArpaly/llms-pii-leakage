"""PII detection pipeline: detector dispatch, prediction I/O, batch orchestration."""
import time
from pathlib import Path

from pandas import (
    concat,
    DataFrame,
    json_normalize,
    read_csv,
    Series,
)

from data_manipulation.defenses.preprocess import (
    defensive_preprocess,
    is_suspicious,
    light_defensive_preprocess,
)
from detectors import unload_models
from detectors.gliner import (
    GLINER_MODELS,
    gliner_pii_detector_batch,
)
from detectors.guards import (
    GRANITE_GUARDIAN_MODELS,
    LLAMA_GUARD_MODELS,
    QWEN_GUARD_MODELS,
    granite_guardian_classify_pii_batch,
    guard_pii_detector_batch,
    llama_guard_classify_pii_batch,
    nemotron_classify_pii_batch,
    qwen_guard_classify_pii_batch,
    wildguard_classify_pii_batch,
)
from detectors.llm import llm_pii_detector
from detectors.slm import (
    LLAMA_SLM_MODELS,
    classify_pii_batch as llama_slm_classify_pii_batch,
)
from detectors.presidio import (
    get_fuzzy_recognizers,
    presidio_pii_analyzer,
)
from logger import logger
from pipelines.generation import DATASET_PATH
from utils import (
    cast_to_json,
    infer_json,
    parallel_apply,
)

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
        name: (lambda data, _name=name, logprobs=False, **_: (
            _llama_guard_with_logprobs(data, _name)
            if logprobs
            else guard_pii_detector_batch(
                data, llama_guard_classify_pii_batch, model_name=_name,
            )
        ))
        for name in LLAMA_GUARD_MODELS
    },
    "nemotron-content-safety-4b": lambda data, logprobs=False, **_: (
        _nemotron_with_logprobs(data)
        if logprobs
        else guard_pii_detector_batch(data, nemotron_classify_pii_batch)
    ),
    "wildguard-7b": lambda data, logprobs=False, **_: (
        _wildguard_with_logprobs(data)
        if logprobs
        else guard_pii_detector_batch(data, wildguard_classify_pii_batch)
    ),
    **{
        name: (lambda data, _name=name, logprobs=False, **_: (
            _qwen_guard_with_logprobs(data, _name)
            if logprobs
            else guard_pii_detector_batch(
                data, qwen_guard_classify_pii_batch, model_name=_name,
            )
        ))
        for name in QWEN_GUARD_MODELS
    },
    **{
        name: (lambda data, _name=name, logprobs=False, **_: (
            _granite_guardian_with_logprobs(data, _name)
            if logprobs
            else guard_pii_detector_batch(
                data, granite_guardian_classify_pii_batch, model_name=_name,
            )
        ))
        for name in GRANITE_GUARDIAN_MODELS
    },
    **{
        name: (lambda data, _name=name, logprobs=False, **_: (
            _llama_slm_with_logprobs(data, _name)
            if logprobs
            else guard_pii_detector_batch(
                data, llama_slm_classify_pii_batch, model_name=_name,
            )
        ))
        for name in LLAMA_SLM_MODELS
    },
    "pii-shield": lambda data, **_: data.apply(_pii_shield_detect),
}

# Models that support logprobs/perplexity output.
_LOGPROB_MODELS = {
    "gpt-4o-mini",
    *LLAMA_GUARD_MODELS,
    *QWEN_GUARD_MODELS,
    *GRANITE_GUARDIAN_MODELS,
    "nemotron-content-safety-4b",
    "wildguard-7b",
    *LLAMA_SLM_MODELS,
}

_SLM_MODELS = {
    *LLAMA_GUARD_MODELS,
    *QWEN_GUARD_MODELS,
    *GRANITE_GUARDIAN_MODELS,
    *LLAMA_SLM_MODELS,
    "nemotron-content-safety-4b",
    "wildguard-7b",
    "gpt-4o-mini",
}

# Models that use conditional is_suspicious() gating (Strategy B).
# Presidio uses unconditional full preprocessing (Strategy A).
_CONDITIONAL_DEFEND_MODELS = _SLM_MODELS | set(GLINER_MODELS)


def _qwen_guard_with_logprobs(data: Series, model_name: str) -> list:
    """Run Qwen Guard per-text with logprobs to get perplexity scores."""
    from detectors.guards.qwen_guard import classify_pii
    results = []
    for text in data:
        r = classify_pii(text, model_name=model_name, logprobs=True)
        results.append({
            "spans": r["spans"],
            "perplexity": r["perplexity"],
        })
    return results


def _llama_guard_with_logprobs(data: Series, model_name: str) -> list:
    """Run Llama Guard per-text with logprobs for perplexity."""
    from detectors.guards.llama_guard import classify_pii as lg_classify
    return [
        {"spans": r["spans"], "perplexity": r["perplexity"]}
        for r in (
            lg_classify(text, model_name=model_name, logprobs=True)
            for text in data
        )
    ]


def _nemotron_with_logprobs(data: Series) -> list:
    """Run Nemotron per-text with logprobs for perplexity."""
    from detectors.guards.nemotron_guard import classify_pii as nm_classify
    return [
        {"spans": r["spans"], "perplexity": r["perplexity"]}
        for r in (nm_classify(text, logprobs=True) for text in data)
    ]


def _wildguard_with_logprobs(data: Series) -> list:
    """Run WildGuard per-text with logprobs for perplexity."""
    from detectors.guards.wildguard import classify_pii as wg_classify
    return [
        {"spans": r["spans"], "perplexity": r["perplexity"]}
        for r in (wg_classify(text, logprobs=True) for text in data)
    ]


def _granite_guardian_with_logprobs(data: Series, model_name: str) -> list:
    """Run Granite Guardian per-text with logprobs for perplexity."""
    from detectors.guards.granite_guardian import classify_pii as gg_classify
    results = []
    for text in data:
        r = gg_classify(text, model_name=model_name, logprobs=True)
        results.append({
            "spans": r["spans"],
            "perplexity": r["perplexity"],
        })
    return results


def _llama_slm_with_logprobs(data: Series, model_name: str) -> list:
    """Run Llama SLM as a true batch (with logprobs for perplexity)."""
    from detectors.slm.llama import classify_pii_batch_full
    return [
        {"spans": r["spans"], "perplexity": r["perplexity"]}
        for r in classify_pii_batch_full(list(data), model_name=model_name)
    ]


def _pii_shield_detect(text):
    from config import Config
    from detectors.guards.qwen_guard import classify_pii
    from functools import partial
    from pii_shield import guard
    result = guard(
        text,
        perplexity_threshold=Config.PERPLEXITY_THRESHOLD,
        gliner_threshold=Config.GLINER_THRESHOLD,
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
        if base_model in _CONDITIONAL_DEFEND_MODELS:
            # NER/SLM: use pre-computed column if available, else compute
            if "_defended_ner_slm" in data.columns:
                input_col = data["_defended_ner_slm"].copy()
            else:
                input_col = input_col.apply(
                    lambda t: (
                        defensive_preprocess(t, include_sandwich=False)
                        if is_suspicious(t)
                        else light_defensive_preprocess(t)
                    ),
                )
        else:
            # Presidio: use pre-computed column if available, else compute
            if "_defended_presidio" in data.columns:
                input_col = data["_defended_presidio"].copy()
            else:
                input_col = input_col.apply(defensive_preprocess)
    # For models supporting logprobs, call per-text with logprobs=True
    # to get perplexity scores alongside detection results.
    if logprobs and base_model in _LOGPROB_MODELS:
        return _DETECTOR_DISPATCH[base_model](input_col, logprobs=True)
    return _DETECTOR_DISPATCH[base_model](input_col, logprobs=False)


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


# ── Preprocessing pre-computation ──────────────────────────────────

def _precompute_defense_columns(dataset: DataFrame, models: list[str]):
    """Pre-compute defensive preprocessing once, shared across all defend models.

    Two strategies:
      A (Presidio): unconditional full preprocess with sandwich
      B (NER/SLM): is_suspicious() gate → full (no sandwich) or light
    """
    defend_models = [m for m in models if m.endswith("-defend")]
    if not defend_models:
        return

    bases = {m.removesuffix("-defend") for m in defend_models}
    need_b = bool(bases & _CONDITIONAL_DEFEND_MODELS)
    need_a = bool(bases - _CONDITIONAL_DEFEND_MODELS)

    if need_b:
        suspicious = dataset["llm_input"].apply(is_suspicious)
        dataset["_defended_ner_slm"] = dataset["llm_input"].combine(
            suspicious,
            lambda t, s: (
                defensive_preprocess(t, include_sandwich=False)
                if s else light_defensive_preprocess(t)
            ),
        )
        logger.info("Pre-computed NER/SLM defense preprocessing")

    if need_a:
        dataset["_defended_presidio"] = dataset["llm_input"].apply(
            defensive_preprocess,
        )
        logger.info("Pre-computed Presidio defense preprocessing")


def _cleanup_defense_columns(dataset: DataFrame):
    for col in ("_defended_ner_slm", "_defended_presidio"):
        if col in dataset.columns:
            dataset.drop(columns=col, inplace=True)


# ── Paired inference helpers ───────────────────────────────────────

def _collect_batch_rows(
    batch_df: DataFrame, model: str, preds, batch_elapsed: float,
) -> list[dict]:
    """Extract prediction rows from detector output."""
    if isinstance(preds, list):
        preds = json_normalize(preds)
        pred_spans = preds["spans"]
        perplexity = preds["perplexity"]
    else:
        pred_spans = preds
        perplexity = Series([None] * len(preds), index=preds.index)

    ms_per_sample = round(batch_elapsed / len(batch_df) * 1000, 1)
    return [
        {
            "uid": uid, "model": model,
            "prediction": pred, "perplexity": perp,
            "latency_ms": ms_per_sample,
        }
        for uid, pred, perp in zip(
            batch_df["uid"], pred_spans, perplexity,
        )
    ]


# ── Main pipeline ───────────────────────────────────────────────────

_BUCKET_NAMES = (
    "positives",
    "adv_positives_direct",
    "adv_positives_direct_indirect",
    "negatives",
    "hard_negatives",
)


def _classify_bucket(row) -> str | None:
    """Map a dataset row to one of the five sampling buckets, or None."""
    cat = row["category"]
    if cat == "negative":
        return "negatives"
    if cat == "hard_negative":
        return "hard_negatives"
    if cat != "positive":
        return None
    at = row["attack_target"]
    if not isinstance(at, dict) or not at:
        return "positives"
    if not at.get("context"):
        return "adv_positives_direct"
    return "adv_positives_direct_indirect"


def _cell_key(row):
    """Stratification cell for a positive row.

    For ``adv_positives_direct``: the fuzzy technique tuple.
    For ``adv_positives_direct_indirect``: (fuzzy, adv) tuple.
    """
    at = row["attack_target"]
    if not isinstance(at, dict):
        return ()
    pii = tuple(at.get("pii") or ())
    ctx = tuple(at.get("context") or ())
    return (pii, ctx)


def _sample_bucket(bucket_df, quota, stratify: bool, random_state: int = 42):
    """Apply quota to a bucket. If stratify, split evenly across cell keys."""
    if quota is None or quota >= len(bucket_df):
        return bucket_df
    if quota <= 0:
        return bucket_df.iloc[0:0]
    if not stratify:
        return bucket_df.sample(n=quota, random_state=random_state)

    keys = bucket_df.apply(_cell_key, axis=1)
    cells = list(bucket_df.groupby(keys))
    n_cells = len(cells)
    per_cell = max(1, quota // n_cells)
    sampled = [
        g.sample(n=min(len(g), per_cell), random_state=random_state)
        for _, g in cells
    ]
    out = concat(sampled) if sampled else bucket_df.iloc[0:0]

    leftover = quota - len(out)
    if leftover > 0:
        remaining = bucket_df[~bucket_df.index.isin(out.index)]
        if len(remaining) > 0:
            extra = remaining.sample(
                n=min(leftover, len(remaining)), random_state=random_state,
            )
            out = concat([out, extra])
    return out


def _apply_sample_quotas(dataset, quotas: dict):
    """Stratified sample by named buckets."""
    bucket_col = dataset.apply(_classify_bucket, axis=1)
    parts = []
    log_lines = []
    for name in _BUCKET_NAMES:
        bucket_df = dataset[bucket_col == name]
        if name not in quotas:
            raise KeyError(
                f"DETECTION_SAMPLE_N missing bucket {name!r}. "
                f"Set to None (all), 0 (exclude), or an integer cap.",
            )
        quota = quotas[name]
        stratify = name in (
            "adv_positives_direct", "adv_positives_direct_indirect",
        )
        sampled = _sample_bucket(bucket_df, quota, stratify=stratify)
        parts.append(sampled)
        cap_str = "all" if quota is None else str(quota)
        log_lines.append(
            f"  {name:<32s} pool={len(bucket_df):>6d} cap={cap_str:>5s} -> {len(sampled):>5d}",
        )
    sampled_dataset = concat(parts, ignore_index=True)
    logger.info(
        "Sampled %d rows by bucket quota:\n%s",
        len(sampled_dataset),
        "\n".join(log_lines),
    )
    return sampled_dataset


def pii_detection_pipeline(
    models: list[str],
    logprobs: bool = False,
    checkpoint=None,
    sample_quotas: dict | None = None,
):
    from pipelines.cli import (
        Spinner,
        print_model_done,
        print_model_skip,
        print_model_skipped,
        print_model_start,
    )

    dataset = read_csv(DATASET_PATH).apply(infer_json)

    if sample_quotas is not None:
        dataset = _apply_sample_quotas(dataset, sample_quotas)

    # Pre-compute defensive preprocessing (shared across all defend models)
    _precompute_defense_columns(dataset, models)

    total_rows = len(dataset)
    skipped_models = set()
    paired_done = set()  # defend models already processed via pairing
    prev_base = None

    if checkpoint:
        det = checkpoint.get_detection_state()
        skipped_models = set(det["skipped"].keys())
        _verify_predictions(det, checkpoint, set(dataset["uid"].tolist()))

    _aggregate_predictions()

    for idx, model in enumerate(models):
        base_model = model.removesuffix("-defend")
        is_defend = model.endswith("-defend")

        if checkpoint and checkpoint.is_done(model):
            print_model_skip(model)
            prev_base = base_model
            continue

        if model in skipped_models:
            prev_base = base_model
            continue

        # Skip defend models already completed via paired inference
        if model in paired_done:
            prev_base = base_model
            continue

        if prev_base is not None and base_model != prev_base:
            unload_models()
        prev_base = base_model

        # Check if we can pair this base model with its defend variant
        defend_model = f"{base_model}-defend"
        pair_defend = (
            not is_defend
            and defend_model in models
            and defend_model not in skipped_models
            and defend_model not in paired_done
            and (not checkpoint or not checkpoint.is_done(defend_model))
        )

        done_uids = _read_model_uids(model)
        remaining = dataset[~dataset["uid"].isin(done_uids)] if done_uids else dataset
        n_remaining = len(remaining)

        # For paired inference, also check defend model's progress
        if pair_defend:
            defend_done_uids = _read_model_uids(defend_model)
        else:
            defend_done_uids = set()

        if n_remaining == 0:
            if checkpoint:
                checkpoint.complete_model(model)
            print_model_done(model, 0)
            if pair_defend:
                defend_remaining = (
                    dataset[~dataset["uid"].isin(defend_done_uids)]
                    if defend_done_uids else dataset
                )
                if len(defend_remaining) == 0:
                    if checkpoint:
                        checkpoint.complete_model(defend_model)
                    print_model_done(defend_model, 0)
                    paired_done.add(defend_model)
            continue

        if done_uids:
            print_model_start(model, f"{n_remaining} remaining of {total_rows}")
        else:
            print_model_start(model, total_rows)

        if checkpoint:
            checkpoint.start_model(model)
            if done_uids:
                checkpoint.save_batch(model, len(done_uids))

        label = (
            f"{model} + {defend_model}" if pair_defend
            else model
        )
        spinner = Spinner(f"{label}  0/{n_remaining} rows")
        spinner.start()
        start_time = time.time()
        processed = 0
        batch_size = 32
        failed = False

        for batch_start in range(0, n_remaining, batch_size):
            batch_df = remaining.iloc[batch_start:batch_start + batch_size]

            try:
                batch_t0 = time.time()
                preds = process_predictions(
                    data=batch_df, model=model, logprobs=logprobs,
                )
                batch_elapsed = time.time() - batch_t0
            except Exception as e:
                spinner.stop()
                reason = str(e).split("\n")[0][:80]
                logger.warning(f"Skipping {model}: {e}")
                skipped_models.update([base_model, defend_model])
                if checkpoint:
                    checkpoint.skip_model(model, reason)
                    checkpoint.skip_model(defend_model, f"paired with {model}")
                unload_models()
                print_model_skipped(model, reason)
                failed = True
                break

            rows = _collect_batch_rows(batch_df, model, preds, batch_elapsed)
            _append_model_predictions(model, rows)
            n_batch = len(rows)
            del preds

            # Paired defend inference on the same batch
            if pair_defend:
                # Only process UIDs not already done for defend
                if defend_done_uids:
                    defend_batch = batch_df[
                        ~batch_df["uid"].isin(defend_done_uids)
                    ]
                else:
                    defend_batch = batch_df
                if not defend_batch.empty:
                    try:
                        d_t0 = time.time()
                        d_preds = process_predictions(
                            data=defend_batch, model=defend_model,
                            logprobs=logprobs,
                        )
                        d_elapsed = time.time() - d_t0
                    except Exception as e:
                        spinner.stop()
                        reason = str(e).split("\n")[0][:80]
                        logger.warning(f"Skipping {defend_model}: {e}")
                        skipped_models.add(defend_model)
                        if checkpoint:
                            checkpoint.skip_model(
                                defend_model, f"failed: {reason}",
                            )
                        pair_defend = False
                        # Continue with base model processing
                    else:
                        d_rows = _collect_batch_rows(
                            defend_batch, defend_model, d_preds, d_elapsed,
                        )
                        _append_model_predictions(defend_model, d_rows)
                        del d_preds, d_rows

            del rows, batch_df

            if checkpoint:
                checkpoint.save_batch(model, n_batch)
            processed += n_batch
            spinner.update(f"{label}  {processed}/{n_remaining} rows")

        if failed:
            continue
        spinner.stop()

        elapsed = round(time.time() - start_time, 1)
        if checkpoint:
            checkpoint.complete_model(model)
        if pair_defend:
            if checkpoint:
                checkpoint.complete_model(defend_model)
            paired_done.add(defend_model)
            print_model_done(f"{model} + {defend_model}", elapsed)
        else:
            print_model_done(model, elapsed)
        _aggregate_predictions()

        next_base = None
        for upcoming in models[idx + 1:]:
            if (upcoming not in skipped_models
                    and upcoming not in paired_done):
                next_base = upcoming.removesuffix("-defend")
                break
        if next_base is not None and next_base != base_model:
            unload_models()

    _cleanup_defense_columns(dataset)
    unload_models()
    _aggregate_predictions()

    # Compute PII Shield results from per-model predictions
    _compute_shield_predictions()

    from config import Config as _Cfg
    all_configured = set(_Cfg.MODELS)
    if checkpoint and all_configured <= (
        set(checkpoint.get_detection_state()["completed"])
        | set(checkpoint.get_detection_state()["skipped"])
    ):
        _cleanup_prediction_parts()
    logger.info(f"Predictions saved to {PREDICTIONS_PATH}")


def _compute_shield_predictions():
    """Compute and append PII Shield cascade results."""
    from evaluation.shield_eval import compute_shield_predictions

    if not PREDICTIONS_PATH.exists():
        return

    shield_df = compute_shield_predictions(
        predictions_path=str(PREDICTIONS_PATH),
    )
    if shield_df.empty:
        return

    # Remove old shield rows and append new ones
    existing = read_csv(PREDICTIONS_PATH)
    existing = existing[existing["model"] != "pii-shield"]
    merged = concat(
        [existing, shield_df.apply(cast_to_json)], ignore_index=True,
    )
    merged.to_csv(PREDICTIONS_PATH, index=False)
    logger.info(f"PII Shield: {len(shield_df)} predictions appended")
