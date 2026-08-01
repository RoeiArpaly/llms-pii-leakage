"""Pipeline stage orchestration: archive, run, checkpoint management."""
import shutil
import sys
from datetime import datetime
from pathlib import Path

from config import Config
from logger import logger
from pipelines import (
    DATASET_PATH,
    PREDICTIONS_PATH,
    bulk_generate_baseline,
    generate_fuzzy_adv_dataset,
    generate_fuzzy_dataset,
    pii_detection_pipeline,
)
from pipelines.checkpoint import CheckpointManager
from pipelines.cli import (
    print_banner,
    print_stage_header,
)
from evaluation.scoring import (
    EVALUATIONS_PATH,
    evaluate_and_save_datasets,
)

ARCHIVE_DIR = Path("datasets/archive")
STAGE_NAMES = [
    "Baseline generation",
    "Fuzzy generation",
    "Adversarial generation",
    "PII detection",
    "Evaluation",
]


def archive_previous_run(checkpoint: CheckpointManager):
    """Move existing dataset/predictions/evaluations to timestamped archive."""
    artifacts = [DATASET_PATH, PREDICTIONS_PATH, EVALUATIONS_PATH]
    if not any(p.exists() for p in artifacts):
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ARCHIVE_DIR / timestamp
    dest.mkdir(parents=True, exist_ok=True)
    for path in artifacts:
        if path.exists():
            shutil.move(str(path), str(dest / path.name))
    checkpoint.clear()
    logger.info(f"Archived previous run to {dest}")


def _prepare_skip_gen_rerun(checkpoint: CheckpointManager):
    """Reset downstream artifacts while preserving the baseline dataset.

    For `--skip-gen` reruns: strip fuzzy/adversarial rows from dataset.csv
    (keeping only baseline — negatives, hard_negatives, and positives where
    uid == input_id), archive any predictions.csv and evaluations.csv to a
    timestamped archive folder, clear the checkpoint, and mark stage 0
    complete so the pipeline resumes at stage 1 with fresh generation.
    """
    from pandas import read_csv
    from utils import cast_to_json, infer_json

    # 1. Strip fuzzy/adversarial rows, keep only baseline.
    df = read_csv(DATASET_PATH).apply(infer_json)
    baseline_mask = (df["category"] != "positive") | (df["uid"] == df["input_id"])
    stripped = df[baseline_mask].copy()
    n_dropped = len(df) - len(stripped)
    if n_dropped > 0:
        stripped.apply(cast_to_json).to_csv(DATASET_PATH, index=False)
        logger.info(
            f"Stripped {n_dropped} fuzzy/adversarial rows; "
            f"kept {len(stripped)} baseline rows in {DATASET_PATH.name}"
        )

    # 2. Archive predictions + evaluations (not the baseline dataset).
    artifacts = [PREDICTIONS_PATH, EVALUATIONS_PATH]
    existing = [p for p in artifacts if p.exists()]
    if existing:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = ARCHIVE_DIR / timestamp
        dest.mkdir(parents=True, exist_ok=True)
        for path in existing:
            shutil.move(str(path), str(dest / path.name))
        logger.info(
            f"Archived {[p.name for p in existing]} to {dest}"
        )

    # 3. Clear checkpoint, mark stage 0 complete → pipeline resumes at stage 1.
    checkpoint.clear()
    checkpoint.complete_stage(0)


def run_pipeline(
    models: list[str] | None = None,
    skip_gen: bool = False,
    stage: int | None = None,
    force: bool = False,
    sample_quotas: dict | None = None,
):
    """Execute the full pipeline with checkpoint support."""
    print_banner()
    checkpoint = CheckpointManager()

    if force:
        archive_previous_run(checkpoint)

    if skip_gen:
        if not DATASET_PATH.exists():
            print(
                f"  \033[31mError:\033[0m --skip-gen requires "
                f"{DATASET_PATH} to exist"
            )
            sys.exit(1)
        _prepare_skip_gen_rerun(checkpoint)

    models = models if models else Config.MODELS

    stages = [
        lambda: bulk_generate_baseline(
            n=Config.BULK_TARGET_N,
            out_path=DATASET_PATH,
            weights=Config.BULK_WEIGHTS,
            model=Config.BULK_MODEL,
            workers=Config.BULK_WORKERS,
            checkpoint_every=Config.BULK_CHECKPOINT_EVERY,
        ),
        generate_fuzzy_dataset,
        generate_fuzzy_adv_dataset,
        lambda: pii_detection_pipeline(
            models=models, logprobs=Config.LOGPROBS,
            checkpoint=checkpoint, sample_quotas=sample_quotas,
        ),
        lambda: evaluate_and_save_datasets(
            models=models, match_level=Config.MATCH_LEVEL,
            method=Config.METHOD,
        ),
    ]

    last_completed = checkpoint.stage

    if stage is not None:
        run_range = range(stage, stage + 1)
        if last_completed >= stage:
            last_completed = stage - 1
    else:
        run_range = range(len(stages))

    if last_completed < 0 and not force:
        archive_previous_run(checkpoint)

    partial_detection = (
        models and set(models) != set(Config.MODELS)
    )

    total = len(stages)
    for i in run_range:
        if i <= last_completed:
            print(
                f"  \033[32m✓\033[0m [Stage {i + 1}/{total}] "
                f"{STAGE_NAMES[i]} — skipping (checkpoint)"
            )
            continue

        print_stage_header(i + 1, total, STAGE_NAMES[i])
        stages[i]()

        if i == 3 and partial_detection:
            det = checkpoint.get_detection_state()
            done = set(det["completed"]) | set(det["skipped"])
            if done >= set(Config.MODELS):
                checkpoint.complete_stage(i)
            else:
                print(
                    f"  Detection partial — {len(done)}"
                    f"/{len(Config.MODELS)} models done"
                )
        else:
            checkpoint.complete_stage(i)

    print("\n\033[32m✓ Pipeline complete.\033[0m")
