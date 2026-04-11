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
    generate_baseline_dataset,
    generate_fuzzy_adv_dataset,
    generate_fuzzy_dataset,
    pii_detection_pipeline,
)
from pipelines.checkpoint import CheckpointManager
from pipelines.cli import (
    print_banner,
    print_stage_header,
)
from evaluation.report import generate_report
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
    "Report",
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


def run_pipeline(
    models: list[str] | None = None,
    skip_gen: bool = False,
    stage: int | None = None,
    force: bool = False,
    sample_n: int | None = None,
):
    """Execute the full pipeline with checkpoint support."""
    print_banner()
    checkpoint = CheckpointManager()

    if force:
        archive_previous_run(checkpoint)

    models = models if models else Config.MODELS

    stages = [
        lambda: generate_baseline_dataset(
            n_samples=Config.NUMBER_OF_SAMPLES,
            pii_proba=Config.PII_PROBABILITY,
        ),
        generate_fuzzy_dataset,
        generate_fuzzy_adv_dataset,
        lambda: pii_detection_pipeline(
            models=models, logprobs=Config.LOGPROBS,
            checkpoint=checkpoint, sample_n=sample_n,
        ),
        lambda: evaluate_and_save_datasets(
            models=models, match_level=Config.MATCH_LEVEL,
            method=Config.METHOD,
        ),
        generate_report,
    ]

    last_completed = checkpoint.stage

    if skip_gen:
        if not DATASET_PATH.exists():
            print(
                f"  \033[31mError:\033[0m --skip-gen requires "
                f"{DATASET_PATH} to exist"
            )
            sys.exit(1)
        last_completed = max(last_completed, 0)

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
