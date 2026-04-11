"""CLI entrypoint for the PII detection evaluation pipeline.

Provides subcommands for running, inspecting, and resetting the pipeline
with full checkpoint support at stage, model, and batch granularity.

Usage:
    python main.py                              # run / resume
    python main.py run                          # same
    python main.py run --force                  # fresh start
    python main.py run --skip-gen               # skip baseline generation
    python main.py run --models presidio gliner # subset of models
    python main.py run --stage 3                # single stage only
    python main.py run --sample 200             # detect on 200 stratified rows
    python main.py status                       # show checkpoint state
    python main.py reset                        # clear all checkpoints
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from pipelines.checkpoint import CheckpointManager
from pipelines.cli import (
    print_banner,
    print_reset,
    print_stage_header,
    print_status,
)
from config import Config
from logger import logger
from pipelines import (
    DATASET_PATH,
    PREDICTIONS_PATH,
    generate_baseline_dataset,
    generate_fuzzy_dataset,
    generate_fuzzy_adv_dataset,
    pii_detection_pipeline,
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


def _archive_previous_run(checkpoint: CheckpointManager):
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


# ── commands ─────────────────────────────────────────────────────────────

def cmd_run(args):
    print_banner()
    checkpoint = CheckpointManager()

    # --force: archive and start fresh
    if args.force:
        _archive_previous_run(checkpoint)

    # --models: override Config.MODELS
    models = args.models if args.models else Config.MODELS

    # Build stage functions — detection gets the checkpoint
    stages = [
        lambda: generate_baseline_dataset(
            n_samples=Config.NUMBER_OF_SAMPLES, pii_proba=Config.PII_PROBABILITY,
        ),
        generate_fuzzy_dataset,
        generate_fuzzy_adv_dataset,
        lambda: pii_detection_pipeline(
            models=models, logprobs=Config.LOGPROBS, checkpoint=checkpoint,
            sample_n=args.sample,
        ),
        lambda: evaluate_and_save_datasets(
            models=models, match_level=Config.MATCH_LEVEL, method=Config.METHOD,
        ),
        generate_report,
    ]

    last_completed = checkpoint.stage

    # --skip-gen: skip baseline generation only (reuse existing dataset)
    if args.skip_gen:
        if not DATASET_PATH.exists():
            print(f"  \033[31mError:\033[0m --skip-gen requires {DATASET_PATH} to exist")
            sys.exit(1)
        last_completed = max(last_completed, 0)

    # --stage N: run only that stage, ignoring checkpoint for it
    if args.stage is not None:
        run_range = range(args.stage, args.stage + 1)
        # Allow re-running a completed stage (e.g. adding new models)
        if last_completed >= args.stage:
            last_completed = args.stage - 1
    else:
        run_range = range(len(stages))

    # Fresh run: archive previous artifacts
    if last_completed < 0 and not args.force:
        _archive_previous_run(checkpoint)

    # When --models is a subset, detection stage shouldn't be marked
    # complete since not all models have run.
    partial_detection = args.models and set(args.models) != set(Config.MODELS)

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

        # Only mark detection stage complete when all models finished.
        if i == 3 and partial_detection:
            det = checkpoint.get_detection_state()
            done = set(det["completed"]) | set(det["skipped"])
            if done >= set(Config.MODELS):
                checkpoint.complete_stage(i)
            else:
                print(
                    f"  Detection partial — {len(done)}/{len(Config.MODELS)}"
                    " models done"
                )
        else:
            checkpoint.complete_stage(i)

    print("\n\033[32m✓ Pipeline complete.\033[0m")


def cmd_status(args):
    checkpoint = CheckpointManager()
    models = Config.MODELS

    total_rows = None
    if DATASET_PATH.exists():
        from pandas import read_csv
        total_rows = len(read_csv(DATASET_PATH))

    print_status(checkpoint, models, STAGE_NAMES, total_rows)


def cmd_reset(args):
    checkpoint = CheckpointManager()
    checkpoint.clear()
    # Also remove legacy status file
    legacy = Path("datasets/.run_status")
    legacy.unlink(missing_ok=True)
    print_reset()


# ── CLI parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pii-pipeline",
        description="PII detection evaluation pipeline with checkpoint support",
    )
    sub = parser.add_subparsers(dest="command")

    # run
    run_p = sub.add_parser("run", help="Start or resume the pipeline")
    run_p.add_argument(
        "--models", nargs="+", metavar="MODEL",
        help="Run only these models (overrides Config.MODELS)",
    )
    run_p.add_argument(
        "--stage", type=int, choices=range(6), metavar="N",
        help="Run only stage N (0-5)",
    )
    run_p.add_argument(
        "--force", action="store_true",
        help="Archive previous run and start fresh",
    )
    run_p.add_argument(
        "--skip-gen", action="store_true",
        help="Skip baseline generation (stage 0), reuse existing dataset",
    )
    run_p.add_argument(
        "--sample", type=int, metavar="N", default=None,
        help="Run detection on a stratified sample of N rows (faster iteration)",
    )

    # status
    sub.add_parser("status", help="Show pipeline checkpoint state")

    # reset
    sub.add_parser("reset", help="Clear all checkpoints")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    cmd = args.command
    if cmd is None or cmd == "run":
        # Default to run; ensure args has all run attributes
        if cmd is None:
            args = parser.parse_args(["run"])
        cmd_run(args)
    elif cmd == "status":
        cmd_status(args)
    elif cmd == "reset":
        cmd_reset(args)
