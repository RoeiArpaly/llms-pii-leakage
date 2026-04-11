"""CLI entrypoint for the PII detection evaluation pipeline.

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
from pathlib import Path

from config import Config


def cmd_run(args):
    from pipelines.runner import run_pipeline
    run_pipeline(
        models=args.models,
        skip_gen=args.skip_gen,
        stage=args.stage,
        force=args.force,
        sample_n=args.sample,
    )


def cmd_status(args):
    from pipelines import DATASET_PATH
    from pipelines.checkpoint import CheckpointManager
    from pipelines.cli import print_status
    from pipelines.runner import STAGE_NAMES

    checkpoint = CheckpointManager()
    total_rows = None
    if DATASET_PATH.exists():
        from pandas import read_csv
        total_rows = len(read_csv(DATASET_PATH))
    print_status(checkpoint, Config.MODELS, STAGE_NAMES, total_rows)


def cmd_reset(args):
    from pipelines.checkpoint import CheckpointManager
    from pipelines.cli import print_reset

    checkpoint = CheckpointManager()
    checkpoint.clear()
    Path("datasets/.run_status").unlink(missing_ok=True)
    print_reset()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pii-pipeline",
        description="PII detection evaluation pipeline",
    )
    sub = parser.add_subparsers(dest="command")

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
        help="Cap negatives at N rows, keep all clean positives",
    )

    sub.add_parser("status", help="Show pipeline checkpoint state")
    sub.add_parser("reset", help="Clear all checkpoints")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    cmd = args.command
    if cmd is None or cmd == "run":
        if cmd is None:
            args = parser.parse_args(["run"])
        cmd_run(args)
    elif cmd == "status":
        cmd_status(args)
    elif cmd == "reset":
        cmd_reset(args)
