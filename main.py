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


def _apply_api_key(args) -> None:
    """Set env var from --api-key, then enforce DRYRUN-aware key requirement."""
    import os
    import sys

    if args.dryrun:
        Config.DRYRUN = True

    if args.api_key:
        env_var = "OPENAI_API_KEY" if args.provider == "openai" else "OPENROUTER_API_KEY"
        os.environ[env_var] = args.api_key

    if Config.DRYRUN:
        return

    if not (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")):
        sys.stderr.write(
            "error: no API key found.\n"
            "  pass --api-key sk-... or export "
            "OPENROUTER_API_KEY / OPENAI_API_KEY,\n"
            "  or pass --dryrun (no real LLM calls, mock responses) for "
            "smoke tests.\n",
        )
        sys.exit(2)


def cmd_run(args):
    import sys

    skipping_baseline = args.skip_gen or args.stage not in (None, 0)
    if not skipping_baseline:
        _apply_api_key(args)

    from pipelines.runner import run_pipeline
    from utils.api import AuthenticationError

    try:
        run_pipeline(
            models=args.models,
            skip_gen=args.skip_gen,
            stage=args.stage,
            force=args.force,
            sample_n=args.sample,
        )
    except AuthenticationError as e:
        sys.stderr.write(f"\nerror: {e}\n")
        sys.exit(2)


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


def cmd_generate(args):
    import sys

    _apply_api_key(args)

    from pipelines.generation import bulk_generate_baseline
    from utils.api import AuthenticationError

    weights = tuple(float(w) for w in args.weights.split(","))
    try:
        bulk_generate_baseline(
            n=args.n,
            out_path=args.out,
            weights=weights,
            model=args.model,
            workers=args.workers,
            checkpoint_every=args.checkpoint_every,
        )
    except AuthenticationError as e:
        sys.stderr.write(f"\nerror: {e}\n")
        sys.exit(2)


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
        help=(
            "Reuse baseline dataset. Strips fuzzy/adversarial rows, "
            "archives predictions+evaluations, clears the checkpoint, "
            "and reruns stages 1-5 fresh."
        ),
    )
    run_p.add_argument(
        "--sample", type=int, metavar="N", default=None,
        help="Cap negatives at N rows, keep all clean positives",
    )
    run_p.add_argument(
        "--api-key", type=str, default=None,
        help="API key override (otherwise read from OPENROUTER_API_KEY / OPENAI_API_KEY env)",
    )
    run_p.add_argument(
        "--provider", choices=["openrouter", "openai"], default="openrouter",
        help="Which provider --api-key applies to (default: openrouter)",
    )
    run_p.add_argument(
        "--dryrun", action="store_true",
        help="Smoke test: mock all LLM responses, no API key needed",
    )

    sub.add_parser("status", help="Show pipeline checkpoint state")
    sub.add_parser("reset", help="Clear all checkpoints")

    gen_p = sub.add_parser(
        "generate",
        help="Bulk-generate chat-style samples (negative/positive/hard_negative)",
    )
    gen_p.add_argument(
        "--n", type=int, default=30,
        help="target total rows (default: 30 for a quick validation run)",
    )
    gen_p.add_argument(
        "--out", type=Path, default=Path("datasets/dataset_extra.csv"),
        help="output CSV path (default: datasets/dataset_extra.csv)",
    )
    gen_p.add_argument("--model", type=str, default="gpt-4o-mini")
    gen_p.add_argument("--workers", type=int, default=8)
    gen_p.add_argument("--checkpoint-every", type=int, default=100)
    gen_p.add_argument(
        "--weights", type=str, default="0.90,0.05,0.05",
        help="comma-separated weights for negative,positive,hard_negative",
    )
    gen_p.add_argument(
        "--api-key", type=str, default=None,
        help="API key override (otherwise read from OPENROUTER_API_KEY / OPENAI_API_KEY env)",
    )
    gen_p.add_argument(
        "--provider", choices=["openrouter", "openai"], default="openrouter",
        help="Which provider --api-key applies to (default: openrouter)",
    )
    gen_p.add_argument(
        "--dryrun", action="store_true",
        help="Smoke test: mock all LLM responses, no API key needed",
    )

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
    elif cmd == "generate":
        cmd_generate(args)
