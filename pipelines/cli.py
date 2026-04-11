"""CLI display helpers: banner, status formatting, and live activity spinner.

Uses Unicode box-drawing and braille spinner for a polished terminal experience.
No external dependencies — stdlib only.
"""
import sys
import threading
import time
from pathlib import Path

from pipelines.checkpoint import CheckpointManager


BANNER = """\
\033[36m╔══════════════════════════════════════════╗
║\033[0m\033[1m         PII Detection Pipeline           \033[0m\033[36m║
║\033[0m         ─────────────────────            \033[36m║
║\033[0m  Adversarial PII Detector Evaluation     \033[36m║
╚══════════════════════════════════════════╝\033[0m"""

_DONE = "\033[32m✓\033[0m"
_SKIP = "\033[31m✗\033[0m"
_RUN = "\033[33m►\033[0m"
_PEND = "\033[90m·\033[0m"

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


# ── live spinner ─────────────────────────────────────────────────────────

class Spinner:
    """Background spinner that shows elapsed time and an optional status.

    Usage:
        spinner = Spinner("Loading model")
        spinner.start()
        do_work()
        spinner.update("Running inference")
        do_more_work()
        spinner.stop()
    """

    def __init__(self, text: str = ""):
        self._text = text
        self._start_time = None
        self._stop_event = threading.Event()
        self._thread = None
        self._frame = 0

    def start(self):
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def update(self, text: str):
        self._text = text

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()
        # Clear the spinner line
        sys.stdout.write("\033[2K\r")
        sys.stdout.flush()

    def _spin(self):
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            frame = _SPINNER_FRAMES[self._frame % len(_SPINNER_FRAMES)]
            line = f"\r    \033[36m{frame}\033[0m {self._text}  \033[90m({elapsed:.0f}s)\033[0m"
            sys.stdout.write(f"\033[2K{line}")
            sys.stdout.flush()
            self._frame += 1
            self._stop_event.wait(0.1)


# ── banner ───────────────────────────────────────────────────────────────

def print_banner():
    print(BANNER)
    print()


# ── status command ───────────────────────────────────────────────────────

def print_status(
    checkpoint: CheckpointManager,
    models: list[str],
    stage_names: list[str],
    total_rows: int | None = None,
):
    print_banner()

    if not checkpoint.exists and checkpoint.stage == -1:
        print("No active run. Use \033[1mpython main.py run\033[0m to start.")
        return

    # ── stages ───────────────────────────────────────────────────────
    print("\033[1mPipeline Stages\033[0m")
    print("───────────────")
    current_stage = checkpoint.stage
    det_state = checkpoint.get_detection_state()
    has_detection_progress = (
        det_state["completed"] or det_state["skipped"] or det_state["in_progress"]
    )

    for i, name in enumerate(stage_names):
        num = f"[{i + 1}/{len(stage_names)}]"
        if i <= current_stage:
            print(f"  {num} {name:<30s} {_DONE} done")
        elif i == current_stage + 1 and has_detection_progress and i == 3:
            print(f"  {num} {name:<30s} {_RUN} in progress")
        else:
            print(f"  {num} {name:<30s} {_PEND} pending")
    print()

    # ── detection detail ─────────────────────────────────────────────
    if has_detection_progress:
        _print_detection_progress(det_state, models, total_rows)

    # ── files ────────────────────────────────────────────────────────
    _print_files()


def _print_detection_progress(
    det_state: dict,
    models: list[str],
    total_rows: int | None,
):
    completed = set(det_state["completed"])
    skipped = det_state["skipped"]
    in_progress = det_state["in_progress"]
    ip_model = in_progress["model"] if in_progress else None
    ip_count = len(in_progress.get("processed_uids", [])) if in_progress else 0

    rows_str = f"/{total_rows}" if total_rows else ""

    print("\033[1mDetection Progress\033[0m")
    print("──────────────────")
    for model in models:
        if model in completed:
            print(f"  {_DONE} {model:<36s} {total_rows or ''}{rows_str} rows")
        elif model in skipped:
            reason = skipped[model]
            detail = f"skipped ({reason})" if reason else "skipped"
            print(f"  {_SKIP} {model:<36s} {detail}")
        elif model == ip_model:
            print(f"  {_RUN} {model:<36s} {ip_count}{rows_str} rows")
        else:
            print(f"  {_PEND} {model:<36s} pending")
    print()


def _print_files():
    print("\033[1mFiles\033[0m")
    print("─────")
    for name, path in [
        ("Dataset", Path("datasets/dataset.csv")),
        ("Predictions", Path("datasets/predictions.csv")),
        ("Evaluations", Path("datasets/evaluations.csv")),
        ("Report", Path("/evaluation/report/report.html")),
    ]:
        if path.exists():
            size = path.stat().st_size
            if size > 1_000_000:
                size_str = f"{size / 1_000_000:.1f} MB"
            elif size > 1_000:
                size_str = f"{size / 1_000:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {name:<14s} {path}  ({size_str})")
        else:
            print(f"  {name:<14s} {path}  \033[90m(not found)\033[0m")
    print()


# ── run-time output ──────────────────────────────────────────────────────

def print_stage_header(stage_num: int, total: int, name: str):
    print(f"\n\033[1m[Stage {stage_num}/{total}] {name}\033[0m")


def print_model_skip(model: str):
    print(f"  {_DONE} {model:<30s} — skipping (checkpoint)")


def print_model_start(model: str, detail):
    print(f"  {_RUN} {model} ({detail} rows)")


def print_model_done(model: str, elapsed: float):
    print(f"  {_DONE} {model:<30s} — {elapsed:.1f}s, checkpoint saved")


def print_model_skipped(model: str, reason: str):
    print(f"  {_SKIP} {model:<30s} — skipped ({reason})")


def print_reset():
    print_banner()
    print("Cleared checkpoint.")
    print("Pipeline ready for fresh run.")
