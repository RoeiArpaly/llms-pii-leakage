"""Atomic JSON checkpoint manager with stage, model, and batch-level tracking.

Persists pipeline progress to disk so runs can resume after failures or
machine restarts. Uses atomic writes (temp file + os.replace) to prevent
corruption on crash.
"""
import json
import os
import tempfile
from pathlib import Path


CHECKPOINT_PATH = Path("datasets/.checkpoint.json")

_EMPTY = {
    "stage": -1,
    "detection": {
        "completed": [],
        "skipped": {},
        "in_progress": None,
    },
}


class CheckpointManager:

    def __init__(self, path: Path = CHECKPOINT_PATH):
        self.path = path
        self.data = self._load()
        self._migrate()

    # ── persistence ──────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not self.path.exists():
            # Migrate from legacy .run_status if present
            legacy = self.path.parent / ".run_status"
            if legacy.exists():
                try:
                    stage = int(legacy.read_text().strip())
                except (ValueError, OSError):
                    stage = -1
                data = _deep_copy(_EMPTY)
                data["stage"] = stage
                legacy.unlink(missing_ok=True)
                return data
            return _deep_copy(_EMPTY)
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return _deep_copy(_EMPTY)

    def _migrate(self):
        """Convert old processed_uids list to processed_count int."""
        ip = self.data.get("detection", {}).get("in_progress")
        if ip and "processed_uids" in ip:
            ip["processed_count"] = len(ip.pop("processed_uids"))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            dir=self.path.parent, suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(self.data, f, indent=2)
            os.replace(tmp, self.path)
        except BaseException:
            os.unlink(tmp)
            raise

    def clear(self):
        self.data = _deep_copy(_EMPTY)
        self.path.unlink(missing_ok=True)

    @property
    def exists(self) -> bool:
        return self.path.exists()

    # ── stage level ──────────────────────────────────────────────────────

    @property
    def stage(self) -> int:
        return self.data.get("stage", -1)

    def complete_stage(self, stage: int):
        self.data["stage"] = stage
        # Clear detection sub-state when stage 3 fully completes
        if stage >= 3:
            self.data["detection"] = _deep_copy(_EMPTY["detection"])
        self._save()

    # ── model level ──────────────────────────────────────────────────────

    @property
    def _det(self) -> dict:
        return self.data.setdefault("detection", _deep_copy(_EMPTY["detection"]))

    def is_done(self, model: str) -> bool:
        return model in self._det.get("completed", [])

    def is_skipped(self, model: str) -> bool:
        return model in self._det.get("skipped", {})

    def start_model(self, model: str):
        self._det["in_progress"] = {
            "model": model,
            "processed_count": 0,
        }
        self._save()

    def complete_model(self, model: str):
        completed = self._det.setdefault("completed", [])
        if model not in completed:
            completed.append(model)
        self._det["in_progress"] = None
        self._save()

    def skip_model(self, model: str, reason: str = ""):
        skipped = self._det.setdefault("skipped", {})
        skipped[model] = reason
        self._det["in_progress"] = None
        self._save()

    # ── batch level ──────────────────────────────────────────────────────

    def save_batch(self, model: str, count: int):
        """Update the processed count for the in-progress model."""
        ip = self._det.get("in_progress")
        if ip and ip.get("model") == model:
            ip["processed_count"] = ip.get("processed_count", 0) + count
            self._save()

    # ── status reporting ─────────────────────────────────────────────────

    def get_detection_state(self) -> dict:
        det = self._det
        return {
            "completed": list(det.get("completed", [])),
            "skipped": dict(det.get("skipped", {})),
            "in_progress": det.get("in_progress"),
        }

    def in_progress_model(self) -> str | None:
        ip = self._det.get("in_progress")
        return ip["model"] if ip else None

    def in_progress_count(self) -> int:
        ip = self._det.get("in_progress")
        if not ip:
            return 0
        val = ip.get("processed_count", ip.get("processed_uids", 0))
        return len(val) if isinstance(val, list) else val


def _deep_copy(d: dict) -> dict:
    return json.loads(json.dumps(d))
