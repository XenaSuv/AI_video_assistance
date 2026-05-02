"""Checkpoint/resume support for pipeline runs.

Usage:
    cp = PipelineCheckpoint(run_dir)
    if not cp.is_done("scrape"):
        ...do work...
        cp.mark_done("scrape", {"num_items": 42})

    # On restart: is_done("scrape") returns True and prior metadata is accessible
    meta = cp.metadata("scrape")
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger


_CHECKPOINT_FILE = "checkpoint.json"


class PipelineCheckpoint:
    """Persists step completion state to a JSON file in run_dir.

    Thread-safety: writes are atomic (write-then-rename) on POSIX; callers
    are expected to be single-process, so no locking is needed.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self._path = run_dir / _CHECKPOINT_FILE
        self._state: dict[str, Any] = self._load()

    # ── public API ────────────────────────────────────────────────────────────

    def is_done(self, step: str) -> bool:
        """Return True if *step* completed successfully in a prior run."""
        return self._state.get("steps", {}).get(step, {}).get("status") == "done"

    def mark_done(self, step: str, metadata: dict[str, Any] | None = None) -> None:
        """Record *step* as successfully completed and persist to disk."""
        self._state.setdefault("steps", {})[step] = {
            "status": "done",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **(metadata or {}),
        }
        self._save()
        logger.debug(f"Checkpoint: step '{step}' done")

    def metadata(self, step: str) -> dict[str, Any]:
        """Return stored metadata dict for *step* (empty dict if not found)."""
        entry = self._state.get("steps", {}).get(step, {})
        return {k: v for k, v in entry.items() if k not in ("status", "ts")}

    def completed_steps(self) -> list[str]:
        """Return names of all completed steps in insertion order."""
        return [
            name
            for name, info in self._state.get("steps", {}).items()
            if info.get("status") == "done"
        ]

    def reset(self, step: str | None = None) -> None:
        """Remove checkpoint for *step*, or wipe all steps if *step* is None."""
        if step is None:
            self._state["steps"] = {}
            logger.info("Checkpoint: all steps reset")
        else:
            self._state.get("steps", {}).pop(step, None)
            logger.info(f"Checkpoint: step '{step}' reset")
        self._save()

    # ── internals ─────────────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text())
                done = [k for k, v in data.get("steps", {}).items() if v.get("status") == "done"]
                if done:
                    logger.info(f"Checkpoint: resuming — already done: {done}")
                return data
            except Exception as exc:
                logger.warning(f"Checkpoint file unreadable ({exc}); starting fresh")
        return {"steps": {}}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        tmp.replace(self._path)
