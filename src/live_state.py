"""Global live-state file for real-time pipeline monitoring.

Writes to data/live_state.json on every change using atomic tmp-file replace
so the dashboard never reads a partial write.

Usage from the pipeline::

    live = LiveState(settings.data_dir / "live_state.json")
    live.start("daily", steps_total=8)
    live.log_event("Scraped 23 stories")
    live.set_strategy({"mode": "growth", "top_angle": "technical_breakthrough"})
    live.step_running("voice")
    live.step_done("voice", duration_sec=34.2)
    live.finish("success")

Usage from the dashboard::

    state = LiveState.load(Path("data/live_state.json"))
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from src.state_io import json_lock

MAX_LOG_EVENTS = 100


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_hms() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


_IDLE: dict[str, Any] = {
    "pipeline": None,
    "status": "idle",
    "started_at": None,
    "updated_at": None,
    "current_step": None,
    "steps_done": 0,
    "steps_total": 0,
    "progress": 0.0,
    "strategy": {},
    "steps": [],
    "logs": [],
    "metrics": {},
    "cost": {},
}


class LiveState:
    """Atomic-write JSON sink for real-time pipeline monitoring.

    All public methods are non-fatal: write errors are silently swallowed
    so a disk issue never crashes the pipeline.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._state: dict[str, Any] = {**_IDLE}

    # ── pipeline lifecycle ────────────────────────────────────────────────────

    def start(self, pipeline: str, steps_total: int) -> None:
        self._state = {
            **_IDLE,
            "pipeline": pipeline,
            "status": "running",
            "started_at": _now_iso(),
            "updated_at": _now_iso(),
            "steps_total": steps_total,
            "logs": [{"time": _now_hms(), "msg": f"Pipeline '{pipeline}' started"}],
        }
        self._write()

    def finish(self, status: str = "success") -> None:
        self._state["status"] = status
        self._state["current_step"] = None
        if status == "success":
            self._state["progress"] = 1.0
        self._state["updated_at"] = _now_iso()
        self._write()

    # ── step lifecycle ────────────────────────────────────────────────────────

    def step_running(self, name: str) -> None:
        self._state["current_step"] = name
        self._state["updated_at"] = _now_iso()
        self._upsert_step(name, "running")
        self._write()

    def step_done(self, name: str, duration_sec: float | None = None) -> None:
        self._state["steps_done"] = self._state.get("steps_done", 0) + 1
        self._state["progress"] = self._calc_progress()
        self._state["updated_at"] = _now_iso()
        extra: dict[str, Any] = {}
        if duration_sec is not None:
            extra["duration_sec"] = round(duration_sec, 1)
        self._upsert_step(name, "done", **extra)
        self._write()

    def step_skipped(self, name: str) -> None:
        self._state["steps_done"] = self._state.get("steps_done", 0) + 1
        self._state["progress"] = self._calc_progress()
        self._state["updated_at"] = _now_iso()
        self._upsert_step(name, "skipped")
        self._write()

    def step_failed(self, name: str, error: str) -> None:
        self._state["updated_at"] = _now_iso()
        self._upsert_step(name, "failed", error=error)
        self._write()

    # ── enrichment ────────────────────────────────────────────────────────────

    def log_event(self, msg: str) -> None:
        logs: list[dict[str, Any]] = self._state.get("logs", [])
        logs.append({"time": _now_hms(), "msg": msg})
        if len(logs) > MAX_LOG_EVENTS:
            logs = logs[-MAX_LOG_EVENTS:]
        self._state["logs"] = logs
        self._state["updated_at"] = _now_iso()
        self._write()

    def set_strategy(self, strategy: dict[str, Any]) -> None:
        self._state["strategy"] = strategy
        self._state["updated_at"] = _now_iso()
        self._write()

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        self._state["metrics"] = metrics
        self._state["updated_at"] = _now_iso()
        self._write()

    def set_cost(self, cost: dict[str, Any]) -> None:
        self._state["cost"] = cost
        self._state["updated_at"] = _now_iso()
        self._write()

    # ── read ──────────────────────────────────────────────────────────────────

    @staticmethod
    def load(path: Path) -> dict[str, Any]:
        """Read current state from disk. Returns idle state on any error."""
        try:
            data: dict[str, Any] = json.loads(path.read_text())
            return data
        except Exception as exc:
            logger.debug(f"LiveState.load: {exc}")
            return {**_IDLE}

    # ── internals ─────────────────────────────────────────────────────────────

    def _calc_progress(self) -> float:
        total = self._state.get("steps_total") or 1
        done = self._state.get("steps_done", 0)
        result: float = min(1.0, done / total)
        return result

    def _upsert_step(self, name: str, status: str, **extra: Any) -> None:
        steps: list[dict[str, Any]] = self._state.get("steps", [])
        for step in steps:
            if step["name"] == name:
                step["status"] = status
                step.update(extra)
                return
        entry: dict[str, Any] = {"name": name, "status": status}
        entry.update(extra)
        steps.append(entry)
        self._state["steps"] = steps

    def _write(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        try:
            with json_lock(self.path):
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(json.dumps(self._state, indent=2, default=str))
                tmp.replace(self.path)
        except Exception as exc:
            logger.warning(f"LiveState._write: {exc}")
