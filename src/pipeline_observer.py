"""Structured per-step timing and metadata for pipeline runs.

Writes pipeline_trace.json to run_dir after every step so partial traces
are available even when the pipeline crashes mid-run.

Usage in main.py::

    observer = PipelineObserver(run_dir, pipeline="daily")

    observer.step_start("scrape")
    try:
        ...work...
        observer.step_done("scrape", scraped_items=len(news))
    except Exception as exc:
        observer.step_fail("scrape", exc)
        raise

    # already-done branch:
    observer.step_skip("scrape")

    trace = observer.finish(status="success")
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from loguru import logger


_TRACE_FILE = "pipeline_trace.json"


@dataclass
class _StepRecord:
    name: str
    status: str                    # running | done | skipped | failed
    started_at: str
    started_ts: float = field(repr=False)
    finished_at: str | None = None
    duration_sec: float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "name":         self.name,
            "status":       self.status,
            "started_at":   self.started_at,
            "finished_at":  self.finished_at,
            "duration_sec": round(self.duration_sec, 3) if self.duration_sec is not None else None,
        }
        if self.error:
            d["error"] = self.error
        if self.metadata:
            d.update(self.metadata)
        return d


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class PipelineObserver:
    """Track timing and metadata for each pipeline step.

    Persists an incremental trace to *run_dir/pipeline_trace.json* after
    every state change so a crash mid-run still leaves a readable file.
    """

    def __init__(self, run_dir: Path, pipeline: str = "daily") -> None:
        self.run_dir  = run_dir
        self.pipeline = pipeline
        self._started_ts  = time.time()
        self._started_at  = _now_iso()
        self._steps: dict[str, _StepRecord] = {}
        self._finished_at: str | None = None
        self._status: str = "running"
        self._extra: dict[str, Any] = {}

    # ── step lifecycle ────────────────────────────────────────────────────────

    def step_start(self, name: str) -> None:
        """Call immediately before starting a step (before the is_done check)."""
        self._steps[name] = _StepRecord(
            name=name,
            status="running",
            started_at=_now_iso(),
            started_ts=time.time(),
        )
        self._persist()

    def step_done(self, name: str, **metadata) -> None:
        """Call after a step completes successfully."""
        rec = self._steps.get(name)
        if rec is None:
            logger.warning(f"PipelineObserver: step_done('{name}') called without step_start")
            return
        elapsed = time.time() - rec.started_ts
        rec.status       = "done"
        rec.finished_at  = _now_iso()
        rec.duration_sec = elapsed
        rec.metadata     = metadata
        logger.debug(f"Step '{name}' done in {elapsed:.1f}s")
        self._persist()

    def step_skip(self, name: str, **metadata) -> None:
        """Call when a step is skipped because it was already done."""
        rec = self._steps.get(name)
        now_ts = time.time()
        if rec is None:
            rec = _StepRecord(name=name, status="skipped",
                              started_at=_now_iso(), started_ts=now_ts)
            self._steps[name] = rec
        elapsed = now_ts - rec.started_ts
        rec.status       = "skipped"
        rec.finished_at  = _now_iso()
        rec.duration_sec = elapsed
        rec.metadata     = metadata
        self._persist()

    def step_fail(self, name: str, exc: BaseException) -> None:
        """Call when a step raises an exception."""
        rec = self._steps.get(name)
        now_ts = time.time()
        if rec is None:
            rec = _StepRecord(name=name, status="failed",
                              started_at=_now_iso(), started_ts=now_ts)
            self._steps[name] = rec
        elapsed = now_ts - rec.started_ts
        rec.status       = "failed"
        rec.finished_at  = _now_iso()
        rec.duration_sec = elapsed
        rec.error        = f"{type(exc).__name__}: {exc}"
        logger.debug(f"Step '{name}' failed after {elapsed:.1f}s: {exc}")
        self._persist()

    # ── pipeline lifecycle ────────────────────────────────────────────────────

    def finish(self, status: str = "success", **extra) -> dict:
        """Mark the pipeline as finished and return the full trace dict."""
        self._status      = status
        self._finished_at = _now_iso()
        self._extra       = extra
        trace = self._persist()
        logger.info(
            f"Pipeline trace: {status} — "
            f"{len(self._steps)} steps, "
            f"{time.time() - self._started_ts:.1f}s total"
        )
        return trace

    # ── persistence ───────────────────────────────────────────────────────────

    def _to_dict(self) -> dict:
        total_sec = time.time() - self._started_ts
        return {
            "pipeline":    self.pipeline,
            "started_at":  self._started_at,
            "finished_at": self._finished_at,
            "total_sec":   round(total_sec, 3),
            "status":      self._status,
            "steps":       [r.to_dict() for r in self._steps.values()],
            **self._extra,
        }

    def _persist(self) -> dict:
        trace = self._to_dict()
        path  = self.run_dir / _TRACE_FILE
        tmp   = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(trace, indent=2))
            tmp.replace(path)
        except Exception as exc:
            logger.warning(f"PipelineObserver: could not write trace: {exc}")
        return trace
