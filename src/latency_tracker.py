"""Pipeline step latency tracking and slow-step detection.

Stores per-step durations across runs in data/latency_history.json,
computes P95 baselines, and surfaces steps that exceeded thresholds.

Usage (in _finalize):
    tracker = LatencyTracker(settings.data_dir)
    tracker.record(trace)
    slow = tracker.slow_steps(trace)
    if slow:
        notify_slow_steps(slow, pipeline="daily", date=date_str)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Soft thresholds per step (seconds).  Exceeded → Slack alert even on success.
STEP_THRESHOLDS: dict[str, float] = {
    "scrape":     120.0,   # 2 min
    "script":     300.0,   # 5 min
    "voice":      600.0,   # 10 min
    "video":      900.0,   # 15 min
    "subtitles":  120.0,   # 2 min
    "thumbnail":  120.0,   # 2 min
    "upload_en":  300.0,   # 5 min
    "upload_ru":  300.0,   # 5 min
}
TOTAL_THRESHOLD: float = 3600.0   # 1 hour end-to-end

_MAX_HISTORY: int = 30  # runs kept per step for P95


# ── helpers ───────────────────────────────────────────────────────────────────

def fmt_duration(sec: float) -> str:
    """Format seconds as '4m 12s' or '45s'."""
    m, s = divmod(int(sec), 60)
    return f"{m}m {s}s" if m else f"{s}s"


# ── core class ────────────────────────────────────────────────────────────────

class LatencyTracker:
    """Persist per-step latencies and detect threshold violations.

    Thread-safety: not needed — one tracker per pipeline run.
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = data_dir / "latency_history.json"
        self._history: dict[str, list[float]] = self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict[str, list[float]]:
        try:
            raw = json.loads(self._path.read_text())
            return {k: [float(v) for v in vals] for k, vals in raw.items()}
        except Exception:
            return {}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._history, indent=2))
            tmp.replace(self._path)
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    def record(self, trace: dict[str, Any]) -> None:
        """Append step durations from a finished pipeline trace."""
        for step in trace.get("steps", []):
            name = step.get("name")
            dur = step.get("duration_sec")
            if name and dur is not None and step.get("status") in ("done", "failed"):
                bucket = self._history.setdefault(name, [])
                bucket.append(float(dur))
                self._history[name] = bucket[-_MAX_HISTORY:]

        # also track total pipeline duration
        total = trace.get("total_sec")
        if total is not None:
            bucket = self._history.setdefault("_total", [])
            bucket.append(float(total))
            self._history["_total"] = bucket[-_MAX_HISTORY:]

        self._save()

    def p95(self, step: str) -> float | None:
        """Return the P95 latency (seconds) for *step* over the last 30 runs."""
        vals = sorted(self._history.get(step, []))
        if not vals:
            return None
        idx = min(int(len(vals) * 0.95), len(vals) - 1)
        return vals[idx]

    def slow_steps(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Return entries for steps (and total) that exceeded their threshold.

        Each entry: {name, duration_sec, threshold_sec, p95_sec | None}
        """
        slow: list[dict[str, Any]] = []

        for step in trace.get("steps", []):
            name = step.get("name", "")
            dur = step.get("duration_sec")
            if dur is None:
                continue
            threshold = STEP_THRESHOLDS.get(name)
            if threshold and dur > threshold:
                slow.append({
                    "name":          name,
                    "duration_sec":  dur,
                    "threshold_sec": threshold,
                    "p95_sec":       self.p95(name),
                })

        total = trace.get("total_sec", 0.0)
        if total and total > TOTAL_THRESHOLD:
            slow.append({
                "name":          "_total",
                "duration_sec":  total,
                "threshold_sec": TOTAL_THRESHOLD,
                "p95_sec":       self.p95("_total"),
            })

        return slow

    def step_summary(self, trace: dict[str, Any]) -> list[dict[str, Any]]:
        """Return timing rows for every completed step (for Slack success card)."""
        rows = []
        for step in trace.get("steps", []):
            name = step.get("name", "")
            dur = step.get("duration_sec")
            if dur is None:
                continue
            threshold = STEP_THRESHOLDS.get(name)
            rows.append({
                "name":         name,
                "duration_str": fmt_duration(dur),
                "slow":         bool(threshold and dur > threshold),
            })
        return rows
