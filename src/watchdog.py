"""Pipeline watchdog: alert via Slack when no successful run has been seen recently.

Scans output_dir for pipeline_trace.json files, finds the most recent successful
run for the requested pipeline, and fires a Slack alert if it's older than
max_hours.

Single-pipeline check (existing behaviour):
    python src/watchdog.py --pipeline daily --max-hours 36

Full health heartbeat — checks all known pipelines and posts one status board:
    python src/watchdog.py --heartbeat
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config import settings
from src.slack_notifier import notify_heartbeat, notify_watchdog_alert

# Expected maximum gap (hours) before we consider a pipeline stale.
# Derived from each workflow's cron schedule + a generous buffer:
#   daily   Mon–Sat 08:00 UTC  → Sat→Mon gap = 48 h  → threshold 50 h
#   weekly  Mon/Wed/Fri 09:00  → Fri→Mon gap = 72 h  → threshold 80 h
#   digest  Sun 10:00 UTC      → weekly gap = 168 h  → threshold 180 h
#   topic   Tue/Thu 10:00 UTC  → Thu→Tue gap = 96 h  → threshold 100 h
#   shorts  daily 10:00 UTC    → gap = 24 h          → threshold 30 h
#   breaking every 30 min      → gap = 0.5 h         → threshold 2 h
_PIPELINE_THRESHOLDS: dict[str, float] = {
    "daily":    50.0,
    "weekly":   80.0,
    "digest":  180.0,
    "topic":   100.0,
    "shorts":   30.0,
    "breaking":  2.0,
}


@dataclass
class PipelineHealth:
    """Health snapshot for a single pipeline."""
    pipeline:        str
    last_run_at:     dt.datetime | None
    hours_since:     float          # math.inf when no run found
    threshold_hours: float
    is_stale:        bool

    @property
    def status_emoji(self) -> str:
        if self.last_run_at is None:
            return "⚫"
        if self.is_stale:
            return "🔴"
        if self.hours_since > self.threshold_hours * 0.75:
            return "⚠️"
        return "✅"


def find_last_success(
    output_dir: Path,
    pipeline: str = "daily",
) -> Optional[dt.datetime]:
    """Return the UTC datetime of the most recent successful run, or None."""
    latest: Optional[dt.datetime] = None

    for trace_path in output_dir.glob("*/pipeline_trace.json"):
        try:
            data = json.loads(trace_path.read_text())
        except Exception:
            continue
        if data.get("pipeline") != pipeline:
            continue
        if data.get("status") != "success":
            continue
        raw_ts = data.get("finished_at")
        if not raw_ts:
            continue
        try:
            ts = dt.datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
        except ValueError:
            continue
        if latest is None or ts > latest:
            latest = ts

    return latest


def check(
    pipeline: str = "daily",
    max_hours: float = 36.0,
    output_dir: Optional[Path] = None,
    now: Optional[dt.datetime] = None,
) -> bool:
    """Check whether the pipeline has run recently and alert if not.

    Returns True when an alert was sent, False when everything looks healthy.
    """
    output_dir = output_dir or settings.output_dir
    now = now or dt.datetime.now(dt.timezone.utc)

    last = find_last_success(output_dir, pipeline=pipeline)

    if last is None:
        logger.warning(
            f"Watchdog: no successful {pipeline!r} run found in {output_dir}"
        )
        notify_watchdog_alert(last_run_at=None, hours_since=float("inf"), pipeline=pipeline)
        return True

    hours_since = (now - last).total_seconds() / 3600
    logger.info(
        f"Watchdog: last {pipeline!r} success {hours_since:.1f}h ago ({last.isoformat()})"
    )

    if hours_since > max_hours:
        notify_watchdog_alert(last_run_at=last, hours_since=hours_since, pipeline=pipeline)
        return True

    return False


def check_all(
    output_dir: Optional[Path] = None,
    now: Optional[dt.datetime] = None,
    thresholds: Optional[dict[str, float]] = None,
) -> list[PipelineHealth]:
    """Return a PipelineHealth snapshot for every known pipeline.

    *thresholds* defaults to _PIPELINE_THRESHOLDS; pass a custom dict in tests.
    """
    output_dir = output_dir or settings.output_dir
    now        = now        or dt.datetime.now(dt.timezone.utc)
    thresholds = thresholds or _PIPELINE_THRESHOLDS

    statuses: list[PipelineHealth] = []
    for pipeline, max_h in thresholds.items():
        last       = find_last_success(output_dir, pipeline=pipeline)
        if last is None:
            h_since = float("inf")
        else:
            h_since = (now - last).total_seconds() / 3600
        statuses.append(PipelineHealth(
            pipeline        = pipeline,
            last_run_at     = last,
            hours_since     = h_since,
            threshold_hours = max_h,
            is_stale        = h_since > max_h,
        ))
    return statuses


def heartbeat(
    output_dir: Optional[Path] = None,
    now: Optional[dt.datetime] = None,
    thresholds: Optional[dict[str, float]] = None,
) -> list[PipelineHealth]:
    """Post a full health status board to Slack and alert for any stale pipelines.

    Returns the list of PipelineHealth objects for inspection / testing.
    """
    statuses = check_all(output_dir=output_dir, now=now, thresholds=thresholds)

    # Single consolidated board — always posted so operators see the full picture
    notify_heartbeat(statuses)

    # Individual stale alerts so each pipeline gets its own red card
    for s in statuses:
        if s.is_stale:
            notify_watchdog_alert(
                last_run_at = s.last_run_at,
                hours_since = s.hours_since,
                pipeline    = s.pipeline,
            )
            logger.warning(
                f"Heartbeat: {s.pipeline!r} stale ({s.hours_since:.1f}h "
                f"> threshold {s.threshold_hours}h)"
            )
        else:
            logger.info(
                f"Heartbeat: {s.pipeline!r} OK "
                f"({s.hours_since:.1f}h ago)"
                if s.last_run_at else
                f"Heartbeat: {s.pipeline!r} no run found — below stale threshold"
            )

    return statuses


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watchdog: check pipeline health")
    parser.add_argument("--pipeline",  default="daily",  help="Pipeline name to check")
    parser.add_argument("--max-hours", type=float, default=36.0,
                        help="Alert threshold in hours (default 36 to cover Sunday gap)")
    parser.add_argument("--heartbeat", action="store_true",
                        help="Check all pipelines and post a consolidated health board")
    args = parser.parse_args()

    if args.heartbeat:
        heartbeat()
    else:
        check(pipeline=args.pipeline, max_hours=args.max_hours)
