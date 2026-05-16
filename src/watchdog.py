"""Pipeline watchdog: alert via Slack when no successful run has been seen recently.

Scans output_dir for pipeline_trace.json files, finds the most recent successful
run for the requested pipeline, and fires a Slack alert if it's older than
max_hours.

Typical usage (GitHub Actions every 6 h):
    python src/watchdog.py --pipeline daily --max-hours 36
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger

from config import settings
from src.slack_notifier import notify_watchdog_alert


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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Watchdog: check pipeline health")
    parser.add_argument("--pipeline",  default="daily",  help="Pipeline name to check")
    parser.add_argument("--max-hours", type=float, default=36.0,
                        help="Alert threshold in hours (default 36 to cover Sunday gap)")
    args = parser.parse_args()

    check(pipeline=args.pipeline, max_hours=args.max_hours)
