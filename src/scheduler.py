"""Daemon process that runs all video pipelines on their schedules.

Jobs:
  daily_news    — every day at DAILY_RUN_HOUR_UTC (default 08:00 UTC)
  topic_tuesday — every Tuesday  at TOPIC_RUN_HOUR_UTC (default 10:00 UTC)
  topic_thursday— every Thursday at TOPIC_RUN_HOUR_UTC (default 10:00 UTC)

Each job retries up to 3 times, 30 minutes apart, before giving up.

Run under systemd, tmux, or docker:
    python src/scheduler.py
    python src/scheduler.py --list      # print next run times and exit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

MAX_RETRIES    = 3
RETRY_DELAY_SEC = 30 * 60


# ── Retry wrapper ─────────────────────────────────────────────────────────────

def _run_with_retry(label: str, fn, **kwargs) -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"[{label}] attempt {attempt}/{MAX_RETRIES}")
            fn(**kwargs)
            logger.info(f"[{label}] finished successfully")
            return
        except Exception as exc:
            logger.exception(f"[{label}] attempt {attempt} failed: {exc}")
            if attempt < MAX_RETRIES:
                logger.info(f"[{label}] retrying in {RETRY_DELAY_SEC // 60} min…")
                time.sleep(RETRY_DELAY_SEC)
            else:
                logger.error(f"[{label}] all retries exhausted — skipping until next scheduled run")


# ── Job functions ─────────────────────────────────────────────────────────────

def job_daily_news() -> None:
    from src.main import run_pipeline
    _run_with_retry("daily_news", run_pipeline)


def job_topic() -> None:
    from src.topic_main import run_topic_pipeline
    _run_with_retry("topic", run_topic_pipeline)


# ── Scheduler setup ───────────────────────────────────────────────────────────

def build_scheduler() -> BlockingScheduler:
    scheduler = BlockingScheduler(timezone="UTC")

    # Daily AI news — every day
    scheduler.add_job(
        job_daily_news,
        CronTrigger(hour=settings.daily_run_hour_utc, minute=0),
        id="daily_news",
        name="Daily AI News Video",
        misfire_grace_time=30 * 60,
    )

    # Topic deep-dive — Tuesday (dow=1)
    scheduler.add_job(
        job_topic,
        CronTrigger(day_of_week="tue", hour=settings.topic_run_hour_utc, minute=0),
        id="topic_tuesday",
        name="Topic Deep-Dive (Tuesday)",
        misfire_grace_time=30 * 60,
    )

    # Topic deep-dive — Thursday (dow=3)
    scheduler.add_job(
        job_topic,
        CronTrigger(day_of_week="thu", hour=settings.topic_run_hour_utc, minute=0),
        id="topic_thursday",
        name="Topic Deep-Dive (Thursday)",
        misfire_grace_time=30 * 60,
    )

    return scheduler


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Video pipeline scheduler")
    parser.add_argument(
        "--list", action="store_true",
        help="Print scheduled jobs and next run times, then exit",
    )
    args = parser.parse_args()

    scheduler = build_scheduler()

    if args.list:
        print("\nScheduled jobs:")
        for job in scheduler.get_jobs():
            next_run = job.next_run_time
            print(f"  {job.id:<20} {job.name}")
            print(f"  {'':20} next: {next_run or '(not yet calculated)'}")
        scheduler.shutdown(wait=False)
        return

    logger.info(
        f"Scheduler starting.\n"
        f"  Daily news  : every day   at {settings.daily_run_hour_utc:02d}:00 UTC\n"
        f"  Topic videos: Tue & Thu   at {settings.topic_run_hour_utc:02d}:00 UTC"
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
