"""Daemon process that runs the pipeline once a day at settings.daily_run_hour_utc.

Includes automatic retry on failure (up to 3 attempts, 30 min apart).
Run under systemd, tmux, or docker to keep alive.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.main import run_pipeline


MAX_RETRIES = 3
RETRY_DELAY_SEC = 30 * 60


def job_with_retry() -> None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(f"Daily run: attempt {attempt}/{MAX_RETRIES}")
            run_pipeline()
            logger.info("Daily run succeeded")
            return
        except Exception as e:
            logger.exception(f"Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                logger.info(f"Retrying in {RETRY_DELAY_SEC // 60} min...")
                time.sleep(RETRY_DELAY_SEC)
            else:
                logger.error("All retries exhausted; giving up until tomorrow")


def main():
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        job_with_retry,
        CronTrigger(hour=settings.daily_run_hour_utc, minute=0),
        id="daily_news_pipeline",
        name="Daily AI News Video",
        misfire_grace_time=60 * 30,
    )
    logger.info(
        f"Scheduler started. Next run: daily at "
        f"{settings.daily_run_hour_utc:02d}:00 UTC"
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    main()
