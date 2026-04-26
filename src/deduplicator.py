"""Track featured stories in a local SQLite DB to prevent cross-day repeats.

Stories are excluded for a rolling TTL window (default 30 days).  After that
window they become eligible again — so a paper from 6 weeks ago can reappear
if it's still genuinely trending.

Typical usage in the pipeline:
    seen = SeenStories(settings.data_dir / "seen_stories.db")
    news = seen.filter_new(news)          # drop already-featured items
    ...build video...
    seen.mark_featured(news)              # record before upload
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.scraper import NewsItem

_DEFAULT_TTL_DAYS = 30
_MIN_FRESH = 5   # warn (but don't abort) if fewer fresh stories than this


class SeenStories:
    def __init__(self, db_path: Path, ttl_days: int = _DEFAULT_TTL_DAYS) -> None:
        self.db_path = db_path
        self.ttl_days = ttl_days
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # --------------------- internal ---------------------

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS seen_stories (
                    url           TEXT PRIMARY KEY,
                    title         TEXT NOT NULL,
                    source        TEXT NOT NULL,
                    featured_date TEXT NOT NULL
                )
            """)
            conn.commit()

    def _cutoff(self) -> str:
        return (dt.date.today() - dt.timedelta(days=self.ttl_days)).isoformat()

    # --------------------- public API ---------------------

    def filter_new(self, items: list[NewsItem]) -> list[NewsItem]:
        """Return only items not featured within the TTL window.

        Never returns an empty list — if everything was seen recently, the
        full list is returned with a warning so the pipeline can still run.
        """
        if not items:
            return items

        with self._connect() as conn:
            seen_urls: set[str] = {
                row[0]
                for row in conn.execute(
                    "SELECT url FROM seen_stories WHERE featured_date > ?",
                    (self._cutoff(),),
                )
            }

        fresh = [item for item in items if item.url not in seen_urls]
        skipped = len(items) - len(fresh)

        if skipped:
            logger.info(
                f"Dedup: {skipped} stories skipped (seen in last {self.ttl_days}d), "
                f"{len(fresh)} fresh"
            )

        if len(fresh) < _MIN_FRESH:
            logger.warning(
                f"Only {len(fresh)} fresh stories after dedup — "
                f"falling back to full list to avoid thin episode"
            )
            return items

        return fresh

    def mark_featured(self, items: list[NewsItem]) -> None:
        """Record *items* as featured today.  Call after a successful video build."""
        today = dt.date.today().isoformat()
        with self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO seen_stories
                       (url, title, source, featured_date)
                   VALUES (?, ?, ?, ?)""",
                [(item.url, item.title, item.source, today) for item in items],
            )
            conn.commit()
        logger.info(f"Marked {len(items)} stories as featured on {today}")

    def stats(self) -> dict:
        """Return counts useful for logging / monitoring."""
        with self._connect() as conn:
            total   = conn.execute("SELECT COUNT(*) FROM seen_stories").fetchone()[0]
            in_ttl  = conn.execute(
                "SELECT COUNT(*) FROM seen_stories WHERE featured_date > ?",
                (self._cutoff(),),
            ).fetchone()[0]
        return {"total_seen": total, "in_ttl_window": in_ttl, "ttl_days": self.ttl_days}
