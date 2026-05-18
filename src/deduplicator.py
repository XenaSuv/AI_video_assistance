"""Track featured stories in a local SQLite DB to prevent cross-day repeats.

Stories are excluded for a rolling TTL window (default 30 days).  After that
window they become eligible again — so a paper from 6 weeks ago can reappear
if it's still genuinely trending.

Schema versioning
-----------------
The DB header field ``PRAGMA user_version`` stores the current schema version.
``_MIGRATIONS`` is the authoritative migration list: index 0 is version 1, index 1
is version 2, etc.  On every startup ``_init_db`` applies only the migrations
that follow the stored version, so existing databases are upgraded in-place
without requiring a manual file deletion.

To add a new migration: append a SQL string to ``_MIGRATIONS`` and bump
``_SCHEMA_VERSION`` by 1.

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

# Bump _SCHEMA_VERSION and append one SQL string to _MIGRATIONS for each change.
# Migrations are applied once and never modified; they must be idempotent
# (use IF NOT EXISTS / IF EXISTS guards) so a repeated startup is a no-op.
_SCHEMA_VERSION = 2
_MIGRATIONS: list[str] = [
    # v1 — initial schema
    """CREATE TABLE IF NOT EXISTS seen_stories (
        url           TEXT PRIMARY KEY,
        title         TEXT NOT NULL,
        source        TEXT NOT NULL,
        featured_date TEXT NOT NULL
    )""",
    # v2 — index for O(log n) TTL filtering instead of full-table scan
    "CREATE INDEX IF NOT EXISTS idx_seen_featured_date ON seen_stories (featured_date)",
]


class SeenStories:
    def __init__(self, db_path: Path, ttl_days: int = _DEFAULT_TTL_DAYS) -> None:
        self.db_path = db_path
        self.ttl_days = ttl_days
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # --------------------- internal ---------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        # busy_timeout makes SQLite retry on SQLITE_BUSY at the C level
        # (complements Python-level timeout which retries the initial open).
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            # WAL allows concurrent readers while a writer is active.
            # Without it, any writer exclusively locks the file, causing
            # SQLITE_BUSY when daily and breaking-news pipelines overlap.
            # journal_mode is persistent in the DB file — only needs setting once.
            conn.execute("PRAGMA journal_mode=WAL")
            current = conn.execute("PRAGMA user_version").fetchone()[0]
            for target, sql in enumerate(_MIGRATIONS, start=1):
                if current < target:
                    conn.execute(sql)
                    # PRAGMA user_version cannot be parameterised; target is a
                    # trusted internal integer, never derived from user input.
                    conn.execute(f"PRAGMA user_version = {target}")
                    conn.commit()
                    logger.debug(f"DeduplicatorDB: schema upgraded to v{target}")

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

    def recent_titles(self, limit: int = 100) -> list[str]:
        """Return the most recent featured story titles for editorial history."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT title FROM seen_stories ORDER BY featured_date DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [row[0] for row in rows]
