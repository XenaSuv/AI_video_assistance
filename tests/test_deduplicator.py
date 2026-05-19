"""Tests for SeenStories deduplicator — including concurrent access."""
from __future__ import annotations

import datetime as dt
import sqlite3
import threading
from pathlib import Path

import pytest

from src.deduplicator import _MIGRATIONS, _SCHEMA_VERSION, SeenStories
from src.scraper import NewsItem


def _story(url: str, title: str = "Title", source: str = "Src") -> NewsItem:
    return NewsItem(
        url=url,
        title=title,
        source=source,
        summary="summary",
        authors=[],
        published=dt.date.today().isoformat(),
    )


@pytest.fixture()
def db(tmp_path: Path) -> SeenStories:
    return SeenStories(tmp_path / "seen.db")


# ── basic behaviour ──────────────────────────────────────────────────────────

class TestFilterNew:
    def test_fresh_stories_pass_through(self, db: SeenStories) -> None:
        items = [_story("https://a.com"), _story("https://b.com")]
        assert db.filter_new(items) == items

    def test_seen_stories_are_filtered(self, db: SeenStories) -> None:
        seen = _story("https://seen.com")
        db.mark_featured([seen])
        # Need >= _MIN_FRESH (5) fresh stories so the fallback-to-full-list
        # branch does not trigger and hide the deduplication result.
        fresh = [_story(f"https://fresh-{i}.com") for i in range(6)]
        result = db.filter_new([seen] + fresh)
        assert all(i.url != "https://seen.com" for i in result)

    def test_empty_input_returns_empty(self, db: SeenStories) -> None:
        assert db.filter_new([]) == []

    def test_all_seen_falls_back_to_full_list(self, db: SeenStories) -> None:
        # When every story is seen, pipeline must still run — full list returned.
        items = [_story(f"https://{i}.com") for i in range(3)]
        db.mark_featured(items)
        result = db.filter_new(items)
        assert result == items

    def test_mark_featured_idempotent(self, db: SeenStories) -> None:
        item = _story("https://a.com")
        db.mark_featured([item])
        db.mark_featured([item])  # should not raise
        assert db.stats()["in_ttl_window"] == 1


class TestTTL:
    def test_expired_stories_are_eligible_again(self, tmp_path: Path) -> None:
        db = SeenStories(tmp_path / "seen.db", ttl_days=7)
        item = _story("https://old.com")

        # Insert a row dated before the TTL window.
        conn = sqlite3.connect(str(tmp_path / "seen.db"))
        old_date = (dt.date.today() - dt.timedelta(days=8)).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO seen_stories VALUES (?,?,?,?)",
            (item.url, item.title, item.source, old_date),
        )
        conn.commit()
        conn.close()

        result = db.filter_new([item])
        assert item in result

    def test_recent_story_within_ttl_is_filtered(self, tmp_path: Path) -> None:
        db = SeenStories(tmp_path / "seen.db", ttl_days=7)
        seen = _story("https://recent.com")
        db.mark_featured([seen])
        fresh = [_story(f"https://fresh-{i}.com") for i in range(6)]
        result = db.filter_new([seen] + fresh)
        assert all(i.url != "https://recent.com" for i in result)


class TestStats:
    def test_stats_returns_counts(self, db: SeenStories) -> None:
        items = [_story(f"https://{i}.com") for i in range(5)]
        db.mark_featured(items)
        s = db.stats()
        assert s["total_seen"] == 5
        assert s["in_ttl_window"] == 5
        assert s["ttl_days"] == 30

    def test_recent_titles(self, db: SeenStories) -> None:
        items = [_story(f"https://{i}.com", title=f"Story {i}") for i in range(3)]
        db.mark_featured(items)
        titles = db.recent_titles(limit=10)
        assert len(titles) == 3
        assert all(isinstance(t, str) for t in titles)


# ── WAL / locking ────────────────────────────────────────────────────────────

class TestWALAndConcurrency:
    def test_wal_mode_enabled(self, db: SeenStories) -> None:
        conn = sqlite3.connect(str(db.db_path))
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        conn.close()
        assert mode == "wal"

    def test_concurrent_readers_do_not_block_each_other(
        self, db: SeenStories, tmp_path: Path
    ) -> None:
        """Two threads reading simultaneously must both succeed."""
        items = [_story(f"https://{i}.com") for i in range(10)]
        db.mark_featured(items)

        errors: list[Exception] = []

        def read() -> None:
            try:
                db.filter_new(items)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=read) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent reads failed: {errors}"

    def test_concurrent_writers_do_not_corrupt_data(
        self, tmp_path: Path
    ) -> None:
        """Two threads writing different stories must both persist cleanly."""
        db = SeenStories(tmp_path / "seen.db")
        errors: list[Exception] = []

        def write(offset: int) -> None:
            try:
                stories = [_story(f"https://w{offset}-{i}.com") for i in range(5)]
                db.mark_featured(stories)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(n * 10,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent writes failed: {errors}"
        # All 20 stories must be recorded (4 threads × 5 stories each).
        assert db.stats()["total_seen"] == 20

    def test_simulated_daily_and_breaking_overlap(self, tmp_path: Path) -> None:
        """Simulate daily pipeline (writer) and breaking detector (reader) running
        simultaneously — the canonical race condition this fix addresses."""
        db_path = tmp_path / "seen.db"
        errors: list[Exception] = []
        barrier = threading.Barrier(2)  # both threads start at same moment

        def daily_pipeline() -> None:
            seen = SeenStories(db_path)
            barrier.wait()
            stories = [_story(f"https://daily-{i}.com") for i in range(20)]
            try:
                seen.mark_featured(seen.filter_new(stories))
            except Exception as exc:
                errors.append(exc)

        def breaking_detector() -> None:
            seen = SeenStories(db_path)
            barrier.wait()
            stories = [_story(f"https://breaking-{i}.com") for i in range(5)]
            try:
                seen.filter_new(stories)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=daily_pipeline)
        t2 = threading.Thread(target=breaking_detector)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Overlap simulation failed: {errors}"


# ── schema versioning ────────────────────────────────────────────────────────

def _user_version(db_path: Path) -> int:
    """Read PRAGMA user_version directly without going through SeenStories."""
    conn = sqlite3.connect(str(db_path))
    v = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    return v


def _index_exists(db_path: Path, index_name: str) -> bool:
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?", (index_name,)
    ).fetchone()
    conn.close()
    return row is not None


class TestSchemaVersion:
    def test_constants_consistent(self):
        """_SCHEMA_VERSION must equal len(_MIGRATIONS) — each version has one entry."""
        assert _SCHEMA_VERSION == len(_MIGRATIONS)

    def test_fresh_db_has_current_schema_version(self, tmp_path):
        SeenStories(tmp_path / "seen.db")
        assert _user_version(tmp_path / "seen.db") == _SCHEMA_VERSION

    def test_schema_version_stable_on_second_open(self, tmp_path):
        """Re-opening a fully migrated DB must not change user_version."""
        db_path = tmp_path / "seen.db"
        SeenStories(db_path)
        SeenStories(db_path)
        assert _user_version(db_path) == _SCHEMA_VERSION

    def test_legacy_db_at_v0_is_migrated(self, tmp_path):
        """Legacy DB with the table but user_version=0 is upgraded in-place."""
        db_path = tmp_path / "seen.db"
        # Simulate a legacy database: table exists, but no user_version set (= 0).
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE seen_stories (
            url TEXT PRIMARY KEY, title TEXT NOT NULL,
            source TEXT NOT NULL, featured_date TEXT NOT NULL
        )""")
        conn.commit()
        conn.close()
        assert _user_version(db_path) == 0

        SeenStories(db_path)
        assert _user_version(db_path) == _SCHEMA_VERSION

    def test_partial_migration_v1_to_current(self, tmp_path):
        """DB at version 1 (no index yet) is upgraded to current version."""
        db_path = tmp_path / "seen.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE seen_stories (
            url TEXT PRIMARY KEY, title TEXT NOT NULL,
            source TEXT NOT NULL, featured_date TEXT NOT NULL
        )""")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()
        assert _user_version(db_path) == 1

        SeenStories(db_path)
        assert _user_version(db_path) == _SCHEMA_VERSION

    def test_v2_index_exists_after_fresh_init(self, tmp_path):
        SeenStories(tmp_path / "seen.db")
        assert _index_exists(tmp_path / "seen.db", "idx_seen_featured_date")

    def test_v2_index_created_when_upgrading_from_v1(self, tmp_path):
        db_path = tmp_path / "seen.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE seen_stories (
            url TEXT PRIMARY KEY, title TEXT NOT NULL,
            source TEXT NOT NULL, featured_date TEXT NOT NULL
        )""")
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
        conn.close()
        assert not _index_exists(db_path, "idx_seen_featured_date")

        SeenStories(db_path)
        assert _index_exists(db_path, "idx_seen_featured_date")

    def test_functional_after_migration_from_legacy(self, tmp_path):
        """Full filter/mark cycle must work on a database that was just migrated."""
        db_path = tmp_path / "seen.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE seen_stories (
            url TEXT PRIMARY KEY, title TEXT NOT NULL,
            source TEXT NOT NULL, featured_date TEXT NOT NULL
        )""")
        conn.commit()
        conn.close()

        seen = SeenStories(db_path)
        # Mark 1 item seen; 5 remain fresh (≥ _MIN_FRESH=5) so the fallback
        # to the full list doesn't trigger and dedup is observable.
        items = [_story(f"https://migrated-{i}.com") for i in range(6)]
        seen.mark_featured(items[:1])
        fresh = seen.filter_new(items)
        assert len(fresh) == 5   # one already-seen is filtered

    def test_existing_data_preserved_after_migration(self, tmp_path):
        """Rows inserted before the migration must still be queryable afterwards."""
        db_path = tmp_path / "seen.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""CREATE TABLE seen_stories (
            url TEXT PRIMARY KEY, title TEXT NOT NULL,
            source TEXT NOT NULL, featured_date TEXT NOT NULL
        )""")
        conn.execute(
            "INSERT INTO seen_stories VALUES (?,?,?,?)",
            ("https://old.com", "Old Title", "Src", dt.date.today().isoformat()),
        )
        conn.commit()
        conn.close()

        seen = SeenStories(db_path)
        assert seen.stats()["total_seen"] == 1
        titles = seen.recent_titles()
        assert "Old Title" in titles
