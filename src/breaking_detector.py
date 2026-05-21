"""Detect breaking AI news from official company sources.

Runs every 30 minutes via breaking.yml. When GITHUB_OUTPUT is set, writes
has_breaking=true/false so the publish job can decide whether to run.

Breaking criteria (ALL must pass):
  1. From an official company source (score >= 4.5)
  2. Published today or yesterday
  3. Title contains at least one importance keyword (launches, releases, etc.)
  4. URL not already in data/breaking_seen.json (not previously encountered)
  5. Circuit breaker: >= 4 h since last breaking video, max 2 per calendar day
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.scraper import NewsItem, scrape_official_sources

_SEEN_FILE   = "breaking_seen.json"
_COOLDOWN_H  = 4    # hours between breaking videos
_MAX_PER_DAY = 2    # max breaking news videos per calendar day

# Title keywords that indicate a meaningful announcement rather than a blog op-ed
_IMPORTANCE_KW = [
    "release", "released", "releases", "launching", "launched", "launches",
    "introduces", "introducing", "introduction",
    "announces", "announcing", "announcement",
    "new model", "new api", "open source", "open-source",
    "gpt-", "claude", "gemini", "llama", "grok", "mistral", "deepseek",
    "available now", "available today", "starting today",
    "breakthrough", "milestone", "partnership",
]


# ─────────────────── Seen-state helpers ───────────────────

def _seen_path(data_dir: Path) -> Path:
    return data_dir / _SEEN_FILE


def _load_seen(data_dir: Path) -> dict[str, Any]:
    p = _seen_path(data_dir)
    if p.exists():
        try:
            data: dict[str, Any] = json.loads(p.read_text())
            return data
        except Exception as exc:
            logger.debug(f"BreakingDetector._load_seen: {exc}")
    return {"seen_urls": [], "videos": []}


def _save_seen(data_dir: Path, seen: dict[str, Any]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    _seen_path(data_dir).write_text(json.dumps(seen, indent=2))


# ─────────────────── Filters ───────────────────

def _is_important(title: str) -> bool:
    low = title.lower()
    return any(kw in low for kw in _IMPORTANCE_KW)


def _is_recent(published_iso: str) -> bool:
    """True if the item was published today or yesterday."""
    try:
        pub = date.fromisoformat(published_iso[:10])
    except (ValueError, TypeError):
        return True  # unparseable → assume recent rather than miss it
    return pub >= date.today() - timedelta(days=1)


def _circuit_breaker_ok(seen: dict[str, Any]) -> bool:
    """Return False if we've already published too many breaking videos today
    or published one within the cooldown window."""
    videos = seen.get("videos", [])
    today_str = date.today().isoformat()

    today_count = sum(1 for v in videos if v.get("published_at", "")[:10] == today_str)
    if today_count >= _MAX_PER_DAY:
        logger.info(f"Breaking: daily limit reached ({today_count}/{_MAX_PER_DAY})")
        return False

    if videos:
        last = max(videos, key=lambda v: v.get("published_at", ""))
        last_dt = datetime.fromisoformat(last["published_at"])
        if last_dt.tzinfo is None:
            last_dt = last_dt.replace(tzinfo=timezone.utc)
        age_h = (datetime.now(timezone.utc) - last_dt).total_seconds() / 3600
        if age_h < _COOLDOWN_H:
            logger.info(f"Breaking: cooldown active ({age_h:.1f}h < {_COOLDOWN_H}h)")
            return False

    return True


# ─────────────────── Public API ───────────────────

def check_for_breaking(data_dir: Path | None = None) -> NewsItem | None:
    """Scrape official feeds and return the best breaking item, or None."""
    data_dir = data_dir or settings.data_dir
    seen = _load_seen(data_dir)

    if not _circuit_breaker_ok(seen):
        return None

    items      = scrape_official_sources()
    seen_urls  = set(seen.get("seen_urls", []))
    new_urls   = list(seen_urls)
    candidates: list[NewsItem] = []

    for item in items:
        if item.url in seen_urls:
            continue
        new_urls.append(item.url)     # mark as seen regardless of importance

        if not _is_recent(item.published):
            logger.debug(f"Breaking skip (old): {item.title}")
            continue
        if not _is_important(item.title):
            logger.debug(f"Breaking skip (not important): {item.title}")
            continue
        candidates.append(item)

    seen["seen_urls"] = new_urls
    _save_seen(data_dir, seen)

    if not candidates:
        logger.info("Breaking: no new important items found")
        return None

    best = max(candidates, key=lambda x: x.score)
    logger.info(f"Breaking item: [{best.source}] {best.title}")
    return best


def mark_published(data_dir: Path, item: NewsItem, video_id: str = "") -> None:
    """Record a published breaking video (updates circuit breaker state)."""
    seen = _load_seen(data_dir)
    seen.setdefault("videos", []).append({
        "url":          item.url,
        "title":        item.title,
        "source":       item.source,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "video_id":     video_id,
    })
    _save_seen(data_dir, seen)


def mark_publish_failed(data_dir: Path, item: NewsItem, error: str = "") -> None:
    """Record a failed publish attempt in breaking_seen.json for audit and retry.

    Does NOT increment the circuit-breaker video count — a failed upload should
    not consume one of the day's allowed breaking slots.
    """
    seen = _load_seen(data_dir)
    seen.setdefault("failed_videos", []).append({
        "url":       item.url,
        "title":     item.title,
        "source":    item.source,
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "error":     error,
    })
    _save_seen(data_dir, seen)
    logger.warning(f"Breaking: publish failure recorded for '{item.title}'")


# ─────────────────── CLI ───────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Breaking news detector")
    ap.add_argument("--check", action="store_true",
                    help="Write has_breaking=true/false to $GITHUB_OUTPUT")
    ap.add_argument("--data-dir", default=None, help="Override data directory")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else settings.data_dir
    item = check_for_breaking(data_dir)

    if args.check:
        if item:
            lines = (
                f"has_breaking=true\n"
                f"breaking_title={item.title}\n"
                f"breaking_source={item.source}\n"
            )
            # Save item JSON so the publish job can load it without re-scraping
            (data_dir / "breaking_current.json").write_text(
                json.dumps(item.to_dict(), indent=2)
            )
        else:
            lines = "has_breaking=false\n"

        gh_output = os.environ.get("GITHUB_OUTPUT", "")
        if gh_output:
            with open(gh_output, "a") as f:
                f.write(lines)
        print(lines, end="")
        sys.exit(0)

    # Plain run (local debug)
    if item:
        print(f"BREAKING: [{item.source}] {item.title}\n{item.url}")
    else:
        print("No breaking news at this time.")
