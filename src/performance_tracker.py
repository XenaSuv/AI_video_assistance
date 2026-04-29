from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings


def _get_db_path() -> Path:
    """Return the performance database path."""
    db_path = settings.data_dir / "performance.json"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def _load_db() -> list[dict[str, Any]]:
    """Load performance records from disk."""
    db_path = _get_db_path()
    if not db_path.exists():
        return []
    try:
        with open(db_path, "r") as f:
            return json.load(f) or []
    except Exception as exc:
        logger.warning(f"Failed to load performance DB: {exc}")
        return []


def _save_db(data: list[dict[str, Any]]) -> None:
    """Save performance records to disk."""
    db_path = _get_db_path()
    try:
        with open(db_path, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.error(f"Failed to save performance DB: {exc}")


def save_result(video_id: str, hook: str, title: str, description: str = "") -> None:
    """Record a newly published video with its hook and title."""
    record = {
        "video_id": video_id,
        "hook": hook,
        "title": title,
        "description": description,
        "timestamp": datetime.utcnow().isoformat(),
        "updated": datetime.utcnow().isoformat(),
        "views": 0,
        "likes": 0,
        "comments": 0,
        "shares": 0,
        "ctr": None,
        "watch_time_sec": None,
        "avg_view_duration_sec": None,
        "avg_view_percentage": None,
    }
    data = _load_db()
    data.append(record)
    _save_db(data)
    logger.info(f"Performance record saved: video_id={video_id}, hook={hook[:50]}")


def update_metrics(
    video_id: str,
    views: int | None = None,
    likes: int | None = None,
    comments: int | None = None,
    shares: int | None = None,
    ctr: float | None = None,
    watch_time_sec: float | None = None,
    avg_view_duration_sec: float | None = None,
    avg_view_percentage: float | None = None,
) -> None:
    """Update performance metrics for a published video."""
    data = _load_db()
    for record in data:
        if record["video_id"] == video_id:
            if views is not None:
                record["views"] = views
            if likes is not None:
                record["likes"] = likes
            if comments is not None:
                record["comments"] = comments
            if shares is not None:
                record["shares"] = shares
            if ctr is not None:
                record["ctr"] = ctr
            if watch_time_sec is not None:
                record["watch_time_sec"] = watch_time_sec
            if avg_view_duration_sec is not None:
                record["avg_view_duration_sec"] = avg_view_duration_sec
            if avg_view_percentage is not None:
                record["avg_view_percentage"] = avg_view_percentage
            record["updated"] = datetime.utcnow().isoformat()
            _save_db(data)
            logger.info(f"Performance metrics updated: video_id={video_id}")
            return
    logger.warning(f"Video record not found for update: video_id={video_id}")


def get_top_hooks(limit: int = 10) -> list[dict[str, Any]]:
    """Return the best-performing hooks by likes/views ratio."""
    data = _load_db()
    scored = []
    for record in data:
        if record["views"] > 0:
            engagement = (record["likes"] + record["comments"] * 2 + record["shares"] * 3) / record["views"]
            scored.append({
                "hook": record["hook"],
                "engagement": engagement,
                "views": record["views"],
                "ctr": record.get("ctr"),
                "avg_view_percentage": record.get("avg_view_percentage"),
            })
    scored.sort(key=lambda x: x["engagement"], reverse=True)
    return scored[:limit]


def get_top_titles(limit: int = 10) -> list[dict[str, Any]]:
    """Return the best-performing titles by CTR."""
    data = _load_db()
    scored = [
        {
            "title": r["title"],
            "ctr": r.get("ctr") or 0,
            "views": r["views"],
            "likes": r["likes"],
            "avg_view_percentage": r.get("avg_view_percentage"),
        }
        for r in data
        if r["views"] > 0
    ]
    scored.sort(key=lambda x: x["ctr"] or 0, reverse=True)
    return scored[:limit]


def get_stats_by_keyword(keyword: str, field: str = "hook") -> dict[str, Any]:
    """Get aggregate stats for videos containing a keyword in hook/title/description."""
    data = _load_db()
    matching = [
        r
        for r in data
        if keyword.lower() in r.get(field, "").lower()
    ]
    if not matching:
        return {"keyword": keyword, "count": 0}
    total_views = sum(r["views"] for r in matching)
    total_likes = sum(r["likes"] for r in matching)
    avg_ctr = sum(r.get("ctr") or 0 for r in matching) / len(matching) if matching else 0
    return {
        "keyword": keyword,
        "count": len(matching),
        "total_views": total_views,
        "total_likes": total_likes,
        "avg_engagement": (total_likes / total_views * 100) if total_views > 0 else 0,
        "avg_ctr": avg_ctr,
        "avg_view_percentage": sum(r.get("avg_view_percentage") or 0 for r in matching) / len(matching) if matching else 0,
    }


def get_summary() -> dict[str, Any]:
    """Return high-level performance summary."""
    data = _load_db()
    if not data:
        return {"total_videos": 0, "total_views": 0}
    total_views = sum(r["views"] for r in data)
    total_likes = sum(r["likes"] for r in data)
    total_engagement = total_likes / total_views * 100 if total_views > 0 else 0
    return {
        "total_videos": len(data),
        "total_views": total_views,
        "total_likes": total_likes,
        "avg_engagement_percent": total_engagement,
        "videos_with_metrics": len([r for r in data if r["views"] > 0]),
        "avg_ctr": sum(r.get("ctr") or 0 for r in data) / len(data) if data else 0,
        "avg_view_percentage": sum(r.get("avg_view_percentage") or 0 for r in data) / len(data) if data else 0,
    }


def list_records() -> list[dict[str, Any]]:
    """Return all performance records."""
    return _load_db()


def get_record(video_id: str) -> dict[str, Any] | None:
    """Return a saved performance record by video_id."""
    data = _load_db()
    for record in data:
        if record["video_id"] == video_id:
            return record
    return None
