from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from src.performance_tracker import _load_db


def top_hooks(limit: int = 5, min_views: int = 0) -> list[dict[str, Any]]:
    """Return the best-performing hooks by engagement (likes/views ratio)."""
    data = _load_db()
    scored = []
    for record in data:
        if record["views"] > min_views:
            engagement_rate = (record["likes"] + record["comments"] * 2) / max(record["views"], 1)
            scored.append({
                "hook": record["hook"],
                "views": record["views"],
                "likes": record["likes"],
                "engagement_rate": engagement_rate,
                "ctr": record.get("ctr") or 0,
            })
    scored.sort(key=lambda x: x["engagement_rate"], reverse=True)
    return scored[:limit]


def top_titles(limit: int = 5, min_views: int = 0) -> list[dict[str, Any]]:
    """Return the best-performing titles by CTR."""
    data = _load_db()
    scored = [
        {
            "title": r["title"],
            "ctr": r.get("ctr") or 0,
            "views": r["views"],
            "likes": r["likes"],
        }
        for r in data
        if r["views"] > min_views
    ]
    scored.sort(key=lambda x: x["ctr"], reverse=True)
    return scored[:limit]


def hook_keywords(min_views: int = 500, top_n: int = 15) -> list[tuple[str, int]]:
    """Return the most common words in high-performing hooks."""
    data = _load_db()
    words = []
    for record in data:
        if record["views"] > min_views:
            hook_words = record["hook"].lower().split()
            words.extend(hook_words)
    counter = Counter(words)
    stop_words = {"the", "a", "an", "and", "or", "is", "in", "to", "of", "this", "that"}
    filtered = [(w, c) for w, c in counter.most_common(top_n * 3) if w not in stop_words and len(w) > 2]
    return filtered[:top_n]


def top_keywords(min_views: int = 300, top_n: int = 10) -> list[tuple[str, int]]:
    """Return the most common words across hook/title/description text."""
    data = _load_db()
    words: list[str] = []
    stop_words = {
        "the", "a", "an", "and", "or", "is", "in", "to", "of", "this", "that",
        "with", "for", "from", "on", "at", "by", "as", "new", "ai",
    }
    for record in data:
        if record["views"] >= min_views:
            text = " ".join([
                record.get("hook", ""),
                record.get("title", ""),
                record.get("description", ""),
            ]).lower()
            words.extend([w.strip(".,!?()[]{}\"'”") for w in text.split() if len(w) > 2])
    counter = Counter(words)
    filtered = [(w, c) for w, c in counter.most_common(top_n * 3) if w not in stop_words]
    return filtered[:top_n]


def recent_records(days: int = 30) -> list[dict[str, Any]]:
    """Return records published within the last *days* days."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    records = []
    for record in _load_db():
        timestamp = record.get("timestamp")
        if not timestamp:
            continue
        try:
            record_time = datetime.fromisoformat(timestamp)
        except ValueError:
            continue
        if record_time >= cutoff:
            records.append(record)
    return records


def get_trend_report(days: int = 30) -> dict[str, Any]:
    """Return trend data for recent videos."""
    records = recent_records(days)
    if not records:
        return {"days": days, "total_videos": 0, "avg_ctr": 0, "avg_engagement": 0}
    total_views = sum(r["views"] for r in records)
    total_likes = sum(r["likes"] for r in records)
    avg_ctr = sum(r.get("ctr") or 0 for r in records) / len(records)
    avg_engagement = (total_likes / max(total_views, 1)) * 100 if total_views > 0 else 0
    top_hooks_list = top_hooks(limit=5, min_views=50)
    top_titles_list = top_titles(limit=5, min_views=50)
    return {
        "days": days,
        "total_videos": len(records),
        "total_views": total_views,
        "avg_ctr": avg_ctr,
        "avg_engagement_percent": avg_engagement,
        "top_hooks": [h["hook"] for h in top_hooks_list],
        "top_titles": [t["title"] for t in top_titles_list],
    }


def worst_hooks(limit: int = 5) -> list[dict[str, Any]]:
    """Return hooks with the lowest engagement."""
    data = _load_db()
    scored = [
        {
            "hook": r["hook"],
            "views": r["views"],
            "likes": r["likes"],
            "engagement_rate": (r["likes"] / max(r["views"], 1)) if r["views"] > 0 else 0,
        }
        for r in data
    ]
    scored.sort(key=lambda x: x["engagement_rate"])
    return scored[:limit]


def boost_if_pattern_works(text: str) -> bool:
    """Check if a hook/title contains proven high-performing keywords."""
    keywords = hook_keywords(min_views=300)
    text_lower = text.lower()
    for word, _ in keywords:
        if word in text_lower:
            return True
    return False


def get_proven_hooks(limit: int = 3) -> list[str]:
    """Return proven high-performing hooks to remix into generation."""
    top = top_hooks(limit=limit * 2, min_views=100)
    return [h["hook"] for h in top[:limit]]


def get_proven_titles(limit: int = 3) -> list[str]:
    """Return proven high-performing titles to use as inspiration."""
    top = top_titles(limit=limit * 2, min_views=100)
    return [t["title"] for t in top[:limit]]


def get_engagement_stats() -> dict[str, Any]:
    """Return aggregate engagement statistics."""
    data = _load_db()
    if not data:
        return {"videos": 0, "avg_engagement": 0, "total_views": 0}
    
    total_views = sum(r["views"] for r in data)
    total_likes = sum(r["likes"] for r in data)
    avg_engagement = (total_likes / max(total_views, 1)) * 100 if total_views > 0 else 0
    
    return {
        "videos": len(data),
        "total_views": total_views,
        "total_likes": total_likes,
        "avg_engagement_percent": avg_engagement,
        "avg_ctr": sum(r.get("ctr") or 0 for r in data) / len(data) if data else 0,
    }


def get_recommendations() -> dict[str, Any]:
    """Return actionable recommendations based on performance data."""
    top_hooks_list = top_hooks(limit=3, min_views=100)
    top_titles_list = top_titles(limit=3, min_views=100)
    keywords = hook_keywords(min_views=300, top_n=5)
    search_terms = top_keywords(min_views=200, top_n=5)
    worst = worst_hooks(limit=3)
    
    return {
        "top_hooks": [h["hook"] for h in top_hooks_list],
        "top_titles": [t["title"] for t in top_titles_list],
        "proven_keywords": [w[0] for w in keywords],
        "hot_terms": [w[0] for w in search_terms],
        "avoid_patterns": [h["hook"] for h in worst],
        "stats": get_engagement_stats(),
    }
