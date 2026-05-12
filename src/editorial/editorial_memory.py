"""Editorial Memory — persistent record of editorial decisions and their outcomes.

Each published video writes one entry. On the next run, judgment queries the
memory for the same topic, injects a callback line when a prior prediction
exists, and adjusts angle weights based on observed retention.

Structure
─────────
{
    "topic":              str,
    "angle":              str,           # hype_vs_reality | hidden_risks | misuse | impact
    "persona":            str,
    "format":             str,
    "prediction":         str,
    "prediction_status":  "pending" | "confirmed" | "wrong",
    "retention":          float,
    "comments_sentiment": "positive" | "negative" | "neutral" | "",
    "video_id":           str,
    "date":               str,           # ISO-8601 date
}

Integration points
──────────────────
    EditorialJudgment.judge()      — context lookup + callback injection
    pipeline_orchestrator          — add_entry() after successful upload
    dashboard                      — render_editorial_memory()
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger


_MEMORY_FILE = "editorial_memory.json"
_MAX_ENTRIES = 500        # hard cap to keep the file bounded
_CONTEXT_WINDOW = 5       # entries returned by get_relevant_context
_RECENT_WINDOW  = 10      # entries considered "recent" for repetition check


class EditorialMemory:
    """Persistent editorial memory across pipeline runs.

    Public interface
    ────────────────
    add_entry(entry)                → append + prune + save
    get_relevant_context(topic)     → list of up to _CONTEXT_WINDOW recent entries on topic
    has_prediction(topic)           → bool
    get_callback_line(topic)        → str | None
    recent_topics(n)                → list[str]
    topic_is_recent(topic)          → bool
    confirm_prediction(topic, *, wrong=False) → None
    top_angles(n)                   → list[tuple[str, int]]
    recent_predictions(n)           → list[dict]
    save()                          → None
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        if data_dir is None:
            from config import settings
            data_dir = settings.data_dir
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / _MEMORY_FILE
        self._entries: list[dict[str, Any]] = self._load()

    # ── public interface ──────────────────────────────────────────────────────

    def add_entry(self, entry: dict[str, Any]) -> None:
        """Append an editorial decision record and persist.

        Required keys: topic, angle.
        Optional: persona, format, prediction, retention,
                  comments_sentiment, video_id.
        """
        record: dict[str, Any] = {
            "topic":             str(entry.get("topic", "")),
            "angle":             str(entry.get("angle", "")),
            "persona":           str(entry.get("persona", "")),
            "format":            str(entry.get("format", "")),
            "prediction":        str(entry.get("prediction", "")),
            "prediction_status": str(entry.get("prediction_status", "pending")),
            "retention":         float(entry.get("retention", 0.0)),
            "comments_sentiment": str(entry.get("comments_sentiment", "")),
            "video_id":          str(entry.get("video_id", "")),
            "date":              datetime.now(timezone.utc).date().isoformat(),
        }
        self._entries.append(record)
        if len(self._entries) > _MAX_ENTRIES:
            self._entries = self._entries[-_MAX_ENTRIES:]
        self.save()
        logger.debug(f"EditorialMemory: recorded entry topic={record['topic']!r} angle={record['angle']!r}")

    def get_relevant_context(self, topic: str) -> list[dict[str, Any]]:
        """Return up to _CONTEXT_WINDOW recent entries that share topic keywords.

        Matching is bidirectional: either the query is in the stored topic or
        vice-versa, so short stored topics still match longer query strings.
        """
        topic_lower = topic.lower()
        matching = [
            e for e in self._entries
            if (topic_lower in e.get("topic", "").lower()
                or e.get("topic", "").lower() in topic_lower)
        ]
        return matching[-_CONTEXT_WINDOW:]

    def has_prediction(self, topic: str) -> bool:
        """True if any prior entry for this topic contains a non-empty prediction."""
        return any(
            e.get("prediction", "")
            for e in self.get_relevant_context(topic)
        )

    def get_callback_line(self, topic: str) -> str | None:
        """Return a ready-to-inject callback line referencing the most recent prediction.

        Returns None if no prior prediction exists.
        """
        ctx = self.get_relevant_context(topic)
        for entry in reversed(ctx):
            pred = entry.get("prediction", "")
            date = entry.get("date", "")
            if not pred:
                continue
            try:
                from datetime import date as _date
                days_ago = (datetime.now(timezone.utc).date() - _date.fromisoformat(date)).days
                if days_ago <= 7:
                    when = "last week"
                elif days_ago <= 30:
                    when = "last month"
                else:
                    when = f"{days_ago // 30} months ago"
            except Exception:
                when = "previously"
            return f"This is exactly what I said {when}: {pred}"
        return None

    def recent_topics(self, n: int = _RECENT_WINDOW) -> list[str]:
        """Return topic strings for the n most recent entries."""
        return [e.get("topic", "") for e in self._entries[-n:]]

    def topic_is_recent(self, topic: str, window: int = _RECENT_WINDOW) -> bool:
        """True if this topic appears in the last *window* entries (bidirectional match)."""
        topic_lower = topic.lower()
        return any(
            topic_lower in t.lower() or t.lower() in topic_lower
            for t in self.recent_topics(window)
        )

    def confirm_prediction(self, topic: str, *, wrong: bool = False) -> None:
        """Update prediction_status for the most recent matching entry.

        Marks "confirmed" or "wrong" and saves.
        """
        status = "wrong" if wrong else "confirmed"
        for entry in reversed(self._entries):
            if topic.lower() in entry.get("topic", "").lower():
                if entry.get("prediction"):
                    entry["prediction_status"] = status
                    self.save()
                    logger.info(f"EditorialMemory: prediction {status!r} for topic={topic!r}")
                    return

    def top_angles(self, n: int = 5) -> list[tuple[str, int]]:
        """Return the n most-used angles as (angle, count) pairs, sorted descending."""
        counts: dict[str, int] = {}
        for e in self._entries:
            a = e.get("angle", "")
            if a:
                counts[a] = counts.get(a, 0) + 1
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def avg_retention_by_angle(self) -> dict[str, float]:
        """Return mean retention per angle across all entries with retention > 0."""
        sums: dict[str, float] = {}
        cnts: dict[str, int]   = {}
        for e in self._entries:
            a = e.get("angle", "")
            r = float(e.get("retention", 0.0))
            if a and r > 0:
                sums[a] = sums.get(a, 0.0) + r
                cnts[a] = cnts.get(a, 0) + 1
        return {a: sums[a] / cnts[a] for a in sums}

    def recent_predictions(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the n most recent entries that have a non-empty prediction."""
        preds = [e for e in self._entries if e.get("prediction")]
        return preds[-n:]

    def save(self) -> None:
        try:
            self._path.write_text(json.dumps(self._entries, indent=2))
        except Exception as exc:
            logger.warning(f"EditorialMemory: save failed: {exc}")

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        try:
            data = json.loads(self._path.read_text())
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning(f"EditorialMemory: load failed, starting fresh: {exc}")
        return []


# ── Module-level singleton ────────────────────────────────────────────────────

_memory: EditorialMemory | None = None


def get_editorial_memory() -> EditorialMemory:
    """Return the module-level EditorialMemory singleton."""
    global _memory
    if _memory is None:
        _memory = EditorialMemory()
    return _memory
