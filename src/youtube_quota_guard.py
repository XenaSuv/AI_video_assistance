"""YouTube Data API v3 daily quota tracker.

YouTube grants 10,000 units per day by default.  Each API operation has a
fixed cost (see QUOTA_COSTS).  This module tracks usage in
data/youtube_quota.json so the pipeline can bail out before hitting the
hard limit instead of discovering it mid-upload.

Typical usage (in youtube_uploader.py)::

    guard = YouTubeQuotaGuard(settings.data_dir)
    if not guard.can_afford("videos.insert"):
        raise QuotaExhaustedError(guard.used_today(), guard.daily_limit)
    ...  # do the upload ...
    guard.charge("videos.insert")
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any


# Unit costs from the official YouTube API quota calculator.
QUOTA_COSTS: dict[str, int] = {
    "videos.insert":   1600,
    "thumbnails.set":    50,
    "captions.insert":  400,
    "videos.list":        1,
}

DEFAULT_DAILY_QUOTA: int = 10_000

# Warn via Slack when remaining quota drops below this fraction.
WARN_THRESHOLD: float = 0.20   # 20 %


class YouTubeQuotaGuard:
    """Persist and check YouTube API quota usage for the current calendar day."""

    def __init__(
        self,
        data_dir: Path,
        daily_limit: int = DEFAULT_DAILY_QUOTA,
    ) -> None:
        self._path       = data_dir / "youtube_quota.json"
        self.daily_limit = daily_limit
        self._data       = self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return {}

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(self._data, indent=2))
            tmp.replace(self._path)
        except Exception:
            pass

    # ── public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _today() -> str:
        return dt.date.today().isoformat()

    def used_today(self) -> int:
        """Return units consumed so far today."""
        return int(self._data.get(self._today(), {}).get("used", 0))

    def remaining(self) -> int:
        """Return units still available today (never negative)."""
        return max(0, self.daily_limit - self.used_today())

    def can_afford(self, operation: str) -> bool:
        """Return True if *operation* can run without exceeding the daily quota."""
        cost = QUOTA_COSTS.get(operation, 0)
        return self.remaining() >= cost

    def charge(self, operation: str) -> int:
        """Record that *operation* was executed.  Returns the units deducted."""
        cost = QUOTA_COSTS.get(operation, 0)
        if cost == 0:
            return 0
        today    = self._today()
        day_data = self._data.setdefault(today, {"used": 0, "ops": []})
        day_data["used"] = int(day_data.get("used", 0)) + cost
        day_data["ops"].append(operation)
        self._save()
        return cost

    def mark_exhausted(self) -> None:
        """Force today's usage to the daily limit (called on 403 quotaExceeded)."""
        today    = self._today()
        day_data = self._data.setdefault(today, {"used": 0, "ops": []})
        day_data["used"] = self.daily_limit
        self._save()

    def near_limit(self) -> bool:
        """Return True when remaining quota is below the warning threshold."""
        return self.remaining() < self.daily_limit * WARN_THRESHOLD

    # ── quota-error detection ─────────────────────────────────────────────────

    @staticmethod
    def is_quota_error(exc: Exception) -> bool:
        """Return True when *exc* is a YouTube 403 quotaExceeded response.

        Inspects ``exc.resp.status`` and the JSON body for the reason field
        so that regular 403 auth errors are not mis-classified as quota errors.
        """
        try:
            resp = getattr(exc, "resp", None)
            if resp is None or getattr(resp, "status", None) != 403:
                return False
            raw  = getattr(exc, "content", b"")
            body = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8", "replace"))
            reasons = {
                e.get("reason", "")
                for e in body.get("error", {}).get("errors", [])
            }
            return bool(reasons & {"quotaExceeded", "userRateLimitExceeded"})
        except Exception:
            return False
