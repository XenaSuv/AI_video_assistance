"""Tests for src/youtube_quota_guard.py."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.youtube_quota_guard import (
    YouTubeQuotaGuard, QUOTA_COSTS, DEFAULT_DAILY_QUOTA, WARN_THRESHOLD,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _guard(tmp_path: Path, daily_limit: int = DEFAULT_DAILY_QUOTA) -> YouTubeQuotaGuard:
    return YouTubeQuotaGuard(tmp_path, daily_limit=daily_limit)


def _make_quota_exc(reason: str = "quotaExceeded") -> MagicMock:
    """Fake exception shaped like googleapiclient.errors.HttpError with quota reason."""
    exc = MagicMock()
    exc.resp = MagicMock()
    exc.resp.status = 403
    body = json.dumps({"error": {"code": 403, "errors": [{"reason": reason}]}})
    exc.content = body.encode()
    return exc


# ── used_today / remaining ────────────────────────────────────────────────────

class TestUsedTodayAndRemaining:
    def test_zero_when_no_data(self, tmp_path):
        g = _guard(tmp_path)
        assert g.used_today() == 0

    def test_remaining_equals_limit_when_fresh(self, tmp_path):
        g = _guard(tmp_path, daily_limit=1000)
        assert g.remaining() == 1000

    def test_remaining_decreases_after_charge(self, tmp_path):
        g = _guard(tmp_path, daily_limit=1000)
        g.charge("videos.insert")   # 1600 — over limit, remaining clamps to 0
        assert g.remaining() == 0

    def test_remaining_never_negative(self, tmp_path):
        g = _guard(tmp_path, daily_limit=100)
        g.charge("videos.insert")   # costs 1600 >> 100
        assert g.remaining() == 0

    def test_used_today_accumulates(self, tmp_path):
        g = _guard(tmp_path, daily_limit=10_000)
        g.charge("thumbnails.set")  # 50
        g.charge("thumbnails.set")  # 50
        assert g.used_today() == 100


# ── can_afford ─────────────────────────────────────────────────────────────────

class TestCanAfford:
    def test_true_when_plenty_of_quota(self, tmp_path):
        g = _guard(tmp_path, daily_limit=10_000)
        assert g.can_afford("videos.insert")   # costs 1600

    def test_false_when_quota_exhausted(self, tmp_path):
        g = _guard(tmp_path, daily_limit=100)
        assert not g.can_afford("videos.insert")

    def test_true_for_cheap_op_after_expensive_one(self, tmp_path):
        g = _guard(tmp_path, daily_limit=10_000)
        g.charge("videos.insert")   # costs 1600
        # thumbnails.set costs 50 — still affordable
        assert g.can_afford("thumbnails.set")

    def test_false_when_not_enough_for_captions(self, tmp_path):
        g = _guard(tmp_path, daily_limit=500)
        g.charge("videos.insert")   # 1600 >> 500 — remaining = 0
        assert not g.can_afford("captions.insert")

    def test_zero_cost_op_always_affordable(self, tmp_path):
        # Unknown ops have cost 0; 0 >= 0 is always True even with no quota left.
        g = _guard(tmp_path, daily_limit=0)
        assert g.can_afford("nonexistent_op")  # cost=0, 0>=0 → True


# ── charge ─────────────────────────────────────────────────────────────────────

class TestCharge:
    def test_returns_cost(self, tmp_path):
        g = _guard(tmp_path)
        assert g.charge("videos.insert") == 1600

    def test_persists_to_disk(self, tmp_path):
        g = _guard(tmp_path)
        g.charge("thumbnails.set")
        # load fresh instance to verify persistence
        g2 = _guard(tmp_path)
        assert g2.used_today() == 50

    def test_records_op_name(self, tmp_path):
        g = _guard(tmp_path)
        g.charge("captions.insert")
        data = json.loads((tmp_path / "youtube_quota.json").read_text())
        today_ops = list(data.values())[0]["ops"]
        assert "captions.insert" in today_ops

    def test_multiple_charges_accumulate(self, tmp_path):
        g = _guard(tmp_path)
        g.charge("thumbnails.set")   # 50
        g.charge("captions.insert")  # 400
        assert g.used_today() == 450

    def test_zero_cost_op_no_write(self, tmp_path):
        g = _guard(tmp_path)
        cost = g.charge("nonexistent_op")
        assert cost == 0
        assert not (tmp_path / "youtube_quota.json").exists()


# ── mark_exhausted ────────────────────────────────────────────────────────────

class TestMarkExhausted:
    def test_sets_used_to_daily_limit(self, tmp_path):
        g = _guard(tmp_path, daily_limit=5000)
        g.mark_exhausted()
        assert g.used_today() == 5000

    def test_remaining_becomes_zero(self, tmp_path):
        g = _guard(tmp_path)
        g.mark_exhausted()
        assert g.remaining() == 0

    def test_can_afford_returns_false_after_exhausted(self, tmp_path):
        g = _guard(tmp_path)
        g.mark_exhausted()
        assert not g.can_afford("thumbnails.set")

    def test_persists_to_disk(self, tmp_path):
        g = _guard(tmp_path, daily_limit=3000)
        g.mark_exhausted()
        g2 = _guard(tmp_path, daily_limit=3000)
        assert g2.remaining() == 0


# ── near_limit ────────────────────────────────────────────────────────────────

class TestNearLimit:
    def test_false_when_plenty_remaining(self, tmp_path):
        g = _guard(tmp_path, daily_limit=10_000)
        assert not g.near_limit()

    def test_true_when_below_warn_threshold(self, tmp_path):
        # warn threshold = 20%, so < 2000 units remaining on 10k limit
        g = _guard(tmp_path, daily_limit=10_000)
        # charge 8500 units manually
        today = g._today()
        g._data[today] = {"used": 8500, "ops": []}
        g._save()
        g2 = _guard(tmp_path, daily_limit=10_000)
        assert g2.near_limit()

    def test_false_exactly_at_threshold(self, tmp_path):
        # exactly 20% remaining → not near_limit (boundary: < threshold triggers)
        g = _guard(tmp_path, daily_limit=10_000)
        today = g._today()
        g._data[today] = {"used": 8000, "ops": []}  # 20% remaining
        g._save()
        g2 = _guard(tmp_path, daily_limit=10_000)
        assert not g2.near_limit()


# ── is_quota_error ────────────────────────────────────────────────────────────

class TestIsQuotaError:
    def test_true_for_quota_exceeded(self):
        exc = _make_quota_exc("quotaExceeded")
        assert YouTubeQuotaGuard.is_quota_error(exc)

    def test_true_for_user_rate_limit_exceeded(self):
        exc = _make_quota_exc("userRateLimitExceeded")
        assert YouTubeQuotaGuard.is_quota_error(exc)

    def test_false_for_non_403(self):
        exc = _make_quota_exc("quotaExceeded")
        exc.resp.status = 500
        assert not YouTubeQuotaGuard.is_quota_error(exc)

    def test_false_for_regular_403_forbidden(self):
        exc = MagicMock()
        exc.resp = MagicMock()
        exc.resp.status = 403
        exc.content = json.dumps({
            "error": {"code": 403, "errors": [{"reason": "forbidden"}]}
        }).encode()
        assert not YouTubeQuotaGuard.is_quota_error(exc)

    def test_false_for_non_http_exception(self):
        assert not YouTubeQuotaGuard.is_quota_error(RuntimeError("boom"))

    def test_false_when_no_resp_attribute(self):
        exc = Exception("plain error")
        assert not YouTubeQuotaGuard.is_quota_error(exc)

    def test_false_for_malformed_body(self):
        exc = MagicMock()
        exc.resp = MagicMock()
        exc.resp.status = 403
        exc.content = b"not valid json"
        assert not YouTubeQuotaGuard.is_quota_error(exc)

    def test_false_for_empty_errors_list(self):
        exc = MagicMock()
        exc.resp = MagicMock()
        exc.resp.status = 403
        exc.content = json.dumps({"error": {"code": 403, "errors": []}}).encode()
        assert not YouTubeQuotaGuard.is_quota_error(exc)


# ── multi-day isolation ────────────────────────────────────────────────────────

class TestMultiDayIsolation:
    def test_yesterday_usage_does_not_count_today(self, tmp_path):
        g = _guard(tmp_path)
        # Manually inject yesterday's data
        g._data["2026-05-15"] = {"used": 9999, "ops": ["videos.insert"] * 6}
        g._save()
        g2 = _guard(tmp_path)
        assert g2.used_today() == 0
        assert g2.remaining() == DEFAULT_DAILY_QUOTA
