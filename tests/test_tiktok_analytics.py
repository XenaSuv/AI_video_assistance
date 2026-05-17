"""Tests for tiktok_analytics — pure functions and HTTP-mocked API calls."""
import json
import time
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tiktok_analytics import (
    _estimate_avg_view_percentage,
    _parse_metrics,
    get_video_metrics,
    resolve_publish_id,
)


# ── _estimate_avg_view_percentage() ──────────────────────────────────────────

class TestEstimateAvgViewPercentage:
    def test_zero_views_returns_zero(self):
        assert _estimate_avg_view_percentage(0, 100, 50, 30) == 0.0

    def test_high_engagement_returns_max_tier(self):
        # 10%+ engagement → 0.70
        result = _estimate_avg_view_percentage(views=1000, likes=200, shares=100, comments=50)
        assert result == 0.70

    def test_medium_high_engagement_in_correct_range(self):
        # ~7.5% engagement → between 0.55 and 0.70
        result = _estimate_avg_view_percentage(views=1000, likes=50, shares=25, comments=0)
        assert 0.55 <= result <= 0.70

    def test_medium_engagement_in_correct_range(self):
        # ~3.5% engagement → between 0.40 and 0.55
        result = _estimate_avg_view_percentage(views=1000, likes=20, shares=5, comments=5)
        assert 0.40 <= result <= 0.55

    def test_low_engagement_returns_minimum_tier(self):
        # <2% engagement → between 0.15 and 0.40
        result = _estimate_avg_view_percentage(views=10000, likes=10, shares=2, comments=3)
        assert 0.15 <= result < 0.40

    def test_shares_weighted_double(self):
        # Same total weight: 10 shares vs 20 likes
        by_shares = _estimate_avg_view_percentage(views=1000, likes=0, shares=10, comments=0)
        by_likes = _estimate_avg_view_percentage(views=1000, likes=20, shares=0, comments=0)
        assert by_shares == by_likes

    def test_result_is_always_non_negative(self):
        assert _estimate_avg_view_percentage(1000, 0, 0, 0) >= 0.0


# ── _parse_metrics() ─────────────────────────────────────────────────────────

class TestParseMetrics:
    def test_extracts_all_fields(self):
        video = {
            "id": "v1",
            "view_count": 5000,
            "like_count": 300,
            "share_count": 50,
            "comment_count": 80,
            "duration": 60,
        }
        result = _parse_metrics(video)
        assert result["views"] == 5000
        assert result["likes"] == 300
        assert result["shares"] == 50
        assert result["comments"] == 80
        assert result["duration_sec"] == 60

    def test_falls_back_to_play_count(self):
        video = {"play_count": 3000, "view_count": 0}
        result = _parse_metrics(video)
        assert result["views"] == 3000

    def test_all_zero_fields_handled(self):
        result = _parse_metrics({})
        assert result["views"] == 0
        assert result["avg_view_percentage"] == 0.0

    def test_avg_view_percentage_is_in_range(self):
        video = {"view_count": 1000, "like_count": 80, "share_count": 20, "comment_count": 10}
        result = _parse_metrics(video)
        assert 0.0 <= result["avg_view_percentage"] <= 1.0


# ── get_video_metrics() ───────────────────────────────────────────────────────

class TestGetVideoMetrics:
    def test_returns_none_when_token_file_missing(self, tmp_path):
        result = get_video_metrics("vid_123", tmp_path / "nonexistent.json")
        assert result is None

    def test_returns_none_when_token_expired(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "tok",
            "expires_at": time.time() - 3600,  # already expired
        }))
        result = get_video_metrics("vid_123", token_file)
        assert result is None

    def test_returns_metrics_when_video_found(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "valid-token",
            "expires_at": time.time() + 7200,
        }))

        api_response = {
            "data": {
                "videos": [
                    {
                        "id": "target_video",
                        "view_count": 2500,
                        "like_count": 200,
                        "share_count": 30,
                        "comment_count": 40,
                        "duration": 45,
                    }
                ],
                "has_more": False,
            }
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        with patch("src.tiktok_analytics.requests.post", return_value=mock_resp):
            result = get_video_metrics("target_video", token_file)

        assert result is not None
        assert result["views"] == 2500
        assert result["likes"] == 200

    def test_returns_none_when_video_not_in_list(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "valid-token",
            "expires_at": time.time() + 7200,
        }))

        api_response = {
            "data": {
                "videos": [{"id": "some_other_video", "view_count": 100}],
                "has_more": False,
            }
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        with patch("src.tiktok_analytics.requests.post", return_value=mock_resp):
            result = get_video_metrics("target_video", token_file)

        assert result is None

    def test_returns_none_on_http_error(self, tmp_path):
        import requests as req_lib

        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "valid-token",
            "expires_at": time.time() + 7200,
        }))

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = req_lib.HTTPError(
            response=MagicMock(status_code=500)
        )

        with patch("src.tiktok_analytics.requests.post", return_value=mock_resp):
            result = get_video_metrics("vid", token_file)

        assert result is None

    def test_paginates_to_find_video(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "valid-token",
            "expires_at": time.time() + 7200,
        }))

        page1 = {"data": {"videos": [{"id": "other1"}], "has_more": True, "cursor": 20}}
        page2 = {
            "data": {
                "videos": [{"id": "target_video", "view_count": 100, "like_count": 5}],
                "has_more": False,
            }
        }

        mock_resp1 = MagicMock()
        mock_resp1.json.return_value = page1
        mock_resp1.raise_for_status = MagicMock()

        mock_resp2 = MagicMock()
        mock_resp2.json.return_value = page2
        mock_resp2.raise_for_status = MagicMock()

        with patch("src.tiktok_analytics.requests.post", side_effect=[mock_resp1, mock_resp2]):
            result = get_video_metrics("target_video", token_file)

        assert result is not None
        assert result["views"] == 100


# ── resolve_publish_id() ──────────────────────────────────────────────────────

class TestResolvePublishId:
    def test_returns_video_id_on_publish_complete(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "valid-token",
            "expires_at": time.time() + 7200,
        }))

        api_response = {
            "data": {
                "status": "PUBLISH_COMPLETE",
                "publicaly_available_post_id": ["7123456789"],
            }
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = api_response
        mock_resp.raise_for_status = MagicMock()

        with patch("src.tiktok_analytics.requests.post", return_value=mock_resp):
            result = resolve_publish_id("pub_abc", token_file)

        assert result == "7123456789"

    def test_returns_none_on_publish_failed(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "valid-token",
            "expires_at": time.time() + 7200,
        }))

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"status": "FAILED"}}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.tiktok_analytics.requests.post", return_value=mock_resp):
            result = resolve_publish_id("pub_abc", token_file)

        assert result is None

    def test_returns_none_when_token_missing(self, tmp_path):
        result = resolve_publish_id("pub_abc", tmp_path / "nofile.json")
        assert result is None

    def test_returns_none_on_exception_during_poll(self, tmp_path):
        import requests as req_lib

        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "valid-token",
            "expires_at": time.time() + 7200,
        }))

        call_count = [0]
        original_time = time.time

        def fake_time():
            call_count[0] += 1
            if call_count[0] <= 2:
                return original_time()
            return original_time() + 120  # past deadline

        with patch("src.tiktok_analytics.http_post", side_effect=Exception("network error")):
            with patch("src.tiktok_analytics.time") as mock_time:
                mock_time.time.side_effect = fake_time
                mock_time.sleep = MagicMock()
                result = resolve_publish_id("pub_xyz", token_file)

        assert result is None

    def test_returns_none_when_timed_out(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "valid-token",
            "expires_at": time.time() + 7200,
        }))

        now = time.time()
        # calls: _load_access_token check, deadline=time.time()+60, while condition
        times = [now, now, now + 61]  # deadline already past on third call

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"status": "PROCESSING"}}
        mock_resp.raise_for_status = MagicMock()

        with patch("src.tiktok_analytics.http_post", return_value=mock_resp):
            with patch("src.tiktok_analytics.time") as mock_time:
                mock_time.time.side_effect = times
                mock_time.sleep = MagicMock()
                result = resolve_publish_id("pub_xyz", token_file)

        assert result is None


class TestGetVideoMetricsErrors:
    def test_returns_none_on_401_error(self, tmp_path):
        import requests as req_lib

        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "valid-token",
            "expires_at": time.time() + 7200,
        }))

        http_error = req_lib.HTTPError(response=MagicMock(status_code=401))
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = http_error

        with patch("src.tiktok_analytics.requests.post", return_value=mock_resp):
            result = get_video_metrics("vid", token_file)

        assert result is None

    def test_returns_none_on_generic_exception(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps({
            "access_token": "valid-token",
            "expires_at": time.time() + 7200,
        }))

        with patch("src.tiktok_analytics.requests.post", side_effect=RuntimeError("unexpected")):
            result = get_video_metrics("vid", token_file)

        assert result is None
