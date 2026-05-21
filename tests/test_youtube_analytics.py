"""Tests for src/youtube_analytics.py — with mocked Google Analytics API."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# ── conftest already stubs googleapiclient / google.oauth2, but we ensure
# sub-module paths are registered so lazy imports inside the module work.
for _mod in (
    "googleapiclient",
    "googleapiclient.discovery",
    "google",
    "google.oauth2",
    "google.oauth2.credentials",
):
    sys.modules.setdefault(_mod, MagicMock())

from src.youtube_analytics import (
    get_analytics_service,
    get_retention_curve,
    get_video_metrics,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _make_service_mock(rows: list | None = None) -> MagicMock:
    """Return a mock Analytics service whose reports().query().execute() returns rows."""
    service = MagicMock()
    response = {"rows": rows or []}
    (service.reports.return_value
               .query.return_value
               .execute.return_value) = response
    return service


# ═══════════════════════════════════════════════════════════════════════════════
# get_analytics_service
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetAnalyticsService:
    def test_builds_service_from_json_token(self, tmp_path):
        token_data = {"token": "ya29.xxx", "refresh_token": "1//xxx", "client_id": "id", "client_secret": "secret"}
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps(token_data))

        mock_creds = MagicMock()
        mock_service = MagicMock()

        with patch.dict("sys.modules", {
            "google.oauth2.credentials": MagicMock(Credentials=MagicMock(
                from_authorized_user_info=MagicMock(return_value=mock_creds)
            )),
        }), patch("src.youtube_analytics.build", return_value=mock_service) as mock_build:
            result = get_analytics_service(str(token_file))

        mock_build.assert_called_once_with("youtubeAnalytics", "v2", credentials=mock_creds)
        assert result is mock_service


# ═══════════════════════════════════════════════════════════════════════════════
# get_video_metrics
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetVideoMetrics:
    def test_returns_none_when_no_rows(self):
        service = _make_service_mock(rows=[])

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            result = get_video_metrics("abc123")

        assert result is None

    def test_returns_dict_with_metrics_when_rows_present(self):
        rows = [["abc123", 1500, 320.5, 0.65]]
        service = _make_service_mock(rows=rows)

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            result = get_video_metrics("abc123")

        assert result is not None
        assert result["views"] == 1500
        assert result["avg_view_duration"] == 320.5
        assert result["avg_view_percentage"] == 0.65

    def test_queries_correct_video_id(self):
        rows = [["xyz789", 500, 120.0, 0.45]]
        service = _make_service_mock(rows=rows)

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            get_video_metrics("xyz789")

        query_kwargs = service.reports.return_value.query.call_args[1]
        assert "xyz789" in query_kwargs.get("filters", "")

    def test_queries_correct_metrics_dimensions(self):
        rows = [["vid1", 100, 60.0, 0.50]]
        service = _make_service_mock(rows=rows)

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            get_video_metrics("vid1")

        query_kwargs = service.reports.return_value.query.call_args[1]
        assert "views" in query_kwargs.get("metrics", "")
        assert "averageViewDuration" in query_kwargs.get("metrics", "")
        assert "averageViewPercentage" in query_kwargs.get("metrics", "")

    def test_queries_channel_mine(self):
        rows = [["vid2", 200, 90.0, 0.55]]
        service = _make_service_mock(rows=rows)

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            get_video_metrics("vid2")

        query_kwargs = service.reports.return_value.query.call_args[1]
        assert query_kwargs.get("ids") == "channel==MINE"

    def test_returns_first_row_data(self):
        # Multiple rows — should use the first one
        rows = [
            ["vid1", 1000, 200.0, 0.60],
            ["vid2", 2000, 300.0, 0.70],
        ]
        service = _make_service_mock(rows=rows)

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            result = get_video_metrics("vid1")

        assert result["views"] == 1000
        assert result["avg_view_percentage"] == 0.60


# ═══════════════════════════════════════════════════════════════════════════════
# get_retention_curve
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetRetentionCurve:
    def test_returns_empty_list_when_no_rows(self):
        service = _make_service_mock(rows=[])

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            result = get_retention_curve("abc123")

        assert result == []

    def test_returns_list_of_ratios_from_rows(self):
        rows = [
            [0.0, 1.0],
            [0.1, 0.95],
            [0.2, 0.88],
            [0.5, 0.72],
            [1.0, 0.45],
        ]
        service = _make_service_mock(rows=rows)

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            result = get_retention_curve("abc123")

        assert result == [1.0, 0.95, 0.88, 0.72, 0.45]

    def test_queries_audience_watch_ratio(self):
        rows = [[0.0, 1.0], [1.0, 0.5]]
        service = _make_service_mock(rows=rows)

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            get_retention_curve("vid_xyz")

        query_kwargs = service.reports.return_value.query.call_args[1]
        assert "audienceWatchRatio" in query_kwargs.get("metrics", "")
        assert "elapsedVideoTimeRatio" in query_kwargs.get("dimensions", "")

    def test_queries_correct_video_id(self):
        rows = [[0.0, 1.0]]
        service = _make_service_mock(rows=rows)

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            get_retention_curve("MY_VIDEO_ID")

        query_kwargs = service.reports.return_value.query.call_args[1]
        assert "MY_VIDEO_ID" in query_kwargs.get("filters", "")

    def test_single_row_returns_single_element(self):
        rows = [[0.5, 0.8]]
        service = _make_service_mock(rows=rows)

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            result = get_retention_curve("vid_single")

        assert result == [0.8]

    def test_extracts_second_column_from_each_row(self):
        """Only the second element (index 1) should be in the curve."""
        rows = [
            ["ignored_time_ratio_0", 0.99],
            ["ignored_time_ratio_1", 0.85],
        ]
        service = _make_service_mock(rows=rows)

        with patch("src.youtube_analytics.get_analytics_service", return_value=service):
            result = get_retention_curve("vid_col")

        assert result == [0.99, 0.85]
        assert "ignored_time_ratio_0" not in result
