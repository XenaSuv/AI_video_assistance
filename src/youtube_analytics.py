"""YouTube Analytics client for retrieving video performance metrics."""
import json
from pathlib import Path
from typing import Any

from googleapiclient.discovery import build
from loguru import logger


def get_analytics_service(token_file: str = "config/token.json") -> Any:
    """Build and return YouTube Analytics service using stored credentials."""
    from google.oauth2.credentials import Credentials  # lazy: avoids import at module level

    path = Path(token_file)
    credentials = Credentials.from_authorized_user_info(json.loads(path.read_text()))  # type: ignore[no-untyped-call]
    return build("youtubeAnalytics", "v2", credentials=credentials)


def get_video_metrics(video_id: str) -> dict[str, Any] | None:
    """Get basic video metrics from YouTube Analytics.

    Args:
        video_id: YouTube video ID

    Returns:
        Dict with views, avg_view_duration, avg_view_percentage or None if no data
    """
    service = get_analytics_service()

    response = service.reports().query(
        ids="channel==MINE",
        startDate="2024-01-01",
        endDate="2026-12-31",
        metrics="views,averageViewDuration,averageViewPercentage",
        dimensions="video",
        filters=f"video=={video_id}"
    ).execute()

    rows = response.get("rows", [])

    if not rows:
        return None

    return {
        "views": rows[0][1],
        "avg_view_duration": rows[0][2],
        "avg_view_percentage": rows[0][3]
    }


def get_retention_curve(video_id: str) -> list[float]:
    """Get audience retention curve for a video.

    Args:
        video_id: YouTube video ID

    Returns:
        List of retention ratios over time (audienceWatchRatio by elapsedVideoTimeRatio)
    """
    service = get_analytics_service()

    response = service.reports().query(
        ids="channel==MINE",
        startDate="2024-01-01",
        endDate="2026-12-31",
        metrics="audienceWatchRatio",
        dimensions="elapsedVideoTimeRatio",
        filters=f"video=={video_id}"
    ).execute()

    rows = response.get("rows", [])

    curve = [row[1] for row in rows]

    return curve