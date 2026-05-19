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

    # Auto-migrate legacy pickle token if present.
    if path.suffix == ".pickle" or not path.exists():
        pickle_path = path if path.suffix == ".pickle" else path.with_suffix(".pickle")
        if pickle_path.exists():
            json_path = pickle_path.with_suffix(".json")
            raw = pickle_path.read_bytes()
            try:
                import io
                import pickle as _pickle
                creds = _pickle.load(io.BytesIO(raw))
                token_json = creds.to_json()
            except Exception as exc:
                logger.debug(f"_get_credentials: pickle load failed, trying JSON ({exc})")
                # File was saved as JSON with a .pickle extension.
                token_json = raw.decode()
                json.loads(token_json)  # validate — raises if truly corrupt
            json_path.write_text(token_json)
            pickle_path.unlink()
            path = json_path

    credentials = Credentials.from_authorized_user_info(json.loads(path.read_text()))
    return build("youtubeAnalytics", "v2", credentials=credentials)


def get_video_metrics(video_id: str) -> dict | None:
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