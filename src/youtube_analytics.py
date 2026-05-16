"""YouTube Analytics client for retrieving video performance metrics."""
import json
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build


def get_analytics_service(token_file: str = "config/token.json") -> object:
    """Build and return YouTube Analytics service using stored credentials."""
    path = Path(token_file)

    # Auto-migrate legacy pickle token if present.
    if path.suffix == ".pickle" or not path.exists():
        pickle_path = path if path.suffix == ".pickle" else path.with_suffix(".pickle")
        if pickle_path.exists():
            import pickle as _pickle
            json_path = pickle_path.with_suffix(".json")
            with open(pickle_path, "rb") as f:
                creds = _pickle.load(f)
            json_path.write_text(creds.to_json())
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