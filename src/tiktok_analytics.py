"""TikTok Analytics via the Content API v2 video/list endpoint.

Scope required: video.list
  If the stored token was issued without this scope, re-run the auth flow:
      python src/tiktok_uploader.py --auth

What TikTok's public API exposes:
  - view_count, like_count, share_count, comment_count, duration — real data
  - avg watch time / retention curve — NOT available; estimated from engagement

The publish_id returned by post_short() is an upload reference, not the video
ID used for analytics.  After publishing, call resolve_publish_id() to obtain
the real video_id, then pass that to get_video_metrics().
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests
from loguru import logger

from src.retry_utils import http_post

_VIDEO_LIST_URL   = "https://open.tiktokapis.com/v2/video/list/"
_STATUS_URL       = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
_VIDEO_FIELDS     = "id,duration,view_count,like_count,comment_count,share_count,play_count"
_PAGE_SIZE        = 20
_MAX_PAGES        = 10   # cap to avoid scanning huge accounts


def _load_access_token(token_file: Path) -> str | None:
    """Return stored access token if still valid (>5 min remaining), else None."""
    try:
        tokens = json.loads(token_file.read_text())
    except Exception as exc:
        logger.warning(f"TikTok: cannot read token file {token_file}: {exc}")
        return None
    if time.time() > tokens.get("expires_at", 0) - 300:
        logger.warning("TikTok: access token is expired — re-run auth or let uploader refresh it")
        return None
    return tokens.get("access_token")


def resolve_publish_id(publish_id: str, token_file: Path) -> str | None:
    """Resolve a publish_id (returned by post_short) to the actual video_id.

    TikTok typically processes the video within seconds; this polls until the
    status is PUBLISH_COMPLETE or returns None after 60 s.
    """
    access_token = _load_access_token(token_file)
    if not access_token:
        return None

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            resp = http_post(
                _STATUS_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                },
                json={"publish_id": publish_id},
                timeout=15,
            )
            data = resp.json().get("data", {})
            status = data.get("status", "")
            if status == "PUBLISH_COMPLETE":
                video_id = data.get("publicaly_available_post_id", [None])[0]
                if video_id:
                    logger.info(f"TikTok: publish_id {publish_id} → video_id {video_id}")
                    return str(video_id)
            elif status in ("FAILED", "PUBLISH_FAILED"):
                logger.error(f"TikTok: publish failed for {publish_id}: {data}")
                return None
        except Exception as exc:
            logger.warning(f"TikTok status check error: {exc}")
        time.sleep(5)

    logger.warning(f"TikTok: timed out resolving publish_id {publish_id}")
    return None


def get_video_metrics(video_id: str, token_file: Path) -> dict | None:
    """Fetch real metrics for a published TikTok video.

    Returns a dict with keys:
        views, likes, shares, comments, duration_sec, avg_view_percentage
    or None when the video is not found or the API is unavailable.

    avg_view_percentage is estimated from the engagement rate because TikTok
    does not expose per-video watch time via the public API.
    """
    access_token = _load_access_token(token_file)
    if not access_token:
        return None

    cursor = 0
    for page in range(_MAX_PAGES):
        try:
            resp = http_post(
                _VIDEO_LIST_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; charset=UTF-8",
                },
                json={
                    "max_count": _PAGE_SIZE,
                    "cursor": cursor,
                    "fields": _VIDEO_FIELDS,
                },
                timeout=30,
            )
            payload = resp.json()
            data = payload.get("data", {})

            for video in data.get("videos", []):
                if str(video.get("id")) == str(video_id):
                    metrics = _parse_metrics(video)
                    logger.info(
                        f"TikTok metrics for {video_id}: "
                        f"views={metrics['views']}, "
                        f"likes={metrics['likes']}, "
                        f"shares={metrics['shares']}, "
                        f"est. avg_view%={metrics['avg_view_percentage']:.0%}"
                    )
                    return metrics

            if not data.get("has_more"):
                break
            cursor = data.get("cursor", cursor + _PAGE_SIZE)

        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                logger.error("TikTok: 401 Unauthorized — token may lack video.list scope")
            else:
                logger.error(f"TikTok API HTTP error (page {page}): {exc}")
            return None
        except Exception as exc:
            logger.error(f"TikTok API error (page {page}): {exc}")
            return None

    logger.warning(f"TikTok: video {video_id} not found in video list after {_MAX_PAGES} pages")
    return None


def _parse_metrics(video: dict) -> dict:
    views    = int(video.get("view_count", 0) or video.get("play_count", 0) or 0)
    likes    = int(video.get("like_count", 0) or 0)
    shares   = int(video.get("share_count", 0) or 0)
    comments = int(video.get("comment_count", 0) or 0)
    duration = int(video.get("duration", 0) or 0)
    return {
        "views":               views,
        "likes":               likes,
        "shares":              shares,
        "comments":            comments,
        "duration_sec":        duration,
        "avg_view_percentage": _estimate_avg_view_percentage(views, likes, shares, comments),
    }


def _estimate_avg_view_percentage(
    views: int, likes: int, shares: int, comments: int
) -> float:
    """Estimate avg view % from engagement (TikTok doesn't expose watch time).

    Weights: shares count double (high-intent), comments 1.5×, likes 1×.
    Engagement tiers → estimated retention:
      ≥10 % → 0.70  |  5–10 % → 0.55–0.70  |  2–5 % → 0.40–0.55  |  <2 % → 0.15–0.40
    """
    if views <= 0:
        return 0.0
    eng = (likes + shares * 2 + comments * 1.5) / views
    if eng >= 0.10:
        return 0.70
    if eng >= 0.05:
        return 0.55 + (eng - 0.05) / 0.05 * 0.15
    if eng >= 0.02:
        return 0.40 + (eng - 0.02) / 0.03 * 0.15
    return max(0.15, 0.25 * (eng / 0.02))
