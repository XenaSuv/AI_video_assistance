"""Fetch stock video B-roll clips from Pexels (primary) or Pixabay (fallback).

The hybrid visual model targets 70 % DALL-E / screenshot / infographic
and 30 % video B-roll for scenes with motion-friendly content.

GPT sets scene.video_query to a short stock-search string (e.g.
"person coding laptop") for those scenes. This module searches the API,
downloads the best match, trims/loops it to the exact scene duration,
and resizes to 1280×720.

Falls back silently to DALL-E when:
  - No API key is configured
  - No results match the query
  - Download / processing fails
  - scene.video_query is null

Public API
----------
fetch_broll_clip(scene, out_path, *, target_w, target_h) -> Path | None
    Returns a finished MP4 path, or None to signal "fall back to DALL-E".

Attribution helpers
-------------------
get_pexels_credits(run_dir) -> list[str]
    Read credit strings saved during the run (for description footer).
save_pexels_credit(run_dir, photographer, url)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.retry_utils import http_get
from src.script_generator import Scene
import src.ffmpeg_utils as ffmpeg_utils

_PEXELS_VIDEO_URL = "https://api.pexels.com/videos/search"
_PIXABAY_VIDEO_URL = "https://pixabay.com/api/videos/"

VIDEO_W, VIDEO_H = 1280, 720


# ─────────────────── Pexels ───────────────────

def _pexels_search(query: str, per_page: int = 5) -> list[dict]:
    key = settings.pexels_api_key
    if not key:
        return []
    r = http_get(
        _PEXELS_VIDEO_URL,
        headers={"Authorization": key},
        params={
            "query":       query,
            "orientation": "landscape",
            "size":        "medium",
            "per_page":    per_page,
        },
        timeout=15,
    )
    return r.json().get("videos", [])


def _pexels_best_file(video: dict) -> dict | None:
    files = video.get("video_files", [])
    hd = [f for f in files if f.get("quality") == "hd"]
    candidates = hd or files
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.get("width", 0))


# ─────────────────── Pixabay (fallback) ───────────────────

def _pixabay_search(query: str, per_page: int = 5) -> list[dict]:
    key = settings.pixabay_api_key
    if not key:
        return []
    r = http_get(
        _PIXABAY_VIDEO_URL,
        params={
            "key":         key,
            "q":           query,
            "orientation": "horizontal",
            "per_page":    per_page,
        },
        timeout=15,
    )
    return r.json().get("hits", [])


def _pixabay_best_url(video: dict) -> str | None:
    vids = video.get("videos", {})
    # Prefer large (1920) > medium (1280) > small
    for key in ("large", "medium", "small"):
        url = vids.get(key, {}).get("url")
        if url:
            return url
    return None


# ─────────────────── download + process ───────────────────

def _download(url: str, out_path: Path) -> Path:
    r = http_get(url, timeout=120, stream=True)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            if chunk:
                f.write(chunk)
    return out_path


def _process_clip(raw_path: Path, duration_sec: float, out_path: Path) -> Path:
    """Resize → loop/trim → write."""
    raw_w, raw_h = ffmpeg_utils.video_size(raw_path)
    if (raw_w, raw_h) != (VIDEO_W, VIDEO_H):
        resized_path = raw_path.with_stem(raw_path.stem + "_resized")
        ffmpeg_utils.resize_video(raw_path, resized_path, width=VIDEO_W, height=VIDEO_H)
    else:
        resized_path = raw_path

    ffmpeg_utils.loop_and_trim(resized_path, out_path, target_sec=duration_sec)

    # Clean up intermediates
    raw_path.unlink(missing_ok=True)
    if resized_path != raw_path:
        resized_path.unlink(missing_ok=True)
    return out_path


# ─────────────────── attribution ───────────────────

def _credits_file(run_dir: Path) -> Path:
    return run_dir / "broll_credits.json"


def save_pexels_credit(run_dir: Path, photographer: str, video_url: str) -> None:
    path = _credits_file(run_dir)
    credits = json.loads(path.read_text()) if path.exists() else []
    entry = f"Video by {photographer} on Pexels ({video_url})"
    if entry not in credits:
        credits.append(entry)
        path.write_text(json.dumps(credits, indent=2))


def get_pexels_credits(run_dir: Path) -> list[str]:
    path = _credits_file(run_dir)
    return json.loads(path.read_text()) if path.exists() else []


# ─────────────────── public API ───────────────────

def fetch_broll_clip(
    scene: Scene,
    out_path: Path,
    *,
    run_dir: Path | None = None,
) -> Path | None:
    """Search Pexels (then Pixabay) for a B-roll clip matching scene.video_query.

    Returns the finished MP4 path on success, or None so the caller falls
    back to DALL-E.  All failures are logged as warnings, never raised.
    """
    query = scene.video_query
    if not query:
        return None

    raw_path = out_path.with_stem(out_path.stem + "_raw")

    # ── Pexels ───────────────────────────────────────────────────────────────
    pexels_key = settings.pexels_api_key
    if pexels_key:
        try:
            videos = _pexels_search(query)
            for vid in videos:
                file_info = _pexels_best_file(vid)
                if not file_info:
                    continue
                logger.info(
                    f"Scene {scene.idx}: Pexels B-roll for {query!r} "
                    f"({vid.get('width')}×{vid.get('height')}, "
                    f"{vid.get('duration')}s)"
                )
                _download(file_info["link"], raw_path)
                result = _process_clip(raw_path, float(scene.duration_sec), out_path)
                if run_dir:
                    save_pexels_credit(
                        run_dir,
                        vid.get("user", {}).get("name", "Unknown"),
                        vid.get("url", ""),
                    )
                return result
            logger.warning(f"Scene {scene.idx}: no Pexels results for {query!r}")
        except Exception as e:
            logger.warning(f"Scene {scene.idx}: Pexels failed ({e}) — trying Pixabay")

    # ── Pixabay ──────────────────────────────────────────────────────────────
    pixabay_key = settings.pixabay_api_key
    if pixabay_key:
        try:
            hits = _pixabay_search(query)
            for hit in hits:
                url = _pixabay_best_url(hit)
                if not url:
                    continue
                logger.info(f"Scene {scene.idx}: Pixabay B-roll for {query!r}")
                _download(url, raw_path)
                return _process_clip(raw_path, float(scene.duration_sec), out_path)
            logger.warning(f"Scene {scene.idx}: no Pixabay results for {query!r}")
        except Exception as e:
            logger.warning(f"Scene {scene.idx}: Pixabay failed ({e})")

    if not pexels_key and not pixabay_key:
        logger.warning(
            f"Scene {scene.idx}: video_query set but no PPIXABAY_API_KEY configured — falling back to DALL-E"
        )

    return None   # signal: use DALL-E


def fetch_broll_for_short(
    query: str,
    duration_sec: float,
    out_path: Path,
    *,
    out_w: int = 1080,
    out_h: int = 1920,
) -> Path | None:
    """Fetch a B-roll clip from Pexels (then Pixabay), convert to vertical 9:16.

    Unlike fetch_broll_clip() this function takes a plain query string rather
    than a Scene object, so Shorts can call it without creating a fake scene.
    Returns the finished MP4 path at out_w×out_h or None on failure.
    """
    raw_path      = out_path.with_stem(out_path.stem + "_raw")
    vertical_path = out_path.with_stem(out_path.stem + "_vertical")

    pexels_key = settings.pexels_api_key
    if pexels_key:
        try:
            videos = _pexels_search(query, per_page=5)
            for vid in videos:
                file_info = _pexels_best_file(vid)
                if not file_info:
                    continue
                _download(file_info["link"], raw_path)
                ffmpeg_utils.make_vertical(raw_path, vertical_path, out_w=out_w, out_h=out_h)
                ffmpeg_utils.loop_and_trim(vertical_path, out_path, target_sec=duration_sec)
                raw_path.unlink(missing_ok=True)
                vertical_path.unlink(missing_ok=True)
                logger.info(f"Shorts B-roll: Pexels clip for {query!r} → {out_path.name}")
                return out_path
            logger.debug(f"Shorts B-roll: no Pexels results for {query!r}")
        except Exception as exc:
            logger.warning(f"Shorts B-roll: Pexels failed ({exc})")
            for p in (raw_path, vertical_path):
                p.unlink(missing_ok=True)

    pixabay_key = settings.pixabay_api_key
    if pixabay_key:
        try:
            hits = _pixabay_search(query, per_page=5)
            for hit in hits:
                url = _pixabay_best_url(hit)
                if not url:
                    continue
                _download(url, raw_path)
                ffmpeg_utils.make_vertical(raw_path, vertical_path, out_w=out_w, out_h=out_h)
                ffmpeg_utils.loop_and_trim(vertical_path, out_path, target_sec=duration_sec)
                raw_path.unlink(missing_ok=True)
                vertical_path.unlink(missing_ok=True)
                logger.info(f"Shorts B-roll: Pixabay clip for {query!r} → {out_path.name}")
                return out_path
            logger.debug(f"Shorts B-roll: no Pixabay results for {query!r}")
        except Exception as exc:
            logger.warning(f"Shorts B-roll: Pixabay failed ({exc})")
            for p in (raw_path, vertical_path):
                p.unlink(missing_ok=True)

    return None

