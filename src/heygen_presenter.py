"""HeyGen Video API — AI avatar presenter clips for hook and scene generation.

Supports two generation modes:
  audio   — lip-sync the avatar to pre-recorded ElevenLabs TTS audio
  text    — HeyGen internal TTS (used as fallback when audio upload fails)

Workflow:
  1. Upload local MP3 to HeyGen asset store → get public audio_url
  2. POST /v2/video/generate with avatar + audio_url → get video_id
  3. Poll /v1/video_status.get until status == "completed"
  4. Download result_url → cache at out_path

Both the hook clip and per-scene avatar clips use this module.
All failures are non-fatal: None is returned and the pipeline falls back to
its normal image/B-roll pipeline for that clip.

Settings required:
  HEYGEN_API_KEY    — API key from app.heygen.com → Settings → API
  HEYGEN_AVATAR_ID  — avatar ID from HeyGen's avatar library
  HEYGEN_ENABLED    — set to "true" to activate (default: false)

Optional:
  HEYGEN_VOICE_ID   — HeyGen voice ID used for text-mode fallback
  HEYGEN_ASPECT     — "16:9" (default, 1280×720) or "9:16" (720×1280 for Shorts)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retry_utils import http_get, http_post

_API_BASE      = "https://api.heygen.com"
_UPLOAD_BASE   = "https://upload.heygen.com"
_POLL_INTERVAL = 5    # seconds between status checks
_POLL_TIMEOUT  = 300  # give up after 5 minutes

_DIMENSIONS: dict[str, dict[str, int]] = {
    "16:9": {"width": 1280, "height": 720},
    "9:16": {"width": 720,  "height": 1280},
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _headers(api_key: str) -> dict[str, str]:
    return {
        "X-Api-Key":    api_key,
        "Content-Type": "application/json",
    }


def _upload_audio(audio_path: Path, api_key: str) -> tuple[str, str]:
    """Upload a local MP3 to HeyGen's asset store.

    Returns (audio_url, asset_id) — either may be empty string if absent.

    upload.heygen.com/v1/asset expects raw binary body with Content-Type
    set to the file's MIME type — NOT multipart/form-data.
    """
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    resp = http_post(
        f"{_UPLOAD_BASE}/v1/asset",
        headers={
            "X-Api-Key":    api_key,
            "Content-Type": "audio/mpeg",
        },
        data=audio_bytes,
        timeout=60,
    )
    data = resp.json().get("data", {})
    url      = data.get("url") or data.get("asset_url", "")
    asset_id = data.get("asset_id", "")
    if not url and not asset_id:
        raise RuntimeError(f"HeyGen asset upload returned no URL or asset_id: {resp.text}")
    return url, asset_id


def _create_video(
    api_key: str,
    avatar_id: str,
    audio_url: str,
    aspect: str = "16:9",
    avatar_style: str = "normal",
    audio_asset_id: str = "",
) -> str:
    """Submit a video generation job and return the video_id."""
    dim = _DIMENSIONS.get(aspect, _DIMENSIONS["16:9"])
    # Prefer asset_id (server-side reference) over public URL when both available
    if audio_asset_id:
        voice_block: dict = {"type": "audio", "audio_asset_id": audio_asset_id}
    else:
        voice_block = {"type": "audio", "audio_url": audio_url}
    payload = {
        "video_inputs": [{
            "character": {
                "type":         "avatar",
                "avatar_id":    avatar_id,
                "avatar_style": avatar_style,
            },
            "voice": voice_block,
            "background": {
                "type":  "color",
                "value": "#1a1a2e",   # dark navy background
            },
        }],
        "dimension": dim,
        "aspect_ratio": None,
    }
    resp = http_post(
        f"{_API_BASE}/v2/video/generate",
        json=payload,
        headers=_headers(api_key),
        timeout=30,
    )
    body = resp.json()
    video_id = (body.get("data") or {}).get("video_id") or body.get("video_id", "")
    if not video_id:
        raise RuntimeError(f"HeyGen video creation failed: {resp.text}")
    return video_id


def _create_video_text(
    api_key: str,
    avatar_id: str,
    voice_id: str,
    text: str,
    aspect: str = "16:9",
    avatar_style: str = "normal",
) -> str:
    """Text-to-speech fallback: create video from narration text."""
    dim = _DIMENSIONS.get(aspect, _DIMENSIONS["16:9"])
    payload = {
        "video_inputs": [{
            "character": {
                "type":         "avatar",
                "avatar_id":    avatar_id,
                "avatar_style": avatar_style,
            },
            "voice": {
                "type":       "text",
                "input_text": text[:2000],   # HeyGen text limit
                "voice_id":   voice_id,
                "speed":      1.0,
            },
            "background": {
                "type":  "color",
                "value": "#1a1a2e",
            },
        }],
        "dimension": dim,
        "aspect_ratio": None,
    }
    resp = http_post(
        f"{_API_BASE}/v2/video/generate",
        json=payload,
        headers=_headers(api_key),
        timeout=30,
    )
    body = resp.json()
    video_id = (body.get("data") or {}).get("video_id") or body.get("video_id", "")
    if not video_id:
        raise RuntimeError(f"HeyGen text video creation failed: {resp.text}")
    return video_id


def _poll_video(api_key: str, video_id: str) -> str:
    """Block until video is ready; return the download URL."""
    deadline = time.monotonic() + _POLL_TIMEOUT
    while time.monotonic() < deadline:
        time.sleep(_POLL_INTERVAL)
        resp = http_get(
            f"{_API_BASE}/v1/video_status.get",
            params={"video_id": video_id},
            headers=_headers(api_key),
            timeout=15,
        )
        data   = (resp.json().get("data") or {})
        status = data.get("status", "")
        if status == "completed":
            url = data.get("video_url", "")
            if not url:
                raise RuntimeError(f"HeyGen video {video_id} completed but no URL")
            return url
        if status == "failed":
            raise RuntimeError(
                f"HeyGen video {video_id} failed: {data.get('error', data)}"
            )
        logger.debug(f"HeyGen: polling {video_id} — status={status}")

    raise TimeoutError(f"HeyGen video {video_id} timed out after {_POLL_TIMEOUT}s")


def _download(url: str, out_path: Path) -> None:
    """Stream-download the rendered video to out_path."""
    resp = http_get(url, timeout=120, stream=True)
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65_536):
            if chunk:
                f.write(chunk)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_heygen_clip(
    audio_path: Path,
    out_path: Path,
    *,
    api_key: str,
    avatar_id: str,
    aspect: str = "16:9",
    avatar_style: str = "normal",
    fallback_text: str = "",
    voice_id: str = "",
) -> Path | None:
    """Generate a HeyGen avatar clip lip-synced to *audio_path*.

    Tries audio-driven mode first (best lip-sync quality).  Falls back to
    text-to-speech mode using *fallback_text* when audio upload fails and
    *voice_id* is provided.

    *avatar_style* — "normal" | "closeup" — controls framing. Pass "closeup"
    for high-impact intents (shock, twist, hook).

    Returns *out_path* on success, None on failure (always non-fatal).
    """
    if out_path.exists():
        logger.debug(f"HeyGen: cached clip found — {out_path.name}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not api_key:
        logger.warning("HeyGen: HEYGEN_API_KEY not set — skipping")
        return None

    if not avatar_id:
        logger.warning("HeyGen: HEYGEN_AVATAR_ID not set — skipping")
        return None

    try:
        # ── Audio-driven mode ─────────────────────────────────────────────────
        if audio_path.exists():
            logger.info(f"HeyGen: uploading audio {audio_path.name} (style={avatar_style!r})")
            audio_url, audio_asset_id = _upload_audio(audio_path, api_key)
            video_id  = _create_video(
                api_key, avatar_id, audio_url,
                aspect=aspect, avatar_style=avatar_style,
                audio_asset_id=audio_asset_id,
            )
            logger.info(f"HeyGen: video generation started — {video_id}")
            result_url = _poll_video(api_key, video_id)
            _download(result_url, out_path)
            logger.info(f"HeyGen: clip saved — {out_path.name}")
            return out_path

        # ── Text-to-speech fallback ───────────────────────────────────────────
        if fallback_text and voice_id:
            logger.info("HeyGen: audio missing, falling back to TTS mode")
            video_id   = _create_video_text(
                api_key, avatar_id, voice_id, fallback_text,
                aspect=aspect, avatar_style=avatar_style,
            )
            result_url = _poll_video(api_key, video_id)
            _download(result_url, out_path)
            logger.info(f"HeyGen: TTS clip saved — {out_path.name}")
            return out_path

        logger.warning("HeyGen: no audio and no fallback text — skipping")
        return None

    except Exception as exc:
        logger.warning(f"HeyGen: clip generation failed (non-fatal): {exc}")
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return None
