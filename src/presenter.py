"""D-ID Talks API — lip-sync a talking-head presenter clip from hook audio.

Workflow:
  1. Avatar PNG + hook MP3 are sent to D-ID as base64 data URIs (no hosting required)
  2. D-ID renders a lip-synced MP4 (audio embedded) and returns a result_url
  3. The clip is downloaded and cached at *out_path*

The clip is non-fatal: if D-ID is unavailable or the key is missing, None is
returned and the pipeline continues without a presenter.

Settings required:
  DID_API_KEY           — base64 credential from d-id.com dashboard (the part
                          after "Basic " in their curl examples)
  PRESENTER_AVATAR_PATH — local path to avatar PNG/JPG (default: assets/avatar.png)
  PRESENTER_ENABLED     — set to "true" to activate (default: false)
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.retry_utils import http_get, http_post

_DID_API       = "https://api.d-id.com"
_POLL_INTERVAL = 3    # seconds between status polls
_POLL_TIMEOUT  = 180  # give up after 3 minutes


def _b64_uri(path: Path) -> str:
    """Encode *path* as a base64 data URI (image or audio)."""
    ext = path.suffix.lower()
    mime_map = {
        ".png":  "image/png",
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".mp3":  "audio/mpeg",
        ".wav":  "audio/wav",
    }
    mime = mime_map.get(ext, "application/octet-stream")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def generate_presenter_clip(
    hook_audio: Path,
    out_path: Path,
    *,
    api_key: str,
    avatar_path: Path,
) -> Path | None:
    """Submit *hook_audio* + *avatar_path* to D-ID; download result to *out_path*.

    Returns *out_path* on success, None on any failure (non-fatal).
    Both assets are sent as base64 data URIs — no public URL hosting needed.
    The output MP4 contains video + audio (the hook speech is embedded by D-ID).
    """
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not api_key:
        logger.warning("Presenter: DID_API_KEY not set — skipping")
        return None

    if not avatar_path.exists():
        logger.warning(f"Presenter: avatar not found at {avatar_path} — skipping")
        return None

    try:
        headers = {
            "Authorization": f"Basic {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "source_url": _b64_uri(avatar_path),
            "script": {
                "type": "audio",
                "audio_url": _b64_uri(hook_audio),
            },
            "config": {
                "stitch": True,          # blend head into original background
                "result_format": "mp4",
            },
        }

        # ── Create talk ────────────────────────────────────────────────────────
        resp = http_post(
            f"{_DID_API}/talks",
            json=payload,
            headers=headers,
            timeout=60,
        )
        talk_id = resp.json()["id"]
        logger.info(f"Presenter: D-ID talk created — id={talk_id}")

        # ── Poll for completion ────────────────────────────────────────────────
        deadline = time.monotonic() + _POLL_TIMEOUT
        result_url: str | None = None
        while time.monotonic() < deadline:
            time.sleep(_POLL_INTERVAL)
            poll = http_get(
                f"{_DID_API}/talks/{talk_id}",
                headers=headers,
                timeout=15,
            )
            data   = poll.json()
            status = data.get("status")
            if status == "done":
                result_url = data["result_url"]
                break
            if status == "error":
                raise RuntimeError(f"D-ID talk failed: {data.get('error', data)}")
            logger.debug(f"Presenter: polling — status={status}")

        if not result_url:
            raise TimeoutError(f"D-ID talk {talk_id} timed out after {_POLL_TIMEOUT}s")

        # ── Download result ────────────────────────────────────────────────────
        dl = http_get(result_url, timeout=120, stream=True)
        with open(out_path, "wb") as f:
            for chunk in dl.iter_content(chunk_size=65_536):
                if chunk:
                    f.write(chunk)

        logger.info(f"Presenter clip saved: {out_path.name}")
        return out_path

    except Exception as exc:
        logger.warning(f"Presenter: D-ID failed (non-fatal): {exc}")
        # Clean up partial download
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return None
