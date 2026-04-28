"""Generate scene b-roll by producing a DALL-E 3 image and animating it
with a Ken Burns (slow pan) effect via ffmpeg.

Cost comparison:
  RunwayML Gen-4.5  ~$0.50 per 10s clip  × 96 clips/episode  ≈ $48
  DALL-E 3 standard ~$0.04 per image     × 8 images/episode  ≈ $0.32
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
from loguru import logger
from openai import BadRequestError, OpenAI, RateLimitError, APIStatusError
from PIL import Image as PILImage
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import Scene
import src.ffmpeg_utils as ffmpeg_utils

OUT_W, OUT_H = 1280, 720


# --------------------- DALL-E 3 ---------------------

def _is_retryable_dalle(exc: BaseException) -> bool:
    if isinstance(exc, BadRequestError):
        return False  # content policy or malformed prompt — won't succeed on retry
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return isinstance(exc, (requests.exceptions.RequestException, ConnectionError, TimeoutError))


def _log_dalle_retry(retry_state) -> None:
    logger.warning(f"DALL-E retry {retry_state.attempt_number}: {retry_state.outcome.exception()}")


@retry(
    retry=retry_if_exception(_is_retryable_dalle),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    before_sleep=_log_dalle_retry,
    reraise=True,
)
def _call_dalle(client: OpenAI, prompt: str) -> bytes:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",   # landscape; gives 512px horizontal + 304px vertical pan room
        quality="hd",
        n=1,
    )
    url = response.data[0].url
    dl = requests.get(url, timeout=60)
    dl.raise_for_status()
    return dl.content


def generate_dalle_image(prompt: str, out_path: Path) -> Path:
    """Call DALL-E 3 and save the image to *out_path*. Cached if already exists."""
    if out_path.exists():
        logger.info(f"Reusing cached image: {out_path.name}")
        return out_path

    client = OpenAI(api_key=settings.openai_api_key)
    logger.info(f"DALL-E 3: {prompt[:80]}…")
    data = _call_dalle(client, prompt)
    out_path.write_bytes(data)
    logger.info(f"  → {out_path.name} ({len(data) // 1024} KB)")
    return out_path


# --------------------- Ken Burns effect ---------------------

def _ken_burns_clip(img_path: Path, duration: float, clip_path: Path) -> Path:
    """Animate a static image with one of four Ken Burns movements via ffmpeg.

    Variant is chosen deterministically from the filename so re-runs produce
    the same clip and consecutive scenes always differ.

    Variants:
      0 — pan left → right
      1 — pan right → left
      2 — zoom in  (window shrinks from full headroom to centre)
      3 — zoom out (window grows from centre to full headroom)
    """
    stem_last = img_path.stem.split("_")[-1]
    variant = int(stem_last) % 4 if stem_last.isdigit() else (hash(img_path.stem) % 4)

    return ffmpeg_utils.ken_burns(
        img_path,
        clip_path,
        duration_sec=duration,
        variant=variant,
        in_w=1792,
        in_h=1024,
        out_w=OUT_W,
        out_h=OUT_H,
    )


# --------------------- Fallback ---------------------

def _black_placeholder(path: Path, duration: int) -> Path:
    return ffmpeg_utils.black_clip(path, width=OUT_W, height=OUT_H, duration_sec=float(duration))


# --------------------- Public API ---------------------

def _try_real_screenshot(scene: Scene, tool: str | None, img_path: Path) -> bool:
    """If *scene* has a valid screenshot_key for *tool*, capture it to *img_path*.

    Returns True on success. Any failure (Playwright missing, network error,
    timeout) is logged and False is returned so the caller falls back to DALL-E.
    """
    if not tool or not scene.screenshot_key:
        return False
    try:
        from src.screenshot_capturer import capture, has_key
        if not has_key(tool, scene.screenshot_key):
            return False
        capture(tool, scene.screenshot_key, img_path)
        return img_path.exists()
    except Exception as e:
        logger.warning(f"Scene {scene.idx} screenshot failed ({tool}/{scene.screenshot_key}): {e} — falling back to DALL-E")
        return False


def generate_scene_clip(
    scene: Scene,
    clip_dir: Path,
    *,
    tool: str | None = None,
    run_dir: Path | None = None,
) -> Path:
    """
    Produce a single .mp4 clip for *scene* at exactly scene.duration_sec length.

    Priority order:
      1. infographic_data — animated chart/stat rendered entirely in Python
      2. video_query      — stock B-roll from Pexels / Pixabay (real motion)
      3. screenshot_key   — real UI screenshot with Ken Burns (weekly tutorials)
      4. visual_prompt    — DALL-E 3 image with Ken Burns effect
      5. placeholder      — solid dark frame if all else fails

    *run_dir* is passed to the B-roll fetcher so Pexels credits are saved
    alongside the video output.
    """
    img_dir = clip_dir.parent / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    img_path  = img_dir  / f"scene_{scene.idx:02d}.png"
    clip_path = clip_dir / f"scene_{scene.idx:02d}_clip_0.mp4"

    if clip_path.exists():
        logger.info(f"Reusing cached clip: {clip_path.name}")
        return clip_path

    try:
        # 1. Infographic — fully animated, skips the image→Ken-Burns pipeline
        if scene.infographic_data:
            from src.infographic_generator import generate_infographic_clip
            return generate_infographic_clip(
                scene.infographic_data,
                clip_path,
                duration_sec=float(max(scene.duration_sec, 4)),
            )

        # 2. Stock video B-roll
        if scene.video_query:
            from src.broll_fetcher import fetch_broll_clip
            result = fetch_broll_clip(scene, clip_path, run_dir=run_dir)
            if result:
                return result
            logger.info(f"Scene {scene.idx}: B-roll unavailable — falling back to DALL-E")

        # 3 & 4. Screenshot / DALL-E → Ken Burns
        if not img_path.exists():
            if not _try_real_screenshot(scene, tool, img_path):
                generate_dalle_image(scene.visual_prompt, img_path)
        else:
            logger.info(f"Reusing cached image: {img_path.name}")

        result = _ken_burns_clip(img_path, float(scene.duration_sec), clip_path)
        logger.info(f"Scene {scene.idx} clip written: {clip_path.name}")
        return result
    except Exception as e:
        logger.error(f"Scene {scene.idx} clip failed, using placeholder: {e}")
        return _black_placeholder(clip_path, scene.duration_sec)
