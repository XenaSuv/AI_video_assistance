"""Generate scene b-roll by producing a DALL-E 3 image and animating it
with a Ken Burns (slow pan) effect via MoviePy.

Cost comparison:
  RunwayML Gen-4.5  ~$0.50 per 10s clip  × 96 clips/episode  ≈ $48
  DALL-E 3 standard ~$0.04 per image     × 8 images/episode  ≈ $0.32
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import requests
from loguru import logger
from moviepy.editor import ColorClip, VideoClip
from openai import BadRequestError, OpenAI, RateLimitError, APIStatusError
from PIL import Image as PILImage
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import Scene

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
        quality="standard",
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

def _ken_burns_clip(img_path: Path, duration: float) -> VideoClip:
    """
    Animate a static image with a gentle horizontal pan (Ken Burns style).

    Loads the 1792×1024 DALL-E image once, pre-converts to a numpy array,
    then drifts a 1280×720 crop window using pure numpy slicing per frame
    (no per-frame PIL resize — fast).
    """
    img_arr = np.array(PILImage.open(str(img_path)).convert("RGB"))
    ih, iw = img_arr.shape[:2]          # 1024 × 1792 from DALL-E 3

    max_x = max(0, iw - OUT_W)          # 512px horizontal headroom
    max_y = max(0, ih - OUT_H)          # 304px vertical headroom
    pan_x = int(max_x * 0.35)           # drift 35% of headroom → ~179px
    y0    = int(max_y * 0.25)           # anchor 25% down from top

    def make_frame(t: float) -> np.ndarray:
        x = int(pan_x * t / duration)
        return img_arr[y0:y0 + OUT_H, x:x + OUT_W]

    return VideoClip(make_frame, duration=duration).set_fps(24)


# --------------------- Fallback ---------------------

def _black_placeholder(path: Path, duration: int) -> Path:
    clip = ColorClip(size=(OUT_W, OUT_H), color=(15, 20, 30), duration=duration)
    clip.write_videofile(str(path), fps=24, codec="libx264", audio=False, logger=None)
    clip.close()
    return path


# --------------------- Public API ---------------------

def generate_scene_clip(scene: Scene, clip_dir: Path) -> Path:
    """
    Produce a single .mp4 clip for *scene* at exactly scene.duration_sec length.
    Generates one DALL-E 3 image and applies a Ken Burns pan.
    Falls back to a dark placeholder if DALL-E fails.
    """
    img_dir = clip_dir.parent / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    img_path  = img_dir  / f"scene_{scene.idx:02d}.png"
    clip_path = clip_dir / f"scene_{scene.idx:02d}_clip_0.mp4"

    if clip_path.exists():
        logger.info(f"Reusing cached clip: {clip_path.name}")
        return clip_path

    try:
        generate_dalle_image(scene.visual_prompt, img_path)
        video = _ken_burns_clip(img_path, float(scene.duration_sec))
        video.write_videofile(
            str(clip_path),
            fps=24,
            codec="libx264",
            audio=False,
            preset="medium",
            logger=None,
        )
        video.close()
        logger.info(f"Scene {scene.idx} clip written: {clip_path.name}")
        return clip_path
    except Exception as e:
        logger.error(f"Scene {scene.idx} clip failed, using placeholder: {e}")
        return _black_placeholder(clip_path, scene.duration_sec)
