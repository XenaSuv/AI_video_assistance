"""Generate a custom thumbnail by scoring frames from the final video.

Strategy:
  - Sample 16 frames evenly across the video (skipping the first/last 5%)
  - Score each frame by colorfulness + contrast; reject under/over-exposed frames
  - Resize the winner to 1280×720 and overlay the episode title with a gradient bar
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from loguru import logger
from moviepy.editor import VideoFileClip

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

THUMB_W, THUMB_H = 1280, 720
_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONT_SIZE = 54
_CANDIDATES = 16


# --------------------- Frame scoring ---------------------

def _score_frame(arr: np.ndarray) -> float:
    """Higher is better. Returns -1 for frames that are too dark or blown out."""
    mean = float(arr.mean())
    if mean < 35 or mean > 225:
        return -1.0
    colorfulness = float(arr.std(axis=(0, 1)).mean())  # per-channel std
    contrast = float(arr.std())
    return colorfulness * 0.65 + contrast * 0.35


def _sample_frames(video_path: Path, n: int = _CANDIDATES) -> list[tuple[float, np.ndarray]]:
    """Return (timestamp, rgb_array) pairs sampled evenly across the video."""
    with VideoFileClip(str(video_path)) as clip:
        duration = clip.duration
        timestamps = [
            duration * (0.05 + 0.90 * i / (n - 1))
            for i in range(n)
        ]
        return [(t, clip.get_frame(t)) for t in timestamps]


# --------------------- Overlay ---------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _wrap_title(title: str, max_chars: int = 38) -> list[str]:
    """Split title into at most 2 lines, breaking on whole words."""
    lines = textwrap.wrap(title, width=max_chars)
    if len(lines) > 2:
        # Force into 2 lines by re-wrapping with a wider limit
        lines = textwrap.wrap(title, width=max_chars + 12)[:2]
    return lines


def _overlay_title(img: Image.Image, title: str) -> Image.Image:
    """Composite a dark gradient bar + white title text onto the image."""
    w, h = img.size
    base = img.convert("RGBA")

    # Semi-transparent gradient covering the bottom third
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    gradient_h = h // 3
    for y in range(gradient_h):
        alpha = int(200 * (y / gradient_h))
        draw.rectangle(
            [(0, h - gradient_h + y), (w, h - gradient_h + y + 1)],
            fill=(0, 0, 0, alpha),
        )
    base = Image.alpha_composite(base, overlay)

    # Title text
    draw2 = ImageDraw.Draw(base)
    font = _load_font(_FONT_SIZE)
    lines = _wrap_title(title)

    line_height = _FONT_SIZE + 10
    total_text_h = len(lines) * line_height
    y_start = h - total_text_h - 28  # 28px padding from bottom

    for i, line in enumerate(lines):
        y = y_start + i * line_height
        # Drop shadow
        draw2.text((22, y + 3), line, font=font, fill=(0, 0, 0, 220))
        # Main text
        draw2.text((20, y), line, font=font, fill=(255, 255, 255, 255))

    return base.convert("RGB")


# --------------------- Public API ---------------------

def generate_thumbnail(
    video_path: Path,
    title: str,
    out_dir: Path,
    *,
    out_name: str = "thumbnail.jpg",
) -> Path:
    """Extract the best frame from *video_path*, overlay *title*, and save
    as *out_name* inside *out_dir*.  Returns the saved path.

    *out_name* lets language variants write ``thumbnail_ru.jpg`` alongside
    the default ``thumbnail.jpg``.
    """
    out_path = out_dir / out_name
    if out_path.exists():
        logger.info(f"Reusing cached thumbnail: {out_path.name}")
        return out_path

    logger.info("Scoring candidate frames for thumbnail…")
    candidates = _sample_frames(video_path)

    best_t, best_frame = max(candidates, key=lambda x: _score_frame(x[1]))
    score = _score_frame(best_frame)
    logger.info(f"Best frame at t={best_t:.1f}s (score={score:.1f})")

    img = Image.fromarray(best_frame.astype(np.uint8)).convert("RGB")
    img = img.resize((THUMB_W, THUMB_H), Image.LANCZOS)
    img = _overlay_title(img, title)
    img.save(str(out_path), "JPEG", quality=92, optimize=True)

    logger.info(f"Thumbnail saved: {out_path.name} ({out_path.stat().st_size // 1024} KB)")
    return out_path
