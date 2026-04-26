"""Create a vertical 9:16 YouTube Short (≤60s) from the main video.

Strategy: use the hook + first scene only. Crop center 9:16, add big subtitle
burn-in so it reads on mute (Shorts almost always autoplay muted).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from loguru import logger
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx.all import crop, resize
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import VideoScript


SHORT_MAX_SECONDS = 58   # YouTube Shorts hard-limit is 60
SHORT_W, SHORT_H   = 1080, 1920
_FONT_PATH         = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_CAPTION_FONT_SIZE = 84
_CAPTION_Y_CENTER  = int(SHORT_H * 0.60)   # 60% down the frame
_MAX_TEXT_WIDTH    = SHORT_W - 120          # 60 px padding each side


def _make_vertical(clip: VideoFileClip) -> VideoFileClip:
    """Scale+center-crop a 16:9 clip to 9:16 1080×1920."""
    target_ratio = SHORT_W / SHORT_H   # 0.5625
    src_ratio    = clip.w / clip.h
    if src_ratio > target_ratio:
        new_w = int(clip.h * target_ratio)
        x1    = (clip.w - new_w) // 2
        clip  = crop(clip, x1=x1, x2=x1 + new_w)
    return resize(clip, newsize=(SHORT_W, SHORT_H))


# --------------------- PIL caption helpers ---------------------

def _load_font() -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(_FONT_PATH, _CAPTION_FONT_SIZE)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int) -> list[str]:
    """Split *text* into lines that each fit within *max_width* pixels."""
    words   = text.split()
    lines   = []
    current: list[str] = []
    for word in words:
        probe = " ".join(current + [word])
        bbox  = font.getbbox(probe)
        if bbox[2] - bbox[0] <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [text]


def _caption_clip(text: str, duration: float) -> ImageClip:
    """
    Render a single caption chunk onto a transparent 1080×1920 canvas.
    Yellow text, 4px black stroke, centered horizontally at 60% height.
    Returns a compositable ImageClip with alpha mask.
    """
    font        = _load_font()
    canvas      = Image.new("RGBA", (SHORT_W, SHORT_H), (0, 0, 0, 0))
    draw        = ImageDraw.Draw(canvas)
    line_height = _CAPTION_FONT_SIZE + 18
    lines       = _wrap_text(text, font, _MAX_TEXT_WIDTH)
    total_h     = len(lines) * line_height
    y_start     = _CAPTION_Y_CENTER - total_h // 2

    for i, line in enumerate(lines):
        bbox   = font.getbbox(line)
        text_w = bbox[2] - bbox[0]
        x      = (SHORT_W - text_w) // 2
        y      = y_start + i * line_height
        draw.text(
            (x, y), line, font=font,
            fill=(255, 255, 0, 255),
            stroke_width=4,
            stroke_fill=(0, 0, 0, 255),
        )

    rgb   = np.array(canvas.convert("RGB"))
    alpha = np.array(canvas.split()[3]) / 255.0
    return (
        ImageClip(rgb, duration=duration)
        .set_mask(ImageClip(alpha, ismask=True, duration=duration))
    )


def _burn_captions(clip: VideoFileClip, text: str) -> CompositeVideoClip:
    """Overlay animated captions (5-word chunks) onto *clip*."""
    words      = text.split()
    chunks     = [" ".join(words[i : i + 5]) for i in range(0, len(words), 5)]
    per        = clip.duration / max(len(chunks), 1)
    overlays   = [
        _caption_clip(chunk, per).set_start(i * per)
        for i, chunk in enumerate(chunks)
    ]
    return CompositeVideoClip([clip, *overlays])


# --------------------- Public API ---------------------

def build_short(script: VideoScript, main_video: Path, out_dir: Path) -> Path:
    """Produce the Shorts-ready mp4."""
    # Audio is the master clock — video is trimmed to match, not the other way.
    audio_path = out_dir / "audio" / "scene_00.mp3"
    if audio_path.exists():
        narration       = AudioFileClip(str(audio_path))
        target_duration = min(narration.duration, SHORT_MAX_SECONDS)
    else:
        narration       = None
        target_duration = SHORT_MAX_SECONDS

    logger.info(f"Short target duration: {target_duration:.1f}s")

    clip_dir     = out_dir / "clips"
    source_clips = sorted(clip_dir.glob("scene_00_clip_*.mp4")) \
                 + sorted(clip_dir.glob("scene_01_clip_*.mp4"))
    if not source_clips:
        logger.warning("No source clips found, falling back to main video head")
        source_clips = [main_video]

    base = concatenate_videoclips(
        [VideoFileClip(str(p)).without_audio() for p in source_clips],
        method="compose",
    )
    # Loop if the clips are shorter than the narration, then trim to exact length
    if base.duration < target_duration:
        from moviepy.video.fx.all import loop as fx_loop
        base = fx_loop(base, duration=target_duration)
    base = base.subclip(0, target_duration)
    base = _make_vertical(base)

    if narration is not None:
        base = base.set_audio(narration.subclip(0, target_duration))

    captioned = _burn_captions(base, script.hook)

    out = out_dir / "shorts.mp4"
    logger.info(f"Writing Short to {out} ({captioned.duration:.0f}s)")
    captioned.write_videofile(
        str(out),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        bitrate="8M",
        preset="medium",
        logger=None,
    )
    captioned.close()
    return out
