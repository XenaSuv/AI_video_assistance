"""Create a vertical 9:16 YouTube Short (≤60s) from the main video.

Strategy: use the hook + first scene only. Crop center 9:16, add big subtitle
burn-in so it reads on mute (Shorts almost always autoplay muted).
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx.all import crop, resize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import VideoScript


SHORT_MAX_SECONDS = 58  # YouTube Shorts hard-limit is 60
SHORT_W, SHORT_H = 1080, 1920


def _make_vertical(clip: VideoFileClip) -> VideoFileClip:
    """Scale+center-crop a 16:9 clip to 9:16 1080×1920."""
    target_ratio = SHORT_W / SHORT_H  # 0.5625
    src_ratio = clip.w / clip.h
    if src_ratio > target_ratio:
        new_w = int(clip.h * target_ratio)
        x1 = (clip.w - new_w) // 2
        clip = crop(clip, x1=x1, x2=x1 + new_w)
    clip = resize(clip, newsize=(SHORT_W, SHORT_H))
    return clip


def _burn_captions(clip: VideoFileClip, text: str) -> CompositeVideoClip:
    """Burn animated captions over the clip. Chunks of ~5 words each."""
    words = text.split()
    chunk_size = 5
    chunks = [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
    per = clip.duration / max(len(chunks), 1)

    overlays = []
    t = 0.0
    for ch in chunks:
        caption = (
            TextClip(
                ch,
                fontsize=84,
                color="yellow",
                font="DejaVu-Sans-Bold",
                stroke_color="black",
                stroke_width=4,
                method="caption",
                size=(SHORT_W - 120, None),
                align="center",
            )
            .set_start(t)
            .set_duration(per)
            .set_position(("center", SHORT_H * 0.6))
        )
        overlays.append(caption)
        t += per

    return CompositeVideoClip([clip, *overlays])


def build_short(script: VideoScript, main_video: Path, out_dir: Path) -> Path:
    """Produce the Shorts-ready mp4."""
    # Strategy: narrate the hook over the visuals from scenes 0 and 1
    clip_dir = out_dir / "clips"
    source_clips = sorted(clip_dir.glob("scene_00_clip_*.mp4")) \
                 + sorted(clip_dir.glob("scene_01_clip_*.mp4"))
    if not source_clips:
        logger.warning("No source clips found, falling back to main video head")
        source_clips = [main_video]

    base = concatenate_videoclips(
        [VideoFileClip(str(p)).without_audio() for p in source_clips], method="compose"
    ).subclip(0, min(SHORT_MAX_SECONDS, sum(VideoFileClip(str(p)).duration for p in source_clips)))

    base = _make_vertical(base)

    # Use the hook's scene 0 audio — usually matches the hook text
    audio_path = out_dir / "audio" / "scene_00.mp3"
    if audio_path.exists():
        narration = AudioFileClip(str(audio_path))
        narration = narration.subclip(0, min(narration.duration, base.duration))
        base = base.set_audio(narration)

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
