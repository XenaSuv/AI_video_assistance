"""Backward-compat shim: re-exports all public symbols from the split submodules.

All production callers use ``import src.ffmpeg_utils as ffmpeg_utils`` and
all test files patch ``"src.ffmpeg_utils.<name>"``.  This module keeps that
interface stable while the actual implementation lives in:
  - src/ffmpeg_probe.py    — shared helpers + probe/info functions
  - src/ffmpeg_overlay.py  — title card, chyron, caption overlay
  - src/ffmpeg_compose.py  — av merge, concat, ken burns, music, transforms
"""
from __future__ import annotations

from src.ffmpeg_compose import (
    black_clip,
    concat,
    end_card_clip,
    frames_to_video,
    ken_burns,
    loop_and_trim,
    make_vertical,
    merge_av,
    mix_music,
    quote_card_clip,
    resize_video,
    zoom_clip,
)
from src.ffmpeg_overlay import (
    burn_captions,
    burn_chyron,
    overlay_title_card,
    title_card,
)
from src.ffmpeg_probe import (
    _CLIP_TIMEOUT,
    _FONT,
    _LONG_TIMEOUT,
    _PROBE_TIMEOUT,
    _esc,
    _run,
    duration,
    ensure_audio_stream,
    get_frame,
    has_audio_stream,
    probe,
    video_size,
)

__all__ = [
    # probe
    "_CLIP_TIMEOUT", "_FONT", "_LONG_TIMEOUT", "_PROBE_TIMEOUT",
    "_esc", "_run",
    "duration", "ensure_audio_stream", "get_frame",
    "has_audio_stream", "probe", "video_size",
    # compose
    "black_clip", "concat", "end_card_clip", "frames_to_video",
    "ken_burns", "loop_and_trim", "make_vertical", "merge_av",
    "mix_music", "quote_card_clip", "resize_video", "zoom_clip",
    # overlay
    "burn_captions", "burn_chyron", "overlay_title_card", "title_card",
]
