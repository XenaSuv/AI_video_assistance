"""Create a vertical 9:16 YouTube Short (≤60s) from the main video.

.. deprecated::
    This module is the **legacy** Shorts implementation.  It clips and
    center-crops an already-rendered long-form video, which means every Short
    is derivative content with no independent editorial angle.

    **Use instead**: ``src.shorts_pipeline.ShortsPipeline`` (driven by
    ``src.shorts_engine_v2.ShortsEngineV2``), which builds each Short from
    scratch as a standalone hypothesis-test — an independent idea in 25–30 s.

    Existing callers (breaking_main, weekly_main, topic_main, language_variant,
    pipeline_orchestrator) are kept on this path until migrated.  New code
    must NOT import from this module.

Strategy: use the hook + first scene only. Crop center 9:16, add big subtitle
burn-in so it reads on mute (Shorts almost always autoplay muted).
"""
from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.ffmpeg_utils as ffmpeg_utils
from config import settings
from src.script_generator import VideoScript

SHORT_MAX_SECONDS = 58   # YouTube Shorts hard-limit is 60
SHORT_W, SHORT_H   = 1080, 1920


def _saliency_crop_x(frame: np.ndarray, crop_w: int) -> int:
    """Return x-offset of the most visually interesting *crop_w*-wide window.

    Uses horizontal edge density (column-wise gradient magnitude) as a
    proxy for visual saliency.  Pure numpy — no extra dependencies.
    Falls back to center if the frame is too small.
    """
    h, w = frame.shape[:2]
    max_offset = w - crop_w
    if max_offset <= 0:
        return 0
    gray   = frame.mean(axis=2).astype(np.float32)
    edges  = np.abs(np.diff(gray, axis=1))          # H × (W-1)
    col_sc = edges.sum(axis=0)                       # saliency per column
    wins   = np.convolve(col_sc, np.ones(crop_w, dtype=np.float32), mode="valid")
    best   = int(np.argmax(wins))
    # Blend toward center (weight 0.3) to avoid extreme off-center crops
    center = (w - crop_w) // 2
    return int(best * 0.7 + center * 0.3)


def _resolve_music_path() -> Path | None:
    """Return the background music Path, or None if not configured / file missing."""
    raw = settings.background_music_path.strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = settings.source_dir / raw
    if not p.exists():
        logger.warning(f"Background music not found: {p} — skipping")
        return None
    return p


# --------------------- Public API ---------------------

def build_short(
    script: VideoScript,
    main_video: Path,
    out_dir: Path,
    *,
    audio_subdir: str = "audio",
    out_name: str = "shorts.mp4",
) -> Path:
    """Produce the Shorts-ready mp4.

    .. deprecated::
        Use ``src.shorts_pipeline.ShortsPipeline`` instead.

    *audio_subdir* lets language variants point at e.g. ``audio_ru/``.
    *out_name* lets variants write ``shorts_ru.mp4`` alongside ``shorts.mp4``.
    """
    warnings.warn(
        "shorts_generator.build_short() is deprecated; "
        "use src.shorts_pipeline.ShortsPipeline instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    assembled_dir = out_dir / "assembled"
    assembled_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / out_name

    # Audio is the master clock — video is trimmed to match, not the other way.
    audio_path = out_dir / audio_subdir / "scene_00.mp3"
    if audio_path.exists():
        target_duration = min(ffmpeg_utils.duration(audio_path), SHORT_MAX_SECONDS)
    else:
        audio_path      = None  # type: ignore[assignment]
        target_duration = SHORT_MAX_SECONDS

    logger.info(f"Short target duration: {target_duration:.1f}s")

    clip_dir     = out_dir / "clips"
    source_clips = sorted(clip_dir.glob("scene_00_clip_*.mp4")) \
                 + sorted(clip_dir.glob("scene_01_clip_*.mp4"))
    if not source_clips:
        logger.warning("No source clips found, falling back to main video head")
        source_clips = [main_video]

    # 1. Concat source clips
    if len(source_clips) == 1:
        base_path = source_clips[0]
    else:
        cat_path = assembled_dir / "short_source_cat.mp4"
        base_path = ffmpeg_utils.concat(source_clips, cat_path, video_only=True)

    # 2. Loop/trim to target duration
    looped_path = assembled_dir / "short_looped.mp4"
    looped_path = ffmpeg_utils.loop_and_trim(base_path, looped_path, target_sec=target_duration)

    # 3. Saliency crop_x
    try:
        mid_t   = target_duration / 2
        mid_frame = ffmpeg_utils.get_frame(looped_path, mid_t)
        src_w, src_h = ffmpeg_utils.video_size(looped_path)
        target_ratio = SHORT_W / SHORT_H
        crop_w = int(src_h * target_ratio)
        crop_x = _saliency_crop_x(mid_frame, crop_w)
    except Exception as exc:
        logger.warning(f"Saliency crop failed (non-fatal): {exc} — using centre crop")
        crop_x = None

    # 4. make_vertical
    vertical_path = assembled_dir / "short_vertical.mp4"
    vertical_path = ffmpeg_utils.make_vertical(
        looped_path, vertical_path, crop_x=crop_x, out_w=SHORT_W, out_h=SHORT_H
    )

    # 5. merge_av with audio
    if audio_path is not None:
        with_audio = assembled_dir / "short_with_audio.mp4"
        with_audio = ffmpeg_utils.merge_av(vertical_path, audio_path, with_audio)
    else:
        with_audio = vertical_path

    # 6. burn_captions
    with_captions = assembled_dir / "short_with_captions.mp4"
    with_captions = ffmpeg_utils.burn_captions(
        with_audio, script.hook, with_captions,
        font_size=80, y_pct=0.60, color="yellow",
    )

    # 7. mix_music (non-fatal if file missing)
    music_path = _resolve_music_path()
    if music_path:
        final_path = assembled_dir / "short_with_music.mp4"
        result = ffmpeg_utils.mix_music(
            with_captions, music_path, final_path,
            volume=settings.shorts_music_volume,
            fade_in=1.0, fade_out=1.5,
        )
        import shutil
        shutil.copy2(str(result), str(out))
    else:
        import shutil
        shutil.copy2(str(with_captions), str(out))

    logger.info(f"Short written: {out} ({target_duration:.0f}s)")
    return out


def add_big_captions(video: Path, text: str, output: Path) -> Path:
    """Burn large centered captions onto a short-friendly clip."""
    output.parent.mkdir(parents=True, exist_ok=True)
    return ffmpeg_utils.burn_captions(
        video,
        text,
        output,
        font_size=72,
        y_pct=0.45,
        color="white",
    )


def create_short_video(
    script_text: str,
    out_dir: Path,
    *,
    out_name: str = "shorts_mvp.mp4",
    duration_sec: float = 20.0,
    audio_path: Path | None = None,
) -> Path:
    """Create a simple static Shorts video with big burned-in captions.

    .. deprecated::
        Use ``src.shorts_pipeline.ShortsPipeline`` instead.
    """
    warnings.warn(
        "shorts_generator.create_short_video() is deprecated; "
        "use src.shorts_pipeline.ShortsPipeline instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    assembled_dir = out_dir / "assembled"
    assembled_dir.mkdir(parents=True, exist_ok=True)

    base_clip = assembled_dir / "shorts_mvp_base.mp4"
    captioned = assembled_dir / "shorts_mvp_captioned.mp4"
    out_path = out_dir / out_name

    ffmpeg_utils.black_clip(base_clip, width=1080, height=1920, duration_sec=duration_sec)
    add_big_captions(base_clip, script_text, captioned)
    if audio_path is not None and audio_path.exists():
        with_audio = assembled_dir / "shorts_mvp_with_audio.mp4"
        ffmpeg_utils.merge_av(captioned, audio_path, with_audio)
        shutil.copy2(str(with_audio), str(out_path))
    else:
        shutil.copy2(str(captioned), str(out_path))
    logger.info(f"Created shorts MVP video: {out_path}")
    return out_path
