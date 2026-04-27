"""Generate individual YouTube Shorts from weekly tutorial scenes.

Each scene that has a short_narration (~120 words) becomes one vertical Short:
  - ElevenLabs TTS of short_narration  → audio_shorts/scene_NN.mp3
  - Re-use the already-rendered scene clip → center-crop to 9:16
  - Burn animated captions
  - Write shorts/scene_NN.mp4   (≤58 s)

build_tutorial_shorts() returns a list of Paths to the finished Short files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from elevenlabs.client import ElevenLabs
from loguru import logger
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.video.fx.all import crop, loop as fx_loop, resize
from PIL import Image, ImageDraw, ImageFont
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import Scene, VideoScript

SHORT_MAX_SECONDS  = 58
SHORT_W, SHORT_H   = 1080, 1920
_FONT_PATH         = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_CAPTION_FONT_SIZE = 84
_CAPTION_Y_CENTER  = int(SHORT_H * 0.60)
_MAX_TEXT_WIDTH    = SHORT_W - 120


# ─────────────────── vertical crop helpers ───────────────────

def _make_vertical(clip: VideoFileClip) -> VideoFileClip:
    target_ratio = SHORT_W / SHORT_H
    src_ratio    = clip.w / clip.h
    if src_ratio > target_ratio:
        new_w = int(clip.h * target_ratio)
        x1    = (clip.w - new_w) // 2
        clip  = crop(clip, x1=x1, x2=x1 + new_w)
    return resize(clip, newsize=(SHORT_W, SHORT_H))


# ─────────────────── caption helpers ───────────────────

def _load_font():
    try:
        return ImageFont.truetype(_FONT_PATH, _CAPTION_FONT_SIZE)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int) -> list[str]:
    words, lines, current = text.split(), [], []
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
    words    = text.split()
    chunks   = [" ".join(words[i: i + 5]) for i in range(0, len(words), 5)]
    per      = clip.duration / max(len(chunks), 1)
    overlays = [
        _caption_clip(chunk, per).set_start(i * per)
        for i, chunk in enumerate(chunks)
    ]
    return CompositeVideoClip([clip, *overlays])


# ─────────────────── TTS helper ───────────────────

def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


@retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _tts_convert(client: ElevenLabs, text: str, voice_id: str, model_id: str):
    return client.text_to_speech.convert(
        voice_id=voice_id,
        model_id=model_id,
        text=text,
        output_format="mp3_44100_128",
        voice_settings={
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.3,
            "use_speaker_boost": True,
        },
    )


def _tts_short(
    text: str,
    out_path: Path,
    voice_id: str,
    model_id: str,
) -> float:
    """Synthesize *text* to *out_path*. Returns duration in seconds."""
    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    audio_bytes = _tts_convert(client, text, voice_id, model_id)
    with open(out_path, "wb") as f:
        for chunk in audio_bytes:
            if chunk:
                f.write(chunk)
    with AudioFileClip(str(out_path)) as clip:
        return clip.duration


# ─────────────────── per-scene Short builder ───────────────────

def _build_scene_short(
    scene: Scene,
    clip_dir: Path,
    audio_path: Path,
    out_path: Path,
) -> Path:
    """Crop scene clip to vertical, overlay short_narration audio + captions."""
    # Find the rendered scene clip(s)
    source_clips = sorted(clip_dir.glob(f"scene_{scene.idx:02d}_clip_*.mp4"))
    if not source_clips:
        raise FileNotFoundError(
            f"No clips found for scene {scene.idx} in {clip_dir}"
        )

    with AudioFileClip(str(audio_path)) as aud:
        target_dur = min(aud.duration, SHORT_MAX_SECONDS)

    base = concatenate_videoclips(
        [VideoFileClip(str(p)).without_audio() for p in source_clips],
        method="compose",
    )
    if base.duration < target_dur:
        base = fx_loop(base, duration=target_dur)
    base = base.subclip(0, target_dur)
    base = _make_vertical(base)

    narration = AudioFileClip(str(audio_path)).subclip(0, target_dur)
    base      = base.set_audio(narration)

    captioned = _burn_captions(base, scene.short_narration)  # type: ignore[arg-type]

    logger.info(f"Writing tutorial Short: {out_path.name} ({captioned.duration:.0f}s)")
    captioned.write_videofile(
        str(out_path),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        bitrate="8M",
        preset="medium",
        logger=None,
    )
    captioned.close()
    return out_path


# ─────────────────── Public API ───────────────────

def build_tutorial_shorts(
    script: VideoScript,
    run_dir: Path,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
) -> list[Path]:
    """Generate one Short per scene that has a short_narration.

    Returns a list of Paths to the finished vertical mp4 files.
    Files are written to run_dir/shorts/ and audio to run_dir/audio_shorts/.
    """
    scenes_with_short = [s for s in script.scenes if s.short_narration]
    if not scenes_with_short:
        logger.info("No short_narration found in any scene — skipping tutorial shorts")
        return []

    v_id = voice_id or settings.elevenlabs_voice_id
    m_id = model_id or settings.elevenlabs_model

    audio_dir  = run_dir / "audio_shorts"
    shorts_dir = run_dir / "shorts"
    clip_dir   = run_dir / "clips"
    audio_dir.mkdir(parents=True, exist_ok=True)
    shorts_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building {len(scenes_with_short)} tutorial Shorts")
    outputs: list[Path] = []

    for scene in scenes_with_short:
        audio_path = audio_dir  / f"scene_{scene.idx:02d}.mp3"
        out_path   = shorts_dir / f"scene_{scene.idx:02d}.mp4"

        if out_path.exists():
            logger.info(f"Reusing cached Short: {out_path.name}")
            outputs.append(out_path)
            continue

        # TTS
        if not audio_path.exists():
            logger.info(f"TTS Short scene {scene.idx}: {scene.heading!r}")
            dur = _tts_short(scene.short_narration, audio_path, v_id, m_id)  # type: ignore[arg-type]
            logger.info(f"  → {audio_path.name} ({dur:.1f}s)")
        else:
            logger.info(f"Reusing cached Short audio: {audio_path.name}")

        out = _build_scene_short(scene, clip_dir, audio_path, out_path)
        outputs.append(out)

    logger.info(f"Tutorial Shorts ready: {[p.name for p in outputs]}")
    return outputs
