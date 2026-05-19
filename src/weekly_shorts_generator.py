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
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.ffmpeg_utils as ffmpeg_utils
from config import settings
from src.script_generator import Scene, VideoScript

SHORT_MAX_SECONDS  = 58
SHORT_W, SHORT_H   = 1080, 1920


# ─────────────────── vertical crop helpers ───────────────────

def _saliency_crop_x(frame: np.ndarray, crop_w: int) -> int:
    """Return x-offset of the most visually interesting *crop_w*-wide window."""
    h, w = frame.shape[:2]
    max_offset = w - crop_w
    if max_offset <= 0:
        return 0
    gray   = frame.mean(axis=2).astype(np.float32)
    edges  = np.abs(np.diff(gray, axis=1))
    col_sc = edges.sum(axis=0)
    wins   = np.convolve(col_sc, np.ones(crop_w, dtype=np.float32), mode="valid")
    best   = int(np.argmax(wins))
    center = (w - crop_w) // 2
    return int(best * 0.7 + center * 0.3)


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
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.75,
            style=0.3,
            use_speaker_boost=True,
        ),
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
    return ffmpeg_utils.duration(out_path)


# ─────────────────── per-scene Short builder ───────────────────

def _build_scene_short(
    scene: Scene,
    clip_dir: Path,
    audio_path: Path,
    out_path: Path,
) -> Path:
    """Crop scene clip to vertical, overlay short_narration audio + captions."""
    assembled_dir = out_path.parent.parent / "assembled"
    assembled_dir.mkdir(parents=True, exist_ok=True)

    # Find the rendered scene clip(s)
    source_clips = sorted(clip_dir.glob(f"scene_{scene.idx:02d}_clip_*.mp4"))
    if not source_clips:
        raise FileNotFoundError(
            f"No clips found for scene {scene.idx} in {clip_dir}"
        )

    target_dur = min(ffmpeg_utils.duration(audio_path), SHORT_MAX_SECONDS)

    # 1. Concat source clips for this scene
    tag = f"wshort_{scene.idx:02d}"
    if len(source_clips) == 1:
        cat_path = source_clips[0]
    else:
        cat_path = assembled_dir / f"{tag}_cat.mp4"
        cat_path = ffmpeg_utils.concat(source_clips, cat_path)

    # 2. Loop/trim to target duration
    looped_path = assembled_dir / f"{tag}_looped.mp4"
    looped_path = ffmpeg_utils.loop_and_trim(cat_path, looped_path, target_sec=target_dur)

    # 3. Saliency crop_x
    try:
        mid_t     = target_dur / 2
        mid_frame = ffmpeg_utils.get_frame(looped_path, mid_t)
        src_w, src_h = ffmpeg_utils.video_size(looped_path)
        target_ratio = SHORT_W / SHORT_H
        crop_w    = int(src_h * target_ratio)
        crop_x    = _saliency_crop_x(mid_frame, crop_w)
    except Exception as exc:
        logger.warning(f"Scene {scene.idx} saliency crop failed (non-fatal): {exc}")
        crop_x = None

    # 4. make_vertical
    vertical_path = assembled_dir / f"{tag}_vertical.mp4"
    vertical_path = ffmpeg_utils.make_vertical(
        looped_path, vertical_path, crop_x=crop_x, out_w=SHORT_W, out_h=SHORT_H
    )

    # 5. merge_av with audio
    with_audio = assembled_dir / f"{tag}_with_audio.mp4"
    with_audio = ffmpeg_utils.merge_av(vertical_path, audio_path, with_audio)

    # 6. burn_captions with scene.short_narration
    with_captions = assembled_dir / f"{tag}_with_captions.mp4"
    narration_text = scene.short_narration or scene.heading
    with_captions = ffmpeg_utils.burn_captions(
        with_audio, narration_text, with_captions,
        font_size=80, y_pct=0.60, color="yellow",
    )

    import shutil
    shutil.copy2(str(with_captions), str(out_path))

    logger.info(f"Tutorial Short written: {out_path.name} ({target_dur:.0f}s)")
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

        try:
            out = _build_scene_short(scene, clip_dir, audio_path, out_path)
            outputs.append(out)
        except Exception as exc:
            logger.warning(f"Scene {scene.idx} Short failed (non-fatal): {exc}")

    logger.info(f"Tutorial Shorts ready: {[p.name for p in outputs]}")
    return outputs
