"""Generate narration audio via ElevenLabs for each scene.

We synthesize scene-by-scene so the video generator knows how long each clip
needs to be (RunwayML generates fixed-length clips; we loop/trim to match).
"""
from __future__ import annotations

import sys
from pathlib import Path

from elevenlabs.client import ElevenLabs
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import VideoScript
from src.ffmpeg_utils import duration as ff_duration
from src.cost_tracker import get_ledger


def annotated_to_ssml(text: str) -> str:
    """Convert annotated text with voice tags to SSML for ElevenLabs."""
    # Replace annotations with SSML
    ssml = text.replace("[PAUSE_SHORT]", '<break time="300ms"/>')
    ssml = ssml.replace("[PAUSE_LONG]", '<break time="700ms"/>')
    ssml = ssml.replace("[EMPHASIS]", '<emphasis level="strong">')
    ssml = ssml.replace("[/EMPHASIS]", '</emphasis>')
    # Wrap in speak tag
    return f"<speak>{ssml}</speak>"


def _is_retryable_elevenlabs(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


def _log_retry(retry_state) -> None:
    logger.warning(
        f"ElevenLabs retry {retry_state.attempt_number}: {retry_state.outcome.exception()}"
    )


@retry(
    retry=retry_if_exception(_is_retryable_elevenlabs),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    before_sleep=_log_retry,
    reraise=True,
)
def _tts_convert(client: ElevenLabs, text: str, voice_id: str, model_id: str, use_ssml: bool = False) -> object:
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


def synthesize_script(
    script: VideoScript,
    out_dir: Path,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
    audio_subdir: str = "audio",
) -> list[Path]:
    """Generate one mp3 per scene. Mutates script.scenes[*].duration_sec.

    *voice_id* and *model_id* default to the values in settings, allowing
    language-variant pipelines to pass a different voice without touching config.
    *audio_subdir* lets variants write to e.g. ``audio_ru/`` alongside ``audio/``.
    """
    v_id = voice_id or settings.elevenlabs_voice_id
    m_id = model_id or settings.elevenlabs_model
    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    audio_dir = out_dir / audio_subdir
    audio_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for scene in script.scenes:
        path = audio_dir / f"scene_{scene.idx:02d}.mp3"
        logger.info(f"TTS scene {scene.idx}: {scene.heading!r} "
                    f"({len(scene.narration.split())} words)")

        # Convert annotated narration to SSML if it contains annotations
        narration = scene.narration
        use_ssml = any(tag in narration for tag in ["[PAUSE", "[EMPHASIS"])
        if use_ssml:
            narration = annotated_to_ssml(narration)
            logger.info(f"  Using SSML for scene {scene.idx}")

        audio_bytes = _tts_convert(client, narration, v_id, m_id, use_ssml)
        get_ledger().record_tts(f"tts-scene-{scene.idx:02d}", len(narration), m_id)

        with open(path, "wb") as f:
            for chunk in audio_bytes:
                if chunk:
                    f.write(chunk)

        from src.ffmpeg_utils import duration as _ff_dur
        scene.duration_sec = int(_ff_dur(path)) + 1

        paths.append(path)
        logger.info(f"  → {path.name} ({scene.duration_sec}s)")

    total = sum(s.duration_sec for s in script.scenes)
    logger.info(f"Total narration: {total}s ({total/60:.1f} min)")
    return paths


def synthesize_hook(
    script: VideoScript,
    out_dir: Path,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
    audio_subdir: str = "audio",
) -> Path:
    """Generate hook.mp3 from script.hook — used as audio input for the presenter clip.

    Cached: skips synthesis if hook.mp3 already exists in *out_dir/audio_subdir*.
    """
    hook_text = (script.hook or "").strip()
    if not hook_text:
        raise ValueError("Script has no hook text to synthesize")

    v_id = voice_id or settings.elevenlabs_voice_id
    m_id = model_id or settings.elevenlabs_model
    client = ElevenLabs(api_key=settings.elevenlabs_api_key)

    audio_dir = out_dir / audio_subdir
    audio_dir.mkdir(parents=True, exist_ok=True)
    hook_path = audio_dir / "hook.mp3"

    if hook_path.exists():
        return hook_path

    logger.info(f"TTS hook ({len(hook_text.split())} words): {hook_text[:60]!r}...")
    audio_bytes = _tts_convert(client, hook_text, v_id, m_id)
    with open(hook_path, "wb") as f:
        for chunk in audio_bytes:
            if chunk:
                f.write(chunk)

    dur = ff_duration(hook_path)
    logger.info(f"  → {hook_path.name} ({dur:.1f}s)")
    return hook_path


if __name__ == "__main__":
    import json
    script_path = settings.output_dir / "script.json"
    data = json.loads(script_path.read_text())

    from src.script_generator import Scene, VideoScript
    script = VideoScript(
        title=data["title"],
        description=data["description"],
        tags=data["tags"],
        hook=data["hook"],
        scenes=[Scene(idx=i, **s) for i, s in enumerate(data["scenes"])],
    )
    synthesize_script(script, settings.output_dir)
