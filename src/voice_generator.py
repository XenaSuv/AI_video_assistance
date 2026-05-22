"""Generate narration audio via ElevenLabs for each scene.

We synthesize scene-by-scene so the video generator knows how long each clip
needs to be (RunwayML generates fixed-length clips; we loop/trim to match).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.circuit_breaker import elevenlabs_breaker
from src.cost_tracker import get_ledger
from src.ffmpeg_utils import duration as ff_duration
from src.script_generator import Scene, VideoScript

# Per-attempt timeout and retry parameters for TTS calls.
# asyncio.wait_for enforces a hard per-attempt deadline that fires even when the
# ElevenLabs HTTP call hangs indefinitely inside the thread-pool executor —
# something tenacity's synchronous @retry cannot do.
_TTS_ATTEMPT_TIMEOUT: int   = 90    # seconds before a single attempt is abandoned
_TTS_MAX_ATTEMPTS:    int   = 5     # total attempts before giving up
_TTS_BACKOFF_BASE:    float = 2.0   # initial sleep between retries (seconds)
_TTS_BACKOFF_MAX:     float = 60.0  # cap on inter-retry sleep (seconds)


def annotated_to_ssml(text: str) -> str:
    """Convert annotated text with voice tags to SSML for ElevenLabs."""
    ssml = text.replace("[PAUSE_SHORT]", '<break time="300ms"/>')
    ssml = ssml.replace("[PAUSE_LONG]", '<break time="700ms"/>')
    ssml = ssml.replace("[EMPHASIS]", '<emphasis level="strong">')
    ssml = ssml.replace("[/EMPHASIS]", '</emphasis>')
    return f"<speak>{ssml}</speak>"


def _is_retryable_elevenlabs(exc: BaseException) -> bool:
    status: int | None = getattr(exc, "status_code", None)
    if status is not None:
        return status == 429 or status >= 500
    return isinstance(exc, (ConnectionError, TimeoutError, OSError))


async def _tts_convert_async(
    client: ElevenLabs,
    text: str,
    voice_id: str,
    model_id: str,
) -> bytes:
    """Call ElevenLabs TTS with per-attempt timeout and async-compatible retry.

    Runs the blocking HTTP call in a thread-pool executor so the event loop
    stays responsive.  asyncio.wait_for() enforces a hard deadline per attempt
    that fires even when the underlying thread hangs — something tenacity's
    synchronous @retry cannot achieve inside run_in_executor.

    Retry policy:
      asyncio.TimeoutError (stuck thread)  → retry with backoff
      429 / 5xx / network errors           → retry (via _is_retryable_elevenlabs)
      CircuitOpenError / 4xx               → raise immediately (non-retryable)
      All attempts exhausted               → raise RuntimeError from last cause
    """
    loop = asyncio.get_running_loop()

    def _blocking_call() -> bytes:
        def _http() -> bytes:
            chunks = client.text_to_speech.convert(
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
            return b"".join(chunk for chunk in chunks if chunk)
        return bytes(elevenlabs_breaker.call(_http))

    last_exc: BaseException = RuntimeError("No attempts made")
    delay = _TTS_BACKOFF_BASE

    for attempt in range(1, _TTS_MAX_ATTEMPTS + 1):
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, _blocking_call),
                timeout=_TTS_ATTEMPT_TIMEOUT,
            )
        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.warning(
                f"ElevenLabs attempt {attempt}/{_TTS_MAX_ATTEMPTS} timed out "
                f"after {_TTS_ATTEMPT_TIMEOUT}s — retrying"
            )
        except Exception as exc:
            if not _is_retryable_elevenlabs(exc):
                raise
            last_exc = exc
            logger.warning(
                f"ElevenLabs attempt {attempt}/{_TTS_MAX_ATTEMPTS} failed: {exc} — retrying"
            )
        if attempt < _TTS_MAX_ATTEMPTS:
            sleep_sec = min(delay, _TTS_BACKOFF_MAX)
            logger.debug(f"ElevenLabs retry backoff: {sleep_sec:.0f}s")
            await asyncio.sleep(sleep_sec)
            delay *= 2.0

    raise RuntimeError(
        f"ElevenLabs TTS failed after {_TTS_MAX_ATTEMPTS} attempts"
    ) from last_exc


async def _synthesize_one_scene(
    scene: Scene,
    audio_dir: Path,
    v_id: str,
    m_id: str,
    client: ElevenLabs,
) -> Path:
    """Synthesize TTS for a single scene and write the mp3.  Coroutine-safe."""
    path = audio_dir / f"scene_{scene.idx:02d}.mp3"
    logger.info(
        f"TTS scene {scene.idx}: {scene.heading!r} ({len(scene.narration.split())} words)"
    )
    narration = scene.narration
    use_ssml = any(tag in narration for tag in ["[PAUSE", "[EMPHASIS"])
    if use_ssml:
        narration = annotated_to_ssml(narration)
        logger.info(f"  Using SSML for scene {scene.idx}")
    audio_bytes = await _tts_convert_async(client, narration, v_id, m_id)
    get_ledger().record_tts(f"tts-scene-{scene.idx:02d}", len(narration), m_id)
    path.write_bytes(audio_bytes)
    loop = asyncio.get_running_loop()
    dur = await loop.run_in_executor(None, ff_duration, path)
    scene.duration_sec = int(dur) + 1
    logger.info(f"  → {path.name} ({scene.duration_sec}s)")
    return path


async def async_synthesize_script(
    script: VideoScript,
    out_dir: Path,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
    audio_subdir: str = "audio",
) -> list[Path]:
    """Synthesize all scenes concurrently using async TTS with per-attempt timeout.

    Mutates script.scenes[*].duration_sec — same contract as the sync version.
    *voice_id* and *model_id* default to the values in settings, allowing
    language-variant pipelines to pass a different voice without touching config.
    *audio_subdir* lets variants write to e.g. ``audio_ru/`` alongside ``audio/``.
    """
    v_id = voice_id or settings.elevenlabs_voice_id
    m_id = model_id or settings.elevenlabs_model
    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    audio_dir = out_dir / audio_subdir
    audio_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = await asyncio.gather(*[
        _synthesize_one_scene(s, audio_dir, v_id, m_id, client)
        for s in script.scenes
    ])
    paths = sorted(paths, key=lambda p: p.name)
    total = sum(s.duration_sec for s in script.scenes)
    logger.info(f"Total narration: {total}s ({total/60:.1f} min)")
    return list(paths)


def synthesize_script(
    script: VideoScript,
    out_dir: Path,
    *,
    voice_id: str | None = None,
    model_id: str | None = None,
    audio_subdir: str = "audio",
) -> list[Path]:
    """Generate one mp3 per scene concurrently. Mutates script.scenes[*].duration_sec."""
    return asyncio.run(
        async_synthesize_script(
            script, out_dir,
            voice_id=voice_id,
            model_id=model_id,
            audio_subdir=audio_subdir,
        )
    )


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
    audio_bytes = asyncio.run(_tts_convert_async(client, hook_text, v_id, m_id))
    get_ledger().record_tts("tts-hook", len(hook_text), m_id)
    hook_path.write_bytes(audio_bytes)
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
