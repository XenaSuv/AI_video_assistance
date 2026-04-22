"""Generate narration audio via ElevenLabs for each scene.

We synthesize scene-by-scene so the video generator knows how long each clip
needs to be (RunwayML generates fixed-length clips; we loop/trim to match).
"""
from __future__ import annotations

import sys
from pathlib import Path

from elevenlabs.client import ElevenLabs
from loguru import logger
from moviepy.editor import AudioFileClip

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import VideoScript


def synthesize_script(script: VideoScript, out_dir: Path) -> list[Path]:
    """Generate one mp3 per scene. Mutates script.scenes[*].duration_sec."""
    client = ElevenLabs(api_key=settings.elevenlabs_api_key)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for scene in script.scenes:
        path = audio_dir / f"scene_{scene.idx:02d}.mp3"
        logger.info(f"TTS scene {scene.idx}: {scene.heading!r} "
                    f"({len(scene.narration.split())} words)")

        audio_bytes = client.text_to_speech.convert(
            voice_id=settings.elevenlabs_voice_id,
            model_id=settings.elevenlabs_model,
            text=scene.narration,
            output_format="mp3_44100_128",
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.75,
                "style": 0.3,
                "use_speaker_boost": True,
            },
        )

        with open(path, "wb") as f:
            for chunk in audio_bytes:
                if chunk:
                    f.write(chunk)

        # Measure duration for downstream video timing
        with AudioFileClip(str(path)) as clip:
            scene.duration_sec = int(clip.duration) + 1  # round up

        paths.append(path)
        logger.info(f"  → {path.name} ({scene.duration_sec}s)")

    total = sum(s.duration_sec for s in script.scenes)
    logger.info(f"Total narration: {total}s ({total/60:.1f} min)")
    return paths


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
