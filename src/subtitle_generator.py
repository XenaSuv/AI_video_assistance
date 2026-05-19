"""Generate SRT subtitle files by transcribing scene audio with OpenAI Whisper API.

Each scene MP3 is transcribed individually. Segment timestamps are offset by
the accumulated scene start time so the SRT aligns with the final video
(including any prepended intro clip).
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from openai import OpenAI

from src.retry_utils import make_openai_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.script_generator import VideoScript


def _srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _transcribe_scene(client: OpenAI, audio_path: Path) -> list[dict]:
    """Return [{start, end, text}] segments from Whisper API for one MP3."""
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
    assert resp.segments is not None
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in resp.segments
    ]


def generate_subtitles(
    script: VideoScript,
    audio_dir: Path,
    out_path: Path,
    *,
    intro_duration: float = 0.0,
) -> Path:
    """Transcribe all scene audio files and write a combined SRT subtitle file.

    *intro_duration* (seconds) offsets all timestamps to account for a
    prepended intro clip. Returns *out_path*; cached file is reused if it
    already exists. If transcription yields no segments, *out_path* is not
    written — callers should check existence before use.
    """
    if out_path.exists():
        logger.info(f"Reusing cached subtitles: {out_path.name}")
        return out_path

    client = make_openai_client()
    entries: list[tuple[float, float, str]] = []
    offset = intro_duration

    for scene in script.scenes:
        mp3 = audio_dir / f"scene_{scene.idx:02d}.mp3"
        if not mp3.exists():
            logger.warning(f"Audio missing for scene {scene.idx} — subtitle segment skipped")
            offset += scene.duration_sec
            continue

        logger.info(f"Transcribing scene {scene.idx}: {scene.heading!r}")
        for seg in _transcribe_scene(client, mp3):
            entries.append((offset + seg["start"], offset + seg["end"], seg["text"]))
        offset += scene.duration_sec

    if not entries:
        logger.warning("No subtitle segments generated — SRT not written")
        return out_path

    lines: list[str] = []
    for i, (start, end, text) in enumerate(entries, 1):
        lines += [str(i), f"{_srt_time(start)} --> {_srt_time(end)}", text, ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Subtitles written: {out_path.name} ({len(entries)} segments)")
    return out_path
