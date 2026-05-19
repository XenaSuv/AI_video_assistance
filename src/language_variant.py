"""Language variant pipeline helper.

Handles translation, re-voicing, and reassembly for non-English language
variants. Shared by all pipelines (daily, weekly, digest).
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline_helpers import (
    _get_intro_duration,
    _load_audio_durations,
    _load_cached_script,
    _needs_video_rebuild,
)
from src.script_generator import VideoScript
from src.subtitle_generator import generate_subtitles
from src.thumbnail_generator import generate_thumbnail
from src.translator import translate_script
from src.video_generator import assemble_video
from src.voice_generator import synthesize_script
from src.youtube_uploader import publish_episode


def _run_language_variant(
    english_script: VideoScript,
    run_dir: Path,
    lang_code: str,
    lang_name: str,
    voice_id: str,
    voice_model: str,
    client_secrets: Path,
    token_file: Path,
    skip_upload: bool = False,
    intro_path: Path | None = None,
    outro_path: Path | None = None,
    include_short: bool = True,
) -> dict:
    """Translate + re-voice + reassemble for a non-English language variant.

    DALL-E images and Ken Burns clips are fully reused — only TTS is re-run.
    Shared by all pipelines (daily, weekly, digest).
    """
    logger.info(f"=== {lang_name} variant ===")
    summary: dict = {}

    # 1. Translate
    script_cache = run_dir / f"script_{lang_code}.json"
    script = _load_cached_script(script_cache)
    if script is None:
        script = translate_script(english_script, lang_name)
        script.save(script_cache)
    summary["title"] = script.title

    # 2. Voice
    audio_subdir = f"audio_{lang_code}"
    audio_dir = run_dir / audio_subdir
    if not audio_dir.exists() or len(list(audio_dir.glob("*.mp3"))) < len(script.scenes):
        synthesize_script(
            script, run_dir,
            voice_id=voice_id, model_id=voice_model,
            audio_subdir=audio_subdir,
        )
        script.save(script_cache)
    else:
        logger.info(f"Reusing cached {audio_subdir}; measuring durations")
        _load_audio_durations(script, audio_dir)

    # 3 + 4 concurrently: subtitle transcription and video assembly both need
    # only the audio files produced in step 2 — they don't share output paths.
    long_video = run_dir / f"final_video_{lang_code}.mp4"

    def _do_subtitles() -> Path | None:
        try:
            return generate_subtitles(
                script,
                audio_dir,
                run_dir / f"subtitles_{lang_code}.srt",
                intro_duration=_get_intro_duration(intro_path),
            )
        except Exception as exc:
            logger.warning(f"{lang_name}: subtitle generation failed (non-fatal): {exc}")
            return None

    def _do_video() -> None:
        if not _needs_video_rebuild(long_video):
            logger.info(f"Reusing cached {long_video.name}")
            return
        clip_dir = run_dir / "clips"
        clip_paths_by_scene = {
            s.idx: [clip_dir / f"scene_{s.idx:02d}_clip_0.mp4"]
            for s in script.scenes
        }
        audio_paths_by_scene = {
            s.idx: audio_dir / f"scene_{s.idx:02d}.mp3"
            for s in script.scenes
        }
        assemble_video(
            script, clip_paths_by_scene, audio_paths_by_scene, long_video,
            intro_path=intro_path, outro_path=outro_path,
        )

    async def _subtitles_and_video() -> tuple[Path | None, None]:
        lp = asyncio.get_running_loop()
        sub, _ = await asyncio.gather(
            lp.run_in_executor(None, _do_subtitles),
            lp.run_in_executor(None, _do_video),
        )
        return sub, None

    subtitle_path, _ = asyncio.run(_subtitles_and_video())

    # 5 + 6 concurrently: short generation and thumbnail both depend on
    # long_video (written by step 4) but not on each other.
    def _do_short() -> Path | None:
        if not include_short:
            return None
        sv = run_dir / f"shorts_{lang_code}.mp4"
        if not sv.exists():
            from src.shorts_generator import build_short
            build_short(
                script, long_video, run_dir,
                audio_subdir=audio_subdir,
                out_name=sv.name,
            )
        else:
            logger.info(f"Reusing cached {sv.name}")
        return sv

    def _do_thumbnail() -> Path:
        return generate_thumbnail(
            long_video, script.title, run_dir,
            out_name=f"thumbnail_{lang_code}.jpg",
        )

    async def _short_and_thumbnail() -> tuple[Path | None, Path]:
        lp = asyncio.get_running_loop()
        sv, th = await asyncio.gather(
            lp.run_in_executor(None, _do_short),
            lp.run_in_executor(None, _do_thumbnail),
        )
        return sv, th

    short_video, thumbnail = asyncio.run(_short_and_thumbnail())

    # 6. Upload
    if skip_upload:
        logger.info(f"{lang_name}: --skip-upload; files on disk")
        summary["status"] = "built_not_uploaded"
    else:
        ids = publish_episode(
            script, long_video, short_video,
            thumbnail=thumbnail,
            subtitle_path=subtitle_path,
            subtitle_language=lang_code,
            client_secrets=client_secrets,
            token_file=token_file,
        )
        summary.update(ids)
        summary["status"] = "published"

    logger.info(f"=== {lang_name} variant done: {summary} ===")
    return summary
