"""Main pipeline orchestrator. Runs the full daily workflow:

    scrape → deduplicate → script → voice → video → shorts → thumbnail → upload
    └─ if RU_ENABLED: translate → ru-voice → reassemble → ru-shorts → ru-thumbnail → ru-upload

Safe to re-run: cached artifacts in output/YYYY-MM-DD/ are reused.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import traceback
from pathlib import Path

from loguru import logger
from moviepy.editor import AudioFileClip

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.deduplicator import SeenStories
from src.scraper import scrape_all, NewsItem
from src.script_generator import Scene, VideoScript, generate_script
from src.shorts_generator import build_short
from src.thumbnail_generator import generate_thumbnail
from src.translator import translate_script
from src.video_generator import assemble_video, build_video
from src.voice_generator import synthesize_script
from src.youtube_uploader import publish_episode


def _setup_logging(run_dir: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    logger.add(run_dir / "run.log", level="DEBUG", rotation="10 MB")


def _load_cached_script(path: Path) -> VideoScript | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return VideoScript(
        title=data["title"],
        description=data["description"],
        tags=data["tags"],
        hook=data["hook"],
        scenes=[Scene(idx=i, **{k: v for k, v in s.items() if k != "idx"})
                for i, s in enumerate(data["scenes"])],
    )


def _load_audio_durations(script: VideoScript, audio_dir: Path) -> None:
    """Populate scene.duration_sec from existing mp3 files."""
    for s in script.scenes:
        p = audio_dir / f"scene_{s.idx:02d}.mp3"
        with AudioFileClip(str(p)) as c:
            s.duration_sec = int(c.duration) + 1


# --------------------- Language variant ---------------------

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
) -> dict:
    """Translate + re-voice + reassemble for a non-English language variant.

    DALL-E images and Ken Burns clips are fully reused — only TTS is re-run.
    Returns a summary dict.
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
    audio_dir    = run_dir / audio_subdir
    if not audio_dir.exists() or len(list(audio_dir.glob("*.mp3"))) < len(script.scenes):
        synthesize_script(script, run_dir,
                          voice_id=voice_id, model_id=voice_model,
                          audio_subdir=audio_subdir)
        script.save(script_cache)
    else:
        logger.info(f"Reusing cached {audio_subdir}; measuring durations")
        _load_audio_durations(script, audio_dir)

    # 3. Video — reuse existing clips, just reassemble with new audio
    long_video = run_dir / f"final_video_{lang_code}.mp4"
    if not long_video.exists():
        clip_dir = run_dir / "clips"
        clip_paths_by_scene  = {
            s.idx: [clip_dir / f"scene_{s.idx:02d}_clip_0.mp4"]
            for s in script.scenes
        }
        audio_paths_by_scene = {
            s.idx: audio_dir / f"scene_{s.idx:02d}.mp3"
            for s in script.scenes
        }
        assemble_video(script, clip_paths_by_scene, audio_paths_by_scene, long_video)
    else:
        logger.info(f"Reusing cached {long_video.name}")

    # 4. Shorts
    short_video = run_dir / f"shorts_{lang_code}.mp4"
    if not short_video.exists():
        build_short(script, long_video, run_dir,
                    audio_subdir=audio_subdir,
                    out_name=short_video.name)
    else:
        logger.info(f"Reusing cached {short_video.name}")

    # 5. Thumbnail
    thumbnail = generate_thumbnail(
        long_video, script.title, run_dir,
        out_name=f"thumbnail_{lang_code}.jpg",
    )

    # 6. Upload
    if skip_upload:
        logger.info(f"{lang_name}: --skip-upload; files on disk")
        summary["status"] = "built_not_uploaded"
    else:
        ids = publish_episode(
            script, long_video, short_video,
            thumbnail=thumbnail,
            client_secrets=client_secrets,
            token_file=token_file,
        )
        summary.update(ids)
        summary["status"] = "published"

    logger.info(f"=== {lang_name} variant done: {summary} ===")
    return summary


# --------------------- Main pipeline ---------------------

def run_pipeline(dry_run: bool = False, skip_upload: bool = False) -> dict:
    """Execute the full pipeline. Returns a dict summary."""
    date_str = dt.date.today().isoformat()
    run_dir  = settings.output_dir / date_str
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)
    logger.info(f"=== AI News Pipeline: {date_str} ===")

    summary = {"date": date_str, "run_dir": str(run_dir)}
    seen = SeenStories(
        settings.data_dir / "seen_stories.db",
        ttl_days=settings.dedup_ttl_days,
    )

    try:
        # 1. Scrape
        news_cache = run_dir / "news.json"
        if news_cache.exists():
            logger.info("Reusing cached news.json")
            news = [NewsItem(**n) for n in json.loads(news_cache.read_text())]
        else:
            news = scrape_all(top_n=10)
            news_cache.write_text(json.dumps([n.to_dict() for n in news], indent=2))

        # 1b. Dedup
        dedup_stats = seen.stats()
        logger.info(
            f"Dedup DB: {dedup_stats['in_ttl_window']} stories in window "
            f"({dedup_stats['total_seen']} total)"
        )
        news = seen.filter_new(news)
        summary["num_news_items"] = len(news)

        if dry_run:
            logger.info("Dry run — stopping after scrape")
            return summary

        # 2. Script
        script_cache = run_dir / "script.json"
        script = _load_cached_script(script_cache)
        if script is None:
            script = generate_script(news, num_scenes=8)
            script.save(script_cache)
        summary["title"]      = script.title
        summary["num_scenes"] = len(script.scenes)

        # 3. Voice
        audio_dir = run_dir / "audio"
        if not audio_dir.exists() or len(list(audio_dir.glob("*.mp3"))) < len(script.scenes):
            synthesize_script(script, run_dir)
            script.save(script_cache)
        else:
            logger.info("Reusing cached audio; measuring durations")
            _load_audio_durations(script, audio_dir)

        summary["total_duration_sec"] = sum(s.duration_sec for s in script.scenes)

        # 4. Video
        long_video = run_dir / "final_video.mp4"
        if not long_video.exists():
            build_video(script, run_dir)
            seen.mark_featured(news)
        else:
            logger.info(f"Reusing cached {long_video.name}")

        # 5. Shorts
        short_video = run_dir / "shorts.mp4"
        if not short_video.exists():
            build_short(script, long_video, run_dir)
        else:
            logger.info(f"Reusing cached {short_video.name}")

        # 6. Thumbnail
        thumbnail = generate_thumbnail(long_video, script.title, run_dir)

        # 7. Upload (English)
        if skip_upload:
            logger.info("--skip-upload specified; leaving files on disk")
            summary["status"] = "built_not_uploaded"
        else:
            ids = publish_episode(script, long_video, short_video, thumbnail=thumbnail)
            summary.update(ids)
            summary["status"] = "published"

        # 8. Russian variant (optional)
        if settings.ru_enabled:
            ru = _run_language_variant(
                english_script = script,
                run_dir        = run_dir,
                lang_code      = "ru",
                lang_name      = "Russian",
                voice_id       = settings.ru_elevenlabs_voice_id,
                voice_model    = settings.ru_elevenlabs_model,
                client_secrets = settings.ru_youtube_client_secrets,
                token_file     = settings.ru_youtube_token_file,
                skip_upload    = skip_upload,
            )
            summary["ru"] = ru

        logger.info(f"=== DONE ===  {json.dumps(summary, indent=2)}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        summary["status"] = "failed"
        summary["error"]  = str(e)
        raise

    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",     action="store_true", help="Scrape only")
    ap.add_argument("--skip-upload", action="store_true", help="Build video but don't upload")
    args = ap.parse_args()

    run_pipeline(dry_run=args.dry_run, skip_upload=args.skip_upload)
