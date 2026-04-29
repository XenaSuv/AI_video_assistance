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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.ffmpeg_utils as ffmpeg_utils
from config import settings
from src.deduplicator import SeenStories
from src.digest_script_generator import save_for_digest
from src.hook_selector import record_usage
from src.shorts_mvp import run_shorts_mvp
from src.analytics import get_recommendations
from src.performance_tracker import save_result
from src.scraper import scrape_all, NewsItem
from src.viral_selector import pick_viral_news
from src.script_generator import Scene, VideoScript, generate_script
from src.shorts_generator import build_short
from src.subtitle_generator import generate_subtitles
from src.thumbnail_ab import (
    generate_thumbnail_variants,
    pick_thumbnail,
    record_thumbnail_usage,
)
from src.thumbnail_generator import generate_thumbnail
from src.translator import translate_script
from src.video_generator import assemble_video, build_video
from src.voice_generator import synthesize_script
from src.youtube_uploader import publish_episode
from src.tiktok_uploader import post_short as tiktok_post_short
from src.slack_notifier import notify_success, notify_failure
from src.weekly_shorts_generator import build_tutorial_shorts
from src.youtube_uploader import upload_video


def _setup_logging(run_dir: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    logger.add(run_dir / "run.log", level="DEBUG", rotation="10 MB")


def _get_intro_duration(intro_path: Path | None) -> float:
    """Return duration in seconds of the intro clip, or 0.0 if not present."""
    if not intro_path or not intro_path.exists():
        return 0.0
    return ffmpeg_utils.duration(intro_path)


def _load_cached_script(path: Path) -> VideoScript | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return VideoScript(
        title=data["title"],
        description=data["description"],
        tags=data["tags"],
        hook=data["hook"],
        hook_variants=data.get("hook_variants", []),
        scenes=[Scene(idx=i, **{k: v for k, v in s.items() if k != "idx"})
                for i, s in enumerate(data["scenes"])],
    )


def _load_audio_durations(script: VideoScript, audio_dir: Path) -> None:
    """Populate scene.duration_sec from existing mp3 files."""
    for s in script.scenes:
        p = audio_dir / f"scene_{s.idx:02d}.mp3"
        s.duration_sec = int(ffmpeg_utils.duration(p)) + 1


def _needs_video_rebuild(video_path: Path) -> bool:
    """Return True when a cached final video is missing or has no audio stream."""
    if not video_path.exists():
        return True
    try:
        if ffmpeg_utils.has_audio_stream(video_path):
            return False
    except Exception as exc:
        logger.warning(f"Cached video probe failed for {video_path.name}: {exc}")

    logger.warning(f"Cached video has no audio stream; rebuilding: {video_path}")
    video_path.unlink(missing_ok=True)
    assembled_dir = video_path.parent / "assembled"
    for cached in (
        assembled_dir / "content.mp4",
        assembled_dir / "content_music.mp4",
        assembled_dir / "end_card.mp4",
    ):
        cached.unlink(missing_ok=True)
    for cached in assembled_dir.glob("title_*.mp4"):
        cached.unlink(missing_ok=True)
    return True


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
    intro_path: Path | None = None,
    outro_path: Path | None = None,
) -> dict:
    """Translate + re-voice + reassemble for a non-English language variant.

    DALL-E images and Ken Burns clips are fully reused — only TTS is re-run.
    *intro_path* / *outro_path* are prepended / appended when supplied.
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

    # 3. Subtitles — transcribe from translated audio
    subtitle_path: Path | None = None
    try:
        subtitle_path = generate_subtitles(
            script,
            audio_dir,
            run_dir / f"subtitles_{lang_code}.srt",
            intro_duration=_get_intro_duration(intro_path),
        )
    except Exception as exc:
        logger.warning(f"{lang_name}: subtitle generation failed (non-fatal): {exc}")

    # 4. Video — reuse existing clips, just reassemble with new audio
    long_video = run_dir / f"final_video_{lang_code}.mp4"
    if _needs_video_rebuild(long_video):
        clip_dir = run_dir / "clips"
        clip_paths_by_scene  = {
            s.idx: [clip_dir / f"scene_{s.idx:02d}_clip_0.mp4"]
            for s in script.scenes
        }
        audio_paths_by_scene = {
            s.idx: audio_dir / f"scene_{s.idx:02d}.mp3"
            for s in script.scenes
        }
        assemble_video(script, clip_paths_by_scene, audio_paths_by_scene, long_video,
                       intro_path=intro_path, outro_path=outro_path)
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
            subtitle_path=subtitle_path,
            subtitle_language=lang_code,
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

        # 1c. Pick only the most viral stories for shorts/script generation.
        viral_news = pick_viral_news(news, top_n=2)
        if not viral_news:
            logger.warning("Viral selector found no items; falling back to original scrape")
        else:
            news = viral_news
            summary["num_viral_news"] = len(news)
            logger.info(f"Selected {len(news)} viral stories")

        if dry_run:
            logger.info("Dry run — stopping after scrape")
            return summary

        # 2. Script
        script_cache = run_dir / "script.json"
        script = _load_cached_script(script_cache)
        if script is None:
            script = generate_script(news, num_scenes=8, data_dir=settings.data_dir)
            script.save(script_cache)
        # Persist a digest copy so the Sunday workflow can find it across CI runs
        save_for_digest(settings.data_dir, dt.date.today(), script_cache)
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

        # 4. Subtitles (non-fatal: errors are logged and pipeline continues)
        _en_intro = settings.source_dir / "ai-news-intro.mp4"
        _en_outro = settings.source_dir / "ai-news-outro.mp4"
        subtitle_path: Path | None = None
        try:
            subtitle_path = generate_subtitles(
                script,
                run_dir / "audio",
                run_dir / "subtitles.srt",
                intro_duration=_get_intro_duration(_en_intro if _en_intro.exists() else None),
            )
        except Exception as exc:
            logger.warning(f"Subtitle generation failed (non-fatal): {exc}")

        # 5. Video
        long_video = run_dir / "final_video.mp4"
        if _needs_video_rebuild(long_video):
            build_video(script, run_dir,
                        intro_path=_en_intro if _en_intro.exists() else None,
                        outro_path=_en_outro if _en_outro.exists() else None)
            seen.mark_featured(news)
        else:
            logger.info(f"Reusing cached {long_video.name}")

        # 4b. Append B-roll credits to description (Pexels attribution)
        from src.broll_fetcher import get_pexels_credits
        credits = get_pexels_credits(run_dir)
        if credits:
            script.description += "\n\nVideo clips provided by Pexels:\n" + "\n".join(credits)

        # 5. MVP: run a lightweight Shorts-only flow for the first 1-2 items (with video clips).
        for item in news[:2]:
            try:
                run_shorts_mvp(item)
            except Exception as exc:
                logger.warning(f"Shorts MVP failed for item: {exc}")

        # 5a. Shorts (hook-based digest short)
        short_video = run_dir / "shorts.mp4"
        if not short_video.exists():
            build_short(script, long_video, run_dir)
        else:
            logger.info(f"Reusing cached {short_video.name}")

        # 5b. Per-scene Shorts (one per scene with short_narration)
        scene_shorts = build_tutorial_shorts(script, run_dir)
        summary["scene_shorts_count"] = len(scene_shorts)

        # 7. Thumbnail A/B
        thumb_variants = generate_thumbnail_variants(long_video, script.title, run_dir)
        thumbnail      = pick_thumbnail(thumb_variants, settings.data_dir, "daily")
        summary["thumbnail_style"] = thumbnail.stem.removeprefix("thumbnail_")

        # 8. Upload (English)
        if skip_upload:
            logger.info("--skip-upload specified; leaving files on disk")
            summary["status"] = "built_not_uploaded"
        else:
            ids = publish_episode(
                script, long_video, short_video,
                thumbnail=thumbnail,
                subtitle_path=subtitle_path,
            )
            summary.update(ids)
            summary["status"] = "published"
            summary["analytics"] = get_recommendations()
            if video_id := ids.get("video_id"):
                record_usage(script.hook, video_id, settings.data_dir, "daily")
                record_thumbnail_usage(thumbnail, video_id, settings.data_dir, "daily")
                save_result(video_id, script.hook, script.title, script.description)

        # 8b. Upload per-scene Shorts
        if not skip_upload and scene_shorts:
            scene_short_ids: list[str] = []
            for short_path in scene_shorts:
                try:
                    scene_idx = int(short_path.stem.split("_")[1])
                    scene     = script.scenes[scene_idx]
                    sid = upload_video(
                        short_path,
                        title=f"{scene.heading} | AI News",
                        description=(
                            f"{scene.short_narration or scene.heading}\n\n"
                            f"Full video: {script.title}\n\n"
                            "#Shorts #AI #AINews #ArtificialIntelligence"
                        ),
                        tags=script.tags + ["shorts", "ai news"],
                        is_short=True,
                    )
                    scene_short_ids.append(sid)
                except Exception as exc:
                    logger.warning(f"Per-scene Short upload failed (non-fatal): {exc}")
            summary["scene_short_ids"] = scene_short_ids

        # 9. TikTok Short (optional)
        if settings.tiktok_enabled and not skip_upload and short_video.exists():
            try:
                caption = f"{script.hook}\n\n#AI #ArtificialIntelligence #TechNews #AINews"
                summary["tiktok_id"] = tiktok_post_short(short_video, caption)
            except Exception as e:
                logger.warning(f"TikTok upload failed (non-fatal): {e}")
                summary["tiktok_error"] = str(e)

        # 10. Russian variant (optional)
        if settings.ru_enabled:
            _ru_intro = settings.source_dir / "ai-novosti-intro.mp4"
            _ru_outro = settings.source_dir / "ai-novosti-outro.mp4"
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
                intro_path     = _ru_intro if _ru_intro.exists() else None,
                outro_path     = _ru_outro if _ru_outro.exists() else None,
            )
            summary["ru"] = ru

        notify_success(summary, "daily")
        logger.info(f"=== DONE ===  {json.dumps(summary, indent=2)}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        summary["status"] = "failed"
        summary["error"]  = str(e)
        notify_failure(e, "daily", summary, traceback.format_exc())
        raise

    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run",     action="store_true", help="Scrape only")
    ap.add_argument("--skip-upload", action="store_true", help="Build video but don't upload")
    args = ap.parse_args()

    run_pipeline(dry_run=args.dry_run, skip_upload=args.skip_upload)
