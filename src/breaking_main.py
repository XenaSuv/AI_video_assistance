"""Breaking-news pipeline — triggered when a major AI announcement is detected.

    load_item → short-script → voice → clips → short → upload
    └─ if RU_ENABLED: translate → ru-voice → ru-short → ru-upload

Output lives in output/breaking/YYYY-MM-DD-HHMM/ so simultaneous breaking
runs on the same day stay separate.

Safe to re-run: cached artifacts are reused.

Run manually:
    python src/breaking_main.py --item data/breaking_current.json
    python src/breaking_main.py --item data/breaking_current.json --skip-upload
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.breaking_detector import mark_publish_failed, mark_published
from src.breaking_script_generator import generate_breaking_short_script
from src.main import _load_audio_durations
from src.scraper import NewsItem
from src.script_generator import Scene, VideoScript
from src.shorts_generator import build_short, create_short_video
from src.translator import translate_script
from src.video_generator import generate_clips_for_scene
from src.voice_generator import synthesize_script
from src.youtube_uploader import upload_short
from src.slack_notifier import notify_success, notify_failure


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


def run_breaking_pipeline(item: NewsItem, skip_upload: bool = False) -> dict:
    """Execute the short-only breaking-news pipeline for *item*."""
    now      = dt.datetime.now()
    slug     = now.strftime("%Y-%m-%d-%H%M")
    run_dir  = settings.output_dir / "breaking" / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)

    logger.info(f"=== Breaking News Pipeline: [{item.source}] {item.title} ===")
    summary: dict = {
        "date":   now.date().isoformat(),
        "slug":   slug,
        "source": item.source,
        "title":  item.title,
        "url":    item.url,
        "run_dir": str(run_dir),
        "type":   "breaking",
    }

    try:
        # 1. Short script
        short_script_cache = run_dir / "short_script.json"
        short_script = _load_cached_script(short_script_cache)
        if short_script is None:
            short_script = generate_breaking_short_script(item)
            short_script.save(short_script_cache)
        summary["title"] = short_script.title
        summary["shorts_description"] = short_script.description
        summary["shorts_tags"] = short_script.tags
        summary["num_scenes"] = len(short_script.scenes)

        # 2. Voice
        audio_short_dir = run_dir / "audio_short"
        if not audio_short_dir.exists() or len(list(audio_short_dir.glob("*.mp3"))) < len(short_script.scenes):
            synthesize_script(short_script, run_dir, audio_subdir="audio_short")
            short_script.save(short_script_cache)
        else:
            logger.info("Reusing cached short audio; measuring durations")
            _load_audio_durations(short_script, audio_short_dir)

        summary["total_duration_sec"] = sum(s.duration_sec for s in short_script.scenes)

        # 3. Source visuals for the Short
        clip_dir = run_dir / "clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        for scene in short_script.scenes:
            generate_clips_for_scene(scene, clip_dir, run_dir=run_dir, is_breaking=True)

        # 4. Short assembly
        short_video = run_dir / "shorts.mp4"
        if not short_video.exists():
            try:
                build_short(
                    short_script,
                    run_dir / "_short_visual_fallback.mp4",
                    run_dir,
                    audio_subdir="audio_short",
                )
            except Exception as exc:
                logger.warning(
                    f"Breaking short generation from source clips failed: {exc} "
                    "— falling back to caption-only short"
                )
                create_short_video(
                    short_script.scenes[0].narration,
                    run_dir,
                    out_name="shorts.mp4",
                    duration_sec=min(60.0, float(short_script.scenes[0].duration_sec or 45.0)),
                    audio_path=audio_short_dir / "scene_00.mp3",
                )
        else:
            logger.info(f"Reusing cached {short_video.name}")

        # 5. Upload (English)
        if skip_upload:
            logger.info("--skip-upload; files on disk")
            summary["status"] = "built_not_uploaded"
        else:
            try:
                short_id = upload_short(
                    short_video,
                    title=short_script.title,
                    description=short_script.description,
                    tags=short_script.tags[:10] + ["shorts", "breaking news", "ai news"],
                )
                summary["short_id"] = short_id
                summary["status"] = "published"
                mark_published(settings.data_dir, item, video_id=short_id)
            except Exception as upload_exc:
                summary["status"] = "publish_failed"
                summary["publish_error"] = str(upload_exc)
                mark_publish_failed(settings.data_dir, item, error=str(upload_exc))
                raise

        # 6. Russian variant (optional)
        if settings.ru_enabled:
            ru_script_cache = run_dir / "short_script_ru.json"
            ru_script = _load_cached_script(ru_script_cache)
            if ru_script is None:
                ru_script = translate_script(short_script, "Russian")
                ru_script.save(ru_script_cache)

            ru_audio_dir = run_dir / "audio_short_ru"
            if not ru_audio_dir.exists() or len(list(ru_audio_dir.glob("*.mp3"))) < len(ru_script.scenes):
                synthesize_script(
                    ru_script,
                    run_dir,
                    voice_id=settings.ru_elevenlabs_voice_id,
                    model_id=settings.ru_elevenlabs_model,
                    audio_subdir="audio_short_ru",
                )
                ru_script.save(ru_script_cache)
            else:
                logger.info("Reusing cached Russian short audio; measuring durations")
                _load_audio_durations(ru_script, ru_audio_dir)

            ru_short_video = run_dir / "shorts_ru.mp4"
            if not ru_short_video.exists():
                try:
                    build_short(
                        ru_script,
                        run_dir / "_short_visual_fallback.mp4",
                        run_dir,
                        audio_subdir="audio_short_ru",
                        out_name=ru_short_video.name,
                    )
                except Exception as exc:
                    logger.warning(
                        f"Russian breaking short generation from source clips failed: {exc} "
                        "— falling back to caption-only short"
                    )
                    create_short_video(
                        ru_script.scenes[0].narration,
                        run_dir,
                        out_name=ru_short_video.name,
                        duration_sec=min(60.0, float(ru_script.scenes[0].duration_sec or 45.0)),
                        audio_path=ru_audio_dir / "scene_00.mp3",
                    )

            ru: dict = {"title": ru_script.title}
            if skip_upload:
                ru["status"] = "built_not_uploaded"
            else:
                ru["short_id"] = upload_short(
                    ru_short_video,
                    title=ru_script.title,
                    description=ru_script.description,
                    tags=ru_script.tags[:10] + ["shorts", "ai news"],
                    client_secrets=settings.ru_youtube_client_secrets,
                    token_file=settings.ru_youtube_token_file,
                )
                ru["status"] = "published"
            summary["ru"] = ru

        notify_success(summary, "breaking")
        logger.info(f"=== BREAKING DONE ===  {json.dumps(summary, indent=2)}")

    except Exception as e:
        logger.error(f"Breaking pipeline failed: {e}")
        logger.error(traceback.format_exc())
        # Preserve a more specific status set by an inner step (e.g. "publish_failed")
        if summary.get("status") not in ("publish_failed",):
            summary["status"] = "failed"
        summary["error"] = str(e)
        notify_failure(e, "breaking", summary, traceback.format_exc())
        raise

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Breaking news video pipeline")
    ap.add_argument("--item", required=True,
                    help="Path to JSON file with the NewsItem (written by breaking_detector.py --check)")
    ap.add_argument("--skip-upload", action="store_true",
                    help="Build video but skip all uploads")
    args = ap.parse_args()

    item_data = json.loads(Path(args.item).read_text())
    news_item  = NewsItem(**item_data)
    run_breaking_pipeline(news_item, skip_upload=args.skip_upload)
