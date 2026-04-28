"""Breaking-news pipeline — triggered when a major AI announcement is detected.

    load_item → script → voice → video → shorts → thumbnail → upload
    └─ if RU_ENABLED: translate → ru-voice → reassemble → ru-shorts → ru-thumbnail → ru-upload

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
import src.ffmpeg_utils as ffmpeg_utils
from config import settings
from src.breaking_detector import mark_publish_failed, mark_published
from src.breaking_script_generator import generate_breaking_script
from src.main import _load_audio_durations, _run_language_variant
from src.subtitle_generator import generate_subtitles
from src.scraper import NewsItem
from src.script_generator import Scene, VideoScript
from src.shorts_generator import build_short
from src.thumbnail_generator import generate_thumbnail
from src.video_generator import build_video
from src.voice_generator import synthesize_script
from src.youtube_uploader import publish_episode
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
    """Execute the full breaking-news pipeline for *item*. Returns a summary dict."""
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
        # 1. Script
        script_cache = run_dir / "script.json"
        script = _load_cached_script(script_cache)
        if script is None:
            script = generate_breaking_script(item)
            script.save(script_cache)
        summary["video_title"] = script.title
        summary["num_scenes"]  = len(script.scenes)

        # 2. Voice
        audio_dir = run_dir / "audio"
        if not audio_dir.exists() or len(list(audio_dir.glob("*.mp3"))) < len(script.scenes):
            synthesize_script(script, run_dir)
            script.save(script_cache)
        else:
            logger.info("Reusing cached audio; measuring durations")
            _load_audio_durations(script, audio_dir)

        summary["total_duration_sec"] = sum(s.duration_sec for s in script.scenes)

        # 3. Subtitles (non-fatal)
        subtitle_path = None
        try:
            subtitle_path = generate_subtitles(
                script, run_dir / "audio", run_dir / "subtitles.srt",
            )
        except Exception as exc:
            logger.warning(f"Subtitle generation failed (non-fatal): {exc}")

        # 4. Video (Stable Diffusion replaces DALL-E when STABILITY_API_KEY is set)
        long_video = run_dir / "final_video.mp4"
        if not long_video.exists():
            build_video(script, run_dir, is_breaking=True)
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
            logger.info("--skip-upload; files on disk")
            summary["status"] = "built_not_uploaded"
        else:
            try:
                ids = publish_episode(
                    script, long_video, short_video,
                    thumbnail=thumbnail,
                    subtitle_path=subtitle_path,
                )
                summary.update(ids)
                summary["status"] = "published"
                mark_published(settings.data_dir, item, video_id=ids.get("long_id", ""))
            except Exception as upload_exc:
                summary["status"] = "publish_failed"
                summary["publish_error"] = str(upload_exc)
                mark_publish_failed(settings.data_dir, item, error=str(upload_exc))
                raise

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
