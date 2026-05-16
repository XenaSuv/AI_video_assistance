"""Sunday "Week in AI" digest pipeline.

    collect_recent_daily_scripts → digest_script → voice → video → thumbnail → upload
    └─ if RU_ENABLED: translate → ru-voice → reassemble → ru-thumbnail → ru-upload

Output lives in output/digest/YYYY-MM-DD/ using Sunday's date.
Safe to re-run: cached artifacts are reused.

Run manually:
    python src/digest_main.py
    python src/digest_main.py --skip-upload
    python src/digest_main.py --date 2026-04-27   # specific Sunday
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
from src.digest_script_generator import collect_week_scripts, generate_digest_script
from src.hook_selector import record_usage
from src.pipeline_orchestrator import (
    _get_intro_duration,
    _get_shared_outro,
    _load_audio_durations,
    _load_cached_script,
    _needs_video_rebuild,
    _run_language_variant,
    _setup_logging,
)
from src.subtitle_generator import generate_subtitles
from src.thumbnail_ab import (
    generate_thumbnail_variants,
    pick_thumbnail,
    record_thumbnail_usage,
)
from src.video_generator import build_video
from src.voice_generator import synthesize_script
from src.youtube_uploader import publish_episode
from src.slack_notifier import notify_success, notify_failure




def run_digest_pipeline(
    sunday: dt.date | None = None,
    skip_upload: bool = False,
) -> dict:
    """Run the Sunday digest pipeline. Returns a summary dict."""
    sunday   = sunday or dt.date.today()
    date_str = sunday.isoformat()
    week_of  = sunday.strftime("%b %d, %Y")

    run_dir = settings.output_dir / "digest" / date_str
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)
    logger.info(f"=== Sunday Digest Pipeline: week of {week_of} ===")

    summary: dict = {"date": date_str, "run_dir": str(run_dir), "type": "digest"}

    try:
        # 1. Collect the most recent daily scripts
        found_dates, week_blocks = collect_week_scripts(
            settings.output_dir, settings.data_dir, sunday
        )
        if not week_blocks:
            raise RuntimeError(
                f"No recent daily scripts found before {date_str}. "
                "Run the daily pipeline first."
            )
        summary["days_found"] = [d.isoformat() for d in found_dates]
        logger.info(f"Found {len(found_dates)} recent daily scripts: {[d.isoformat() for d in found_dates]}")

        # 2. Script
        script_cache = run_dir / "script.json"
        script = _load_cached_script(script_cache)
        if script is None:
            script = generate_digest_script(week_blocks, week_of, data_dir=settings.data_dir)
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

        # 4. Subtitles (non-fatal)
        _en_intro = settings.source_dir / "ai-digest-intro-en.mp4"
        _en_outro = _get_shared_outro()
        subtitle_path = None
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
                        outro_path=_en_outro)
        else:
            logger.info(f"Reusing cached {long_video.name}")

        # 6. Thumbnail A/B
        thumb_variants = generate_thumbnail_variants(long_video, script.title, run_dir)
        thumbnail      = pick_thumbnail(thumb_variants, settings.data_dir, "digest")
        summary["thumbnail_style"] = thumbnail.stem.removeprefix("thumbnail_")

        # 7. Upload (English)
        if skip_upload:
            logger.info("--skip-upload; files on disk")
            summary["status"] = "built_not_uploaded"
        else:
            ids = publish_episode(
                script, long_video, None,
                thumbnail=thumbnail,
                subtitle_path=subtitle_path,
            )
            summary.update(ids)
            summary["status"] = "published"
            if video_id := ids.get("video_id"):
                record_usage(script.hook, video_id, settings.data_dir, "digest")
                record_thumbnail_usage(thumbnail, video_id, settings.data_dir, "digest")

        # 8. Russian variant (optional)
        if settings.ru_enabled:
            _ru_intro = settings.source_dir / "ai-digest-intro.mp4"
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
                outro_path     = _get_shared_outro(),
                include_short  = False,
            )
            summary["ru"] = ru

        notify_success(summary, "digest")
        logger.info(f"=== DIGEST DONE ===  {json.dumps(summary, indent=2)}")

    except Exception as e:
        logger.error(f"Digest pipeline failed: {e}")
        logger.error(traceback.format_exc())
        summary["status"] = "failed"
        summary["error"]  = str(e)
        notify_failure(e, "digest", summary, traceback.format_exc())
        raise

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sunday Week-in-AI digest pipeline")
    ap.add_argument("--skip-upload", action="store_true",
                    help="Build video but skip YouTube upload")
    ap.add_argument("--date", default=None,
                    help="Sunday date to generate digest for (YYYY-MM-DD, default: today)")
    args = ap.parse_args()

    sunday = dt.date.fromisoformat(args.date) if args.date else None
    run_digest_pipeline(sunday=sunday, skip_upload=args.skip_upload)
