"""Sunday "Week in AI" digest pipeline.

    collect_week_scripts → digest_script → voice → video → shorts → thumbnail → upload
    └─ if RU_ENABLED: translate → ru-voice → reassemble → ru-shorts → ru-thumbnail → ru-upload

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
from moviepy.editor import AudioFileClip

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.digest_script_generator import collect_week_scripts, generate_digest_script
from src.main import _load_audio_durations, _run_language_variant
from src.script_generator import Scene, VideoScript
from src.shorts_generator import build_short
from src.thumbnail_generator import generate_thumbnail
from src.video_generator import build_video
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
        # 1. Collect the week's daily scripts
        found_dates, week_blocks = collect_week_scripts(
            settings.output_dir, settings.data_dir, sunday
        )
        if not week_blocks:
            raise RuntimeError(
                f"No daily scripts found for the week ending {date_str}. "
                "Run the daily pipeline for at least one day first."
            )
        summary["days_found"] = [d.isoformat() for d in found_dates]
        logger.info(f"Found {len(found_dates)} daily scripts: {[d.isoformat() for d in found_dates]}")

        # 2. Script
        script_cache = run_dir / "script.json"
        script = _load_cached_script(script_cache)
        if script is None:
            script = generate_digest_script(week_blocks, week_of)
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
        _en_intro = settings.source_dir / "ai-digest-intro-en.mp4"
        _en_outro = settings.source_dir / "ai-news-outro.mp4"
        long_video = run_dir / "final_video.mp4"
        if not long_video.exists():
            build_video(script, run_dir,
                        intro_path=_en_intro if _en_intro.exists() else None,
                        outro_path=_en_outro if _en_outro.exists() else None)
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
            ids = publish_episode(script, long_video, short_video, thumbnail=thumbnail)
            summary.update(ids)
            summary["status"] = "published"

        # 8. Russian variant (optional)
        if settings.ru_enabled:
            _ru_intro = settings.source_dir / "ai-digest-intro.mp4"
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

        logger.info(f"=== DIGEST DONE ===  {json.dumps(summary, indent=2)}")

    except Exception as e:
        logger.error(f"Digest pipeline failed: {e}")
        logger.error(traceback.format_exc())
        summary["status"] = "failed"
        summary["error"]  = str(e)
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
