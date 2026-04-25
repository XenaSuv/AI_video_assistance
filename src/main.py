"""Main pipeline orchestrator. Runs the full daily workflow:

    scrape → script → voice → video → shorts → upload

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
from config import settings
from src.scraper import scrape_all
from src.script_generator import Scene, VideoScript, generate_script
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


def run_pipeline(dry_run: bool = False, skip_upload: bool = False) -> dict:
    """Execute the full pipeline. Returns a dict summary."""
    date_str = dt.date.today().isoformat()
    run_dir = settings.output_dir / date_str
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)
    logger.info(f"=== AI News Pipeline: {date_str} ===")

    summary = {"date": date_str, "run_dir": str(run_dir)}

    try:
        # 1. Scrape
        news_cache = run_dir / "news.json"
        if news_cache.exists():
            logger.info("Reusing cached news.json")
            news_raw = json.loads(news_cache.read_text())
            from src.scraper import NewsItem
            news = [NewsItem(**n) for n in news_raw]
        else:
            news = scrape_all(top_n=10)
            news_cache.write_text(json.dumps([n.to_dict() for n in news], indent=2))
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
        summary["title"] = script.title
        summary["num_scenes"] = len(script.scenes)

        # 3. Voice (mutates scene.duration_sec)
        audio_dir = run_dir / "audio"
        if not audio_dir.exists() or len(list(audio_dir.glob("*.mp3"))) < len(script.scenes):
            synthesize_script(script, run_dir)
            # Re-save script with updated durations
            script.save(script_cache)
        else:
            logger.info("Reusing cached audio; measuring durations")
            from moviepy.editor import AudioFileClip
            for s in script.scenes:
                p = audio_dir / f"scene_{s.idx:02d}.mp3"
                with AudioFileClip(str(p)) as c:
                    s.duration_sec = int(c.duration) + 1

        summary["total_duration_sec"] = sum(s.duration_sec for s in script.scenes)

        # 4. Video
        long_video = run_dir / "final_video.mp4"
        if not long_video.exists():
            build_video(script, run_dir)
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

        # 7. Upload
        if skip_upload:
            logger.info("--skip-upload specified; leaving files on disk")
            summary["status"] = "built_not_uploaded"
        else:
            ids = publish_episode(script, long_video, short_video, thumbnail=thumbnail)
            summary.update(ids)
            summary["status"] = "published"

        logger.info(f"=== DONE ===  {json.dumps(summary, indent=2)}")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        summary["status"] = "failed"
        summary["error"] = str(e)
        raise

    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Scrape only")
    ap.add_argument("--skip-upload", action="store_true", help="Build video but don't upload")
    args = ap.parse_args()

    run_pipeline(dry_run=args.dry_run, skip_upload=args.skip_upload)
