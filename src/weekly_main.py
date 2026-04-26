"""Weekly Claude tutorial pipeline.

    pick_topic → script → voice → video → shorts → thumbnail → upload

Safe to re-run: cached artifacts in output/weekly/YYYY-MM-DD/ are reused.
Run manually:
    python src/weekly_main.py
    python src/weekly_main.py --skip-upload
    python src/weekly_main.py --topic "Custom topic override"
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
from src.script_generator import Scene, VideoScript
from src.shorts_generator import build_short
from src.thumbnail_generator import generate_thumbnail
from src.video_generator import assemble_video, build_video
from src.voice_generator import synthesize_script
from src.weekly_script_generator import (
    generate_tutorial_script,
    pick_topic,
    _save_used_topic,
)
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
    for s in script.scenes:
        p = audio_dir / f"scene_{s.idx:02d}.mp3"
        with AudioFileClip(str(p)) as c:
            s.duration_sec = int(c.duration) + 1


def run_weekly_pipeline(
    topic_override: str | None = None,
    skip_upload: bool = False,
) -> dict:
    """Execute the weekly tutorial pipeline. Returns a summary dict."""
    date_str = dt.date.today().isoformat()
    run_dir  = settings.output_dir / "weekly" / date_str
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)
    logger.info(f"=== Weekly Claude Tutorial Pipeline: {date_str} ===")

    summary: dict = {"date": date_str, "run_dir": str(run_dir)}

    try:
        # 1. Pick / confirm topic
        topic_file = run_dir / "topic.txt"
        if topic_override:
            topic = topic_override
            topic_file.write_text(topic)
        elif topic_file.exists():
            topic = topic_file.read_text().strip()
            logger.info(f"Reusing cached topic: {topic}")
        else:
            topic = pick_topic(settings.data_dir)
            topic_file.write_text(topic)
        summary["topic"] = topic

        # 2. Script
        script_cache = run_dir / "script.json"
        script = _load_cached_script(script_cache)
        if script is None:
            script = generate_tutorial_script(topic)
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
            _save_used_topic(settings.data_dir, topic)
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
            logger.info("--skip-upload; files on disk")
            summary["status"] = "built_not_uploaded"
        else:
            ids = publish_episode(script, long_video, short_video, thumbnail=thumbnail)
            summary.update(ids)
            summary["status"] = "published"

        logger.info(f"=== WEEKLY DONE ===  {json.dumps(summary, indent=2)}")

    except Exception as e:
        logger.error(f"Weekly pipeline failed: {e}")
        logger.error(traceback.format_exc())
        summary["status"] = "failed"
        summary["error"]  = str(e)
        raise

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Weekly Claude tutorial pipeline")
    ap.add_argument("--skip-upload", action="store_true", help="Build video but don't upload")
    ap.add_argument("--topic", default=None, help="Override topic (skip rotation)")
    args = ap.parse_args()

    run_weekly_pipeline(
        topic_override=args.topic,
        skip_upload=args.skip_upload,
    )
