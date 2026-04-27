"""Weekly AI tutorial pipeline — Claude (Mon), ChatGPT (Wed), Gemini (Fri).

    pick_topic → script → voice → video → shorts → thumbnail → upload
    └─ if RU_ENABLED: translate → ru-voice → reassemble → ru-shorts → ru-thumbnail → ru-upload

Output lives in output/weekly/{tool}/YYYY-MM-DD/ so each tool's content
stays separate even when runs overlap.

Safe to re-run: cached artifacts are reused.
Run manually:
    python src/weekly_main.py --tool claude
    python src/weekly_main.py --tool chatgpt --skip-upload
    python src/weekly_main.py --tool gemini  --topic "Custom topic override"
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
from src.hook_selector import record_usage
from src.main import _load_cached_script, _run_language_variant
from src.subtitle_generator import generate_subtitles
from src.script_generator import VideoScript
from src.shorts_generator import build_short
from src.thumbnail_ab import (
    generate_thumbnail_variants,
    pick_thumbnail,
    record_thumbnail_usage,
)
from src.thumbnail_generator import generate_thumbnail
from src.video_generator import build_video
from src.voice_generator import synthesize_script
from src.weekly_script_generator import (
    generate_tutorial_script,
    get_tool,
    pick_topic,
    save_used_topic,
)
from src.weekly_shorts_generator import build_tutorial_shorts
from src.youtube_uploader import publish_episode, upload_video

_VALID_TOOLS = ["claude", "chatgpt", "gemini"]


def _setup_logging(run_dir: Path) -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO",
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}")
    logger.add(run_dir / "run.log", level="DEBUG", rotation="10 MB")


def _load_audio_durations(script: VideoScript, audio_dir: Path) -> None:
    for s in script.scenes:
        p = audio_dir / f"scene_{s.idx:02d}.mp3"
        with AudioFileClip(str(p)) as c:
            s.duration_sec = int(c.duration) + 1


def run_weekly_pipeline(
    tool_key: str = "claude",
    topic_override: str | None = None,
    skip_upload: bool = False,
) -> dict:
    """Execute the weekly tutorial pipeline for *tool_key*. Returns a summary dict."""
    if tool_key not in _VALID_TOOLS:
        raise ValueError(f"tool must be one of {_VALID_TOOLS}, got {tool_key!r}")

    tool     = get_tool(tool_key)
    date_str = dt.date.today().isoformat()
    run_dir  = settings.output_dir / "weekly" / tool_key / date_str
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)
    logger.info(f"=== Weekly {tool.name} Tutorial Pipeline: {date_str} ===")

    summary: dict = {"date": date_str, "tool": tool_key, "run_dir": str(run_dir)}

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
            topic = pick_topic(settings.data_dir, tool_key)
            topic_file.write_text(topic)
        summary["topic"] = topic

        # 2. Script
        script_cache = run_dir / "script.json"
        script = _load_cached_script(script_cache)
        if script is None:
            script = generate_tutorial_script(topic, tool_key, data_dir=settings.data_dir)
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
        subtitle_path = None
        try:
            subtitle_path = generate_subtitles(
                script, run_dir / "audio", run_dir / "subtitles.srt",
            )
        except Exception as exc:
            logger.warning(f"Subtitle generation failed (non-fatal): {exc}")

        # 5. Video
        long_video = run_dir / "final_video.mp4"
        if not long_video.exists():
            build_video(script, run_dir, tool=tool_key)
            save_used_topic(settings.data_dir, tool_key, topic)
        else:
            logger.info(f"Reusing cached {long_video.name}")

        # 6. Tutorial Shorts (one per scene that has short_narration)
        tutorial_shorts = build_tutorial_shorts(script, run_dir)
        summary["tutorial_shorts_count"] = len(tutorial_shorts)

        # 7. Digest Short (hook-based, from first scene clip)
        digest_short = run_dir / "shorts.mp4"
        if not digest_short.exists():
            build_short(script, long_video, run_dir)
        else:
            logger.info(f"Reusing cached {digest_short.name}")

        # 8. Thumbnail A/B
        thumb_variants = generate_thumbnail_variants(long_video, script.title, run_dir)
        thumbnail      = pick_thumbnail(thumb_variants, settings.data_dir, tool_key)
        summary["thumbnail_style"] = thumbnail.stem.removeprefix("thumbnail_")

        # 9. Upload (English)
        if skip_upload:
            logger.info("--skip-upload; files on disk")
            summary["status"] = "built_not_uploaded"
        else:
            ids = publish_episode(
                script, long_video, digest_short,
                thumbnail=thumbnail,
                subtitle_path=subtitle_path,
            )
            summary.update(ids)
            summary["status"] = "published"
            if video_id := ids.get("video_id"):
                record_usage(script.hook, video_id, settings.data_dir, tool_key)
                record_thumbnail_usage(thumbnail, video_id, settings.data_dir, tool_key)

            # Upload each tutorial Short separately
            short_ids: list[str] = []
            for short_path in tutorial_shorts:
                scene_idx = int(short_path.stem.split("_")[1])
                scene     = script.scenes[scene_idx]
                short_title = f"{scene.heading} | {tool.name} Tutorial"
                short_id = upload_video(
                    short_path,
                    title=short_title,
                    description=(
                        f"{scene.short_narration or scene.heading}\n\n"
                        f"Watch the full tutorial: {script.title}\n\n"
                        "#Shorts #AI #Tutorial"
                    ),
                    tags=script.tags + ["shorts", "tutorial"],
                    is_short=True,
                )
                short_ids.append(short_id)
            summary["tutorial_short_ids"] = short_ids

        # 10. Russian variant (optional)
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

        logger.info(f"=== WEEKLY {tool.name.upper()} DONE ===  {json.dumps(summary, indent=2)}")

    except Exception as e:
        logger.error(f"Weekly {tool.name} pipeline failed: {e}")
        logger.error(traceback.format_exc())
        summary["status"] = "failed"
        summary["error"]  = str(e)
        raise

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Weekly AI tutorial pipeline")
    ap.add_argument(
        "--tool", default="claude", choices=_VALID_TOOLS,
        help="Which AI tool tutorial to produce (default: claude)",
    )
    ap.add_argument("--skip-upload", action="store_true", help="Build video but don't upload")
    ap.add_argument("--topic", default=None, help="Override topic (skips rotation)")
    args = ap.parse_args()

    run_weekly_pipeline(
        tool_key       = args.tool,
        topic_override = args.topic,
        skip_upload    = args.skip_upload,
    )
