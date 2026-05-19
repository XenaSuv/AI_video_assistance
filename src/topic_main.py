"""Topic deep-dive pipeline — produce a full video for one topic idea.

Formats:
  honest_review     – "I tried X — honest take"
  hidden_gems       – "5 AI tools you've never heard of"
  ai_project_build  – "I built X using only AI"
  news_with_opinion – "Big AI news + what it means for you"

Usage:
    python src/topic_main.py                        # auto-pick next queued idea
    python src/topic_main.py --format honest_review # pick from a specific format
    python src/topic_main.py --list-ideas           # show queued ideas
    python src/topic_main.py --refresh-ideas        # ask GPT for more ideas
    python src/topic_main.py --skip-upload          # build but don't publish
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
from src.comment_magnet_agent import apply_comment_magnet
from src.hook_selector import record_usage
from src.open_loop_agent import apply_open_loops
from src.pacing_auditor import audit_pacing
from src.pipeline_helpers import (
    _get_shared_outro,
    _load_audio_durations,
    _load_cached_script,
    _needs_video_rebuild,
    _setup_logging,
)
from src.shorts_generator import build_short
from src.slack_notifier import notify_failure, notify_success
from src.subtitle_generator import generate_subtitles
from src.thumbnail_ab import (
    generate_thumbnail_variants,
    pick_thumbnail,
    record_thumbnail_usage,
)
from src.topic_segment_generator import (
    TopicFormat,
    TopicIdea,
    _load_ideas,
    generate_next_topic_ideas,
    generate_topic_script,
    pick_next_topic,
)
from src.video_generator import build_video
from src.voice_generator import synthesize_script
from src.youtube_uploader import publish_episode


def run_topic_pipeline(
    format_filter: TopicFormat | None = None,
    idea_override: TopicIdea | None = None,
    skip_upload: bool = False,
) -> dict:
    """Run the full pipeline for one topic idea. Returns a summary dict."""
    date_str = dt.date.today().isoformat()

    idea = idea_override or pick_next_topic(format_filter)
    slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
    run_dir = settings.output_dir / "topic" / date_str / slug
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)
    logger.info(f"=== Topic Pipeline [{idea.format}]: {idea.subject} ===")

    summary: dict = {
        "date": date_str,
        "format": idea.format,
        "subject": idea.subject,
        "run_dir": str(run_dir),
    }

    try:
        # 1. Save idea metadata
        (run_dir / "idea.json").write_text(
            json.dumps(idea.to_dict(), indent=2)
        )

        # 2. Script
        script_cache = run_dir / "script.json"
        script = _load_cached_script(script_cache)
        if script is None:
            script = generate_topic_script(idea)

            # 2a. Open-loop pass — rewrites intro + adds payoff closings
            ol_result = apply_open_loops(script)
            script = ol_result.script
            if ol_result.modified:
                loops_info = [
                    {"teaser": l.teaser, "closing": l.closing, "target_scene": l.target_scene_idx}
                    for l in ol_result.loops
                ]
                (run_dir / "open_loops.json").write_text(
                    json.dumps(loops_info, indent=2)
                )

            # 2b. Comment-magnet pass — adds one polarizing question before sign-off
            cm_result = apply_comment_magnet(
                script, subject=idea.subject, format_name=idea.format
            )
            script = cm_result.script
            if cm_result.modified:
                (run_dir / "comment_magnet.json").write_text(
                    json.dumps({
                        "question": cm_result.question,
                        "inserted_at_scene": cm_result.inserted_at_scene,
                    }, indent=2)
                )

            # 2c. Pacing audit — detect flat zones and inject pattern interrupts
            pa_result = audit_pacing(script, subject=idea.subject)
            script = pa_result.script
            if pa_result.modified:
                audit_info = {
                    "flat_zones": [
                        {
                            "scenes": z.scene_indices,
                            "interrupt_at": z.interrupt_at,
                            "interrupt": z.interrupt_text,
                        }
                        for z in pa_result.flat_zones
                    ],
                    "scores": [
                        {"scene": s.idx, "score": s.score, "signals": s.signals}
                        for s in pa_result.scores
                    ],
                }
                (run_dir / "pacing_audit.json").write_text(
                    json.dumps(audit_info, indent=2)
                )

            script.save(script_cache)
            logger.info(f"Script saved → {script_cache}")
        else:
            logger.info("Reusing cached script")
        summary["title"] = script.title
        summary["scenes"] = len(script.scenes)

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
                script, run_dir / "audio", run_dir / "subtitles.srt"
            )
        except Exception as exc:
            logger.warning(f"Subtitle generation failed (non-fatal): {exc}")

        # 5. Video
        long_video = run_dir / "final_video.mp4"
        if _needs_video_rebuild(long_video):
            build_video(
                script,
                run_dir,
                tool=idea.format,
                use_presenter=True,
                outro_path=_get_shared_outro(),
            )
        else:
            logger.info(f"Reusing cached {long_video.name}")

        # 6. Digest Short (hook-based)
        digest_short = run_dir / "shorts.mp4"
        if not digest_short.exists():
            build_short(script, long_video, run_dir)
        else:
            logger.info(f"Reusing cached {digest_short.name}")

        # 7. Thumbnail A/B
        thumb_variants = generate_thumbnail_variants(long_video, script.title, run_dir)
        thumbnail = pick_thumbnail(thumb_variants, settings.data_dir, idea.format)
        summary["thumbnail_style"] = thumbnail.stem.removeprefix("thumbnail_")

        # 8. Upload
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
                record_usage(script.hook, video_id, settings.data_dir, idea.format)
                record_thumbnail_usage(thumbnail, video_id, settings.data_dir, idea.format)

        logger.info(f"=== Topic pipeline done: {summary.get('status', 'ok')} ===")
        notify_success(summary, "topic")

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"Topic pipeline failed:\n{tb}")
        notify_failure(exc, "topic", summary, tb)
        summary["status"] = "failed"

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Topic deep-dive video pipeline")
    p.add_argument(
        "--format",
        choices=["honest_review", "hidden_gems", "ai_project_build", "news_with_opinion"],
        help="Filter by topic format",
    )
    p.add_argument(
        "--list-ideas", action="store_true",
        help="Print queued topic ideas and exit",
    )
    p.add_argument(
        "--refresh-ideas", action="store_true",
        help="Ask GPT for new topic ideas and exit",
    )
    p.add_argument(
        "--count", type=int, default=8,
        help="Number of ideas to generate with --refresh-ideas (default 8)",
    )
    p.add_argument(
        "--skip-upload", action="store_true",
        help="Build video but don't upload to YouTube",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_ideas:
        ideas = _load_ideas()
        if not ideas:
            print("Queue is empty. Run with --refresh-ideas to populate it.")
        for i, idea in enumerate(ideas, 1):
            print(f"{i:2}. [{idea.format:20s}] {idea.suggested_title}")
            print(f"     subject: {idea.subject}")
            print(f"     angle:   {idea.angle}")
        return

    if args.refresh_ideas:
        new = generate_next_topic_ideas(count=args.count)
        print(f"Generated {len(new)} new ideas:")
        for idea in new:
            print(f"  [{idea.format}] {idea.suggested_title}")
        return

    run_topic_pipeline(
        format_filter=args.format,
        skip_upload=args.skip_upload,
    )


if __name__ == "__main__":
    main()
