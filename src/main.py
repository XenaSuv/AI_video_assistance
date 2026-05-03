"""Main pipeline orchestrator. Runs the full daily workflow:

    scrape → deduplicate → script → voice → video → thumbnail → upload
    └─ if RU_ENABLED: translate → ru-voice → reassemble → ru-thumbnail → ru-upload

Safe to re-run: cached artifacts in output/YYYY-MM-DD/ are reused.
"""
from __future__ import annotations

import datetime as dt
import json
import re
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
from src.analytics import get_recommendations
from src.decision_engine import DecisionEngine
from src.feedback_analyzer import FeedbackAnalyzer
from src.hook_mutation_engine import HookMutationEngine
from src.performance_tracker import save_result
from src.scraper import scrape_all, NewsItem
from src.viral_selector import pick_viral_news
from src.editorial_brain import EditorialBrain
from src.humanizer_agent import HumanizerAgent
from src.micro_hook_agent import MicroHookAgent
from src.script_generator import Scene, VideoScript, generate_script
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
from src.slack_notifier import notify_success, notify_failure
from src.checkpoint import PipelineCheckpoint
from src.quality_gate import run_gate, QualityGateError
from src.cost_tracker import reset_ledger
from src.pipeline_observer import PipelineObserver


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


def _get_shared_outro() -> Path | None:
    """Return the shared outro clip used across all pipelines."""
    outro_path = settings.source_dir / "ai-news-outro.mp4"
    return outro_path if outro_path.exists() else None


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
    """Return True when a cached final video is missing or built with stale rules."""
    assembled_dir = video_path.parent / "assembled"
    legacy_end_card = any(
        path.exists() for path in (
            assembled_dir / "end_card.png",
            assembled_dir / "end_card.mp4",
        )
    )
    legacy_title_cards = any(assembled_dir.glob("title_*.mp4"))
    intro_outro_audio_missing = any(
        path.exists() and not ffmpeg_utils.has_audio_stream(path)
        for path in (
            assembled_dir / "intro_resized.mp4",
            assembled_dir / "outro_resized.mp4",
            assembled_dir / "body_with_outro.mp4",
        )
    )
    if not video_path.exists():
        return True
    if legacy_end_card or legacy_title_cards or intro_outro_audio_missing:
        if legacy_end_card:
            reason = "legacy generated end card"
        elif legacy_title_cards:
            reason = "legacy standalone title cards"
        else:
            reason = "intro/outro cache without audio stream"
        logger.warning(f"Cached video uses {reason}; rebuilding: {video_path}")
        video_path.unlink(missing_ok=True)
        for cached in (
            assembled_dir / "content.mp4",
            assembled_dir / "content_music.mp4",
            assembled_dir / "body_with_outro.mp4",
            assembled_dir / "body_with_music.mp4",
            assembled_dir / "end_card.png",
            assembled_dir / "end_card.mp4",
            assembled_dir / "intro_with_audio.mp4",
            assembled_dir / "outro_with_audio.mp4",
        ):
            cached.unlink(missing_ok=True)
        for cached in assembled_dir.glob("title_*.mp4"):
            cached.unlink(missing_ok=True)
        return True
    try:
        if ffmpeg_utils.has_audio_stream(video_path):
            return False
    except Exception as exc:
        logger.warning(f"Cached video probe failed for {video_path.name}: {exc}")

    logger.warning(f"Cached video has no audio stream; rebuilding: {video_path}")
    video_path.unlink(missing_ok=True)
    for cached in (
        assembled_dir / "content.mp4",
        assembled_dir / "content_music.mp4",
        assembled_dir / "end_card.png",
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
    include_short: bool = True,
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

    short_video: Path | None = None
    if include_short:
        short_video = run_dir / f"shorts_{lang_code}.mp4"
        if not short_video.exists():
            from src.shorts_generator import build_short
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
    ledger   = reset_ledger()
    cp       = PipelineCheckpoint(run_dir)
    observer = PipelineObserver(run_dir, pipeline="daily")
    seen = SeenStories(
        settings.data_dir / "seen_stories.db",
        ttl_days=settings.dedup_ttl_days,
    )

    try:
        # 1. Scrape, deduplicate, and select stories
        observer.step_start("scrape")
        if not cp.is_done("scrape"):
            news = scrape_all()
            dedup_stats = seen.stats()
            logger.info(
                f"Dedup DB: {dedup_stats['in_ttl_window']} stories in window "
                f"({dedup_stats['total_seen']} total)"
            )
            history = seen.recent_titles()
            news = seen.filter_new(news)

            viral_news = pick_viral_news(news, top_n=6)
            if not viral_news:
                logger.warning("Viral selector found no items; falling back to original scrape")
            else:
                news = viral_news
                logger.info(f"Selected {len(news)} viral stories")

            cp.mark_done("scrape", {
                "scraped_items": len(news),
                "history_count": len(history),
                "num_news_items": len(news),
                "num_viral_news": len(viral_news) if viral_news else 0,
                # Persist news titles so we can restore on resume
                "news_titles": [n.title for n in news],
            })
            observer.step_done("scrape",
                               scraped_items=len(news),
                               num_viral_news=len(viral_news) if viral_news else 0)
        else:
            logger.info("Step 1 (scrape) already done — reusing cached news list")
            # Restore lightweight state from checkpoint; real objects are
            # reconstructed below from cached script.json if needed.
            meta = cp.metadata("scrape")
            history = seen.recent_titles()
            # Re-scrape to get full NewsItem objects (cheap, idempotent)
            raw_news = scrape_all()
            raw_news = seen.filter_new(raw_news)
            viral_news = pick_viral_news(raw_news, top_n=6)
            news = viral_news if viral_news else raw_news
            observer.step_skip("scrape")

        meta = cp.metadata("scrape")
        summary["scraped_items"]  = meta.get("scraped_items", len(news))
        summary["history_count"]  = meta.get("history_count", 0)
        summary["num_news_items"] = meta.get("num_news_items", len(news))
        if meta.get("num_viral_news"):
            summary["num_viral_news"] = meta["num_viral_news"]

        if dry_run:
            logger.info("Dry run — stopping after scrape")
            return summary

        # 2. Script
        # Load feedback history once — reused by DecisionEngine and quality gate.
        feedback_analyzer = FeedbackAnalyzer()
        feedback_analyzer.collect_deferred_feedback(min_age_hours=24.0)
        feedback_history = feedback_analyzer.load_feedback_history()

        script_cache = run_dir / "script.json"
        observer.step_start("script")
        if not cp.is_done("script"):
            decision_engine = DecisionEngine()
            strategy = decision_engine.decide(feedback_history)

            logger.info(
                f"Strategic decisions: mode={strategy.mode}, "
                f"exploration={strategy.exploration_rate:.2f}"
            )

            top_recs = get_recommendations()
            top_hooks = top_recs.get("top_hooks", [])
            mutation_engine = HookMutationEngine()
            mutated = mutation_engine.run(
                top_hooks,
                context={
                    "angle": (max(strategy.angle_weights, key=strategy.angle_weights.get) if strategy.angle_weights else "unknown"),
                    "persona": {"style": settings.channel_name},
                    "format": (max(strategy.format_weights, key=strategy.format_weights.get) if strategy.format_weights else "unknown"),
                },
            )
            mutated_hooks = mutated.get("mutated_hooks", [])

            editorial_brain = EditorialBrain(config={"channel_name": settings.channel_name})
            editorial_plan = editorial_brain.run(
                news,
                history=history,
                channel_config={"persona": settings.channel_name},
                platform="youtube_long",
                strategy=strategy,
                hook_candidates=mutated_hooks,
            )
            summary["editorial_plan"] = {
                "selected_stories": editorial_plan.selected_stories,
                "editorial_plan": editorial_plan.editorial_plan,
                "global_style": editorial_plan.global_style,
            }

            script = _load_cached_script(script_cache)
            if script is None:
                script = generate_script(
                    news,
                    num_scenes=8,
                    data_dir=settings.data_dir,
                    editorial_plan=editorial_plan,
                )
                humanizer = HumanizerAgent()
                persona = editorial_plan.editorial_plan[0]["persona"] if editorial_plan.editorial_plan else {}

                SCENE_MARKER = "<<<SCENE_{idx}>>>"
                marked_narration = "\n\n".join(
                    f"{SCENE_MARKER.format(idx=s.idx)}\n{s.narration}"
                    for s in script.scenes
                )
                humanized = humanizer.run(
                    script=marked_narration,
                    editorial_plan=summary["editorial_plan"],
                    persona=persona,
                )
                logger.info(f"Humanized script with {len(humanized.changes)} changes: {humanized.changes}")

                humanized_text = humanized.final_script
                scene_texts: dict[int, str] = {}
                parts = re.split(r"<<<SCENE_(\d+)>>>", humanized_text)
                for i in range(1, len(parts) - 1, 2):
                    idx = int(parts[i])
                    text = parts[i + 1].strip()
                    if text:
                        scene_texts[idx] = text

                if len(scene_texts) == len(script.scenes):
                    for scene in script.scenes:
                        if scene.idx in scene_texts:
                            scene.narration = scene_texts[scene.idx]
                    logger.info("Applied humanized narration to all scenes via markers")
                else:
                    logger.warning(
                        f"Scene markers lost after humanization "
                        f"(found {len(scene_texts)}/{len(script.scenes)}); "
                        "falling back to proportional split"
                    )
                    words = humanized_text.split()
                    original_counts = [len(s.narration.split()) for s in script.scenes]
                    total_original = sum(original_counts) or 1
                    pos = 0
                    for scene, orig_count in zip(script.scenes, original_counts):
                        share = round(len(words) * orig_count / total_original)
                        chunk = words[pos: pos + share]
                        if chunk:
                            scene.narration = " ".join(chunk)
                        pos += share

                micro_hook_agent = MicroHookAgent()
                scene_plan = editorial_plan.editorial_plan[0]["scene_plan"] if editorial_plan.editorial_plan else []
                for scene in script.scenes:
                    hooked = micro_hook_agent.run(
                        script=scene.narration,
                        scene_plan=scene_plan,
                        persona=persona,
                    )
                    if hooked.final_script:
                        scene.narration = hooked.final_script
                logger.info(f"Applied micro-hooks to {len(script.scenes)} scenes")

                script.save(script_cache)

            save_for_digest(settings.data_dir, dt.date.today(), script_cache)
            cp.mark_done("script", {"title": script.title, "num_scenes": len(script.scenes)})
            observer.step_done("script", title=script.title, num_scenes=len(script.scenes))
        else:
            logger.info("Step 2 (script) already done — loading cached script.json")
            script = _load_cached_script(script_cache)
            if script is None:
                raise RuntimeError("Checkpoint says script is done but script.json is missing")
            save_for_digest(settings.data_dir, dt.date.today(), script_cache)
            observer.step_skip("script", title=script.title, num_scenes=len(script.scenes))

        summary["title"]      = script.title
        summary["num_scenes"] = len(script.scenes)

        # 3. Voice
        audio_dir = run_dir / "audio"
        observer.step_start("voice")
        if not cp.is_done("voice"):
            if not audio_dir.exists() or len(list(audio_dir.glob("*.mp3"))) < len(script.scenes):
                synthesize_script(script, run_dir)
                script.save(script_cache)
            else:
                logger.info("Reusing cached audio; measuring durations")
                _load_audio_durations(script, audio_dir)
            total_dur = sum(s.duration_sec for s in script.scenes)
            cp.mark_done("voice", {"total_duration_sec": total_dur})
            observer.step_done("voice", total_duration_sec=total_dur)
        else:
            logger.info("Step 3 (voice) already done — measuring cached audio durations")
            _load_audio_durations(script, audio_dir)
            observer.step_skip("voice", total_duration_sec=sum(s.duration_sec for s in script.scenes))

        summary["total_duration_sec"] = sum(s.duration_sec for s in script.scenes)

        # 4. Subtitles (non-fatal)
        _en_intro = settings.source_dir / "ai-news-intro.mp4"
        _en_outro = _get_shared_outro()
        subtitle_path: Path | None = run_dir / "subtitles.srt"
        subtitle_path = subtitle_path if subtitle_path.exists() else None
        observer.step_start("subtitles")
        if not cp.is_done("subtitles"):
            try:
                subtitle_path = generate_subtitles(
                    script,
                    run_dir / "audio",
                    run_dir / "subtitles.srt",
                    intro_duration=_get_intro_duration(_en_intro if _en_intro.exists() else None),
                )
                cp.mark_done("subtitles")
                observer.step_done("subtitles")
            except Exception as exc:
                logger.warning(f"Subtitle generation failed (non-fatal): {exc}")
                observer.step_fail("subtitles", exc)
        else:
            logger.info("Step 4 (subtitles) already done")
            observer.step_skip("subtitles")

        # 5. Video
        long_video = run_dir / "final_video.mp4"
        observer.step_start("video")
        if not cp.is_done("video") or _needs_video_rebuild(long_video):
            build_video(script, run_dir,
                        intro_path=_en_intro if _en_intro.exists() else None,
                        outro_path=_en_outro)
            seen.mark_featured(news)
            cp.mark_done("video")
            observer.step_done("video", file=long_video.name)
        else:
            logger.info(f"Step 5 (video) already done — reusing {long_video.name}")
            observer.step_skip("video")

        from src.broll_fetcher import get_pexels_credits
        credits = get_pexels_credits(run_dir)
        if credits:
            script.description += "\n\nVideo clips provided by Pexels:\n" + "\n".join(credits)

        # 6. Thumbnail A/B
        observer.step_start("thumbnail")
        if not cp.is_done("thumbnail"):
            thumb_variants = generate_thumbnail_variants(long_video, script.title, run_dir)
            thumbnail      = pick_thumbnail(thumb_variants, settings.data_dir, "daily")
            _style = thumbnail.stem.removeprefix("thumbnail_")
            cp.mark_done("thumbnail", {"style": _style})
            observer.step_done("thumbnail", style=_style, variants=len(thumb_variants))
        else:
            logger.info("Step 6 (thumbnail) already done")
            _style = cp.metadata("thumbnail").get("style", "default")
            thumbnail = run_dir / f"thumbnail_{_style}.jpg"
            if not thumbnail.exists():
                # Re-generate if file was deleted
                thumb_variants = generate_thumbnail_variants(long_video, script.title, run_dir)
                thumbnail      = pick_thumbnail(thumb_variants, settings.data_dir, "daily")
            observer.step_skip("thumbnail", style=_style)

        summary["thumbnail_style"] = thumbnail.stem.removeprefix("thumbnail_")

        # 6b. Quality gate — blocks publish if hard checks fail
        quality_report = run_gate(script, audio_dir, feedback_history)
        summary["quality_score"] = quality_report.score
        if quality_report.warnings:
            summary["quality_warnings"] = quality_report.warnings

        # 7. Upload (English)
        observer.step_start("upload_en")
        if not cp.is_done("upload_en"):
            if skip_upload:
                logger.info("--skip-upload specified; leaving files on disk")
                summary["status"] = "built_not_uploaded"
                cp.mark_done("upload_en", {"skipped": True})
                observer.step_done("upload_en", skipped=True)
            else:
                ids = publish_episode(
                    script, long_video, None,
                    thumbnail=thumbnail,
                    subtitle_path=subtitle_path,
                )
                summary.update(ids)
                summary["status"] = "published"
                summary["analytics"] = get_recommendations()
                if video_id := ids.get("video_id"):
                    record_usage(script.hook, video_id, settings.data_dir, "daily")
                    record_thumbnail_usage(thumbnail, video_id, settings.data_dir, "daily")
                    _ep = summary.get("editorial_plan", {}).get("editorial_plan") or [{}]
                    save_result(
                        video_id, script.hook, script.title, script.description,
                        angle=_ep[0].get("angle", ""),
                        format=_ep[0].get("format", ""),
                        platform="youtube",
                    )
                cp.mark_done("upload_en", {k: v for k, v in ids.items()})
                observer.step_done("upload_en", **{k: v for k, v in ids.items()
                                                   if isinstance(v, (str, int, float, bool))})
        else:
            logger.info("Step 7 (upload_en) already done")
            meta = cp.metadata("upload_en")
            if meta.get("skipped"):
                summary["status"] = "built_not_uploaded"
            else:
                summary.update(meta)
                summary["status"] = "published"
            observer.step_skip("upload_en")

        # 8. Russian variant (optional)
        observer.step_start("upload_ru")
        if settings.ru_enabled and not cp.is_done("upload_ru"):
            _ru_intro = settings.source_dir / "ai-novosti-intro.mp4"
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
            cp.mark_done("upload_ru", ru)
            observer.step_done("upload_ru", status=ru.get("status"))
        elif settings.ru_enabled:
            logger.info("Step 8 (upload_ru) already done")
            summary["ru"] = cp.metadata("upload_ru")
            observer.step_skip("upload_ru")
        else:
            observer.step_skip("upload_ru")

        cost_report = ledger.save(run_dir / "cost_report.json")
        ledger.log_summary()
        summary["cost_usd"] = cost_report["total_usd"]

        observer.finish(status="success", cost_usd=cost_report["total_usd"])
        notify_success(summary, "daily")
        logger.info(f"=== DONE ===  {json.dumps(summary, indent=2)}")

    except QualityGateError as e:
        logger.error(f"Quality gate blocked publish: {e}")
        summary["status"] = "quality_gate_failed"
        summary["error"]  = str(e)
        observer.finish(status="quality_gate_failed", error=str(e))
        notify_failure(e, "daily", summary, str(e))
        raise

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        summary["status"] = "failed"
        summary["error"]  = str(e)
        observer.finish(status="failed", error=str(e))
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
