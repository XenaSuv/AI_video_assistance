"""Daily AI news pipeline — orchestration class.

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
from src.live_state import LiveState


# ── Stateless helpers ─────────────────────────────────────────────────────────

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
        scenes=[
            Scene(idx=i, **{k: v for k, v in s.items() if k != "idx"})
            for i, s in enumerate(data["scenes"])
        ],
    )


def _load_audio_durations(script: VideoScript, audio_dir: Path) -> None:
    """Populate scene.duration_sec from existing mp3 files."""
    for s in script.scenes:
        p = audio_dir / f"scene_{s.idx:02d}.mp3"
        s.duration_sec = int(ffmpeg_utils.duration(p)) + 1


def _setup_logging(run_dir: Path) -> None:
    logger.remove()
    logger.add(
        sys.stderr, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
    )
    logger.add(run_dir / "run.log", level="DEBUG", rotation="10 MB")


def _needs_video_rebuild(video_path: Path) -> bool:
    """Return True when a cached final video is missing or built with stale rules."""
    assembled_dir = video_path.parent / "assembled"
    legacy_end_card = any(
        path.exists()
        for path in (
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
    Shared by all pipelines (daily, weekly, digest).
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
    audio_dir = run_dir / audio_subdir
    if not audio_dir.exists() or len(list(audio_dir.glob("*.mp3"))) < len(script.scenes):
        synthesize_script(
            script, run_dir,
            voice_id=voice_id, model_id=voice_model,
            audio_subdir=audio_subdir,
        )
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

    # 4. Video — reuse existing clips, reassemble with new audio
    long_video = run_dir / f"final_video_{lang_code}.mp4"
    if _needs_video_rebuild(long_video):
        clip_dir = run_dir / "clips"
        clip_paths_by_scene = {
            s.idx: [clip_dir / f"scene_{s.idx:02d}_clip_0.mp4"]
            for s in script.scenes
        }
        audio_paths_by_scene = {
            s.idx: audio_dir / f"scene_{s.idx:02d}.mp3"
            for s in script.scenes
        }
        assemble_video(
            script, clip_paths_by_scene, audio_paths_by_scene, long_video,
            intro_path=intro_path, outro_path=outro_path,
        )
    else:
        logger.info(f"Reusing cached {long_video.name}")

    short_video: Path | None = None
    if include_short:
        short_video = run_dir / f"shorts_{lang_code}.mp4"
        if not short_video.exists():
            from src.shorts_generator import build_short
            build_short(
                script, long_video, run_dir,
                audio_subdir=audio_subdir,
                out_name=short_video.name,
            )
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


# ── Orchestrator ──────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    """Orchestrates the full daily AI news pipeline end-to-end.

    Per-run state is stored as instance attributes so step methods share context
    without passing large argument lists. A new instance should be created for
    each pipeline run.
    """

    def run(self, dry_run: bool = False, skip_upload: bool = False) -> dict:
        """Execute the full pipeline. Returns a summary dict."""
        date_str = dt.date.today().isoformat()
        self._run_dir = settings.output_dir / date_str
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        logger.info(f"=== AI News Pipeline: {date_str} ===")

        self._dry_run = dry_run
        self._skip_upload = skip_upload
        self._summary: dict = {"date": date_str, "run_dir": str(self._run_dir)}
        self._ledger = reset_ledger()
        self._cp = PipelineCheckpoint(self._run_dir)
        self._live = LiveState(settings.data_dir / "live_state.json")
        self._live.start("daily", steps_total=8)
        self._observer = PipelineObserver(
            self._run_dir, pipeline="daily", live_state=self._live
        )
        self._seen = SeenStories(
            settings.data_dir / "seen_stories.db",
            ttl_days=settings.dedup_ttl_days,
        )
        # Per-run state populated by step methods
        self._script: VideoScript | None = None
        self._news: list[NewsItem] = []
        self._history: list[str] = []
        self._feedback_history: list = []
        self._subtitle_path: Path | None = None
        self._en_intro: Path | None = None
        self._en_outro: Path | None = None
        self._long_video: Path | None = None
        self._short_video: Path | None = None
        self._thumbnail: Path | None = None

        try:
            self._step_scrape()
            if dry_run:
                logger.info("Dry run — stopping after scrape")
                return self._summary
            self._step_script()
            self._step_voice()
            self._step_subtitles()
            self._step_video()
            self._step_thumbnail()
            self._step_quality_gate()
            self._step_upload_en()
            self._step_upload_ru()
            self._finalize()

        except QualityGateError as e:
            logger.error(f"Quality gate blocked publish: {e}")
            self._summary["status"] = "quality_gate_failed"
            self._summary["error"] = str(e)
            self._live.log_event(f"Quality gate failed: {e}")
            self._observer.finish(status="quality_gate_failed", error=str(e))
            notify_failure(e, "daily", self._summary, str(e))
            raise

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.error(traceback.format_exc())
            self._summary["status"] = "failed"
            self._summary["error"] = str(e)
            self._live.log_event(f"Pipeline error: {type(e).__name__}: {e}")
            self._observer.finish(status="failed", error=str(e))
            notify_failure(e, "daily", self._summary, traceback.format_exc())
            raise

        return self._summary

    # ── Step methods ──────────────────────────────────────────────────────────

    def _step_scrape(self) -> None:
        self._observer.step_start("scrape")
        if not self._cp.is_done("scrape"):
            news = scrape_all()
            dedup_stats = self._seen.stats()
            logger.info(
                f"Dedup DB: {dedup_stats['in_ttl_window']} stories in window "
                f"({dedup_stats['total_seen']} total)"
            )
            self._history = self._seen.recent_titles()
            news = self._seen.filter_new(news)

            viral_news = pick_viral_news(news, top_n=6)
            if not viral_news:
                logger.warning("Viral selector found no items; falling back to original scrape")
            else:
                news = viral_news
                logger.info(f"Selected {len(news)} viral stories")

            self._news = news
            self._live.log_event(
                f"Scraped {len(news)} stories"
                + (f", selected {len(viral_news)} viral" if viral_news else "")
            )
            self._cp.mark_done("scrape", {
                "scraped_items": len(news),
                "history_count": len(self._history),
                "num_news_items": len(news),
                "num_viral_news": len(viral_news) if viral_news else 0,
                "news_titles": [n.title for n in news],
            })
            self._observer.step_done(
                "scrape",
                scraped_items=len(news),
                num_viral_news=len(viral_news) if viral_news else 0,
            )
        else:
            logger.info("Step 1 (scrape) already done — reusing cached news list")
            self._history = self._seen.recent_titles()
            raw_news = scrape_all()
            raw_news = self._seen.filter_new(raw_news)
            viral_news = pick_viral_news(raw_news, top_n=6)
            self._news = viral_news if viral_news else raw_news
            self._observer.step_skip("scrape")

        meta = self._cp.metadata("scrape")
        self._summary["scraped_items"] = meta.get("scraped_items", len(self._news))
        self._summary["history_count"] = meta.get("history_count", 0)
        self._summary["num_news_items"] = meta.get("num_news_items", len(self._news))
        if meta.get("num_viral_news"):
            self._summary["num_viral_news"] = meta["num_viral_news"]

    def _step_script(self) -> None:
        # Feedback history is loaded unconditionally — reused by quality gate later.
        feedback_analyzer = FeedbackAnalyzer()
        feedback_analyzer.collect_deferred_feedback(min_age_hours=24.0)
        self._feedback_history = feedback_analyzer.load_feedback_history()

        script_cache = self._run_dir / "script.json"
        self._observer.step_start("script")
        if not self._cp.is_done("script"):
            decision_engine = DecisionEngine()
            strategy = decision_engine.decide(self._feedback_history)

            logger.info(
                f"Strategic decisions: mode={strategy.mode}, "
                f"exploration={strategy.exploration_rate:.2f}"
            )
            _top_angle = (
                max(strategy.angle_weights, key=strategy.angle_weights.get)
                if strategy.angle_weights else None
            )
            _top_format = (
                max(strategy.format_weights, key=strategy.format_weights.get)
                if strategy.format_weights else None
            )
            self._live.set_strategy({
                "mode": strategy.mode,
                "exploration_rate": round(strategy.exploration_rate, 2),
                "confidence": round(strategy.confidence, 2),
                "top_angle": _top_angle,
                "top_format": _top_format,
            })
            self._live.log_event(f"Strategy: {strategy.mode} mode, angle={_top_angle}")

            top_recs = get_recommendations()
            top_hooks = top_recs.get("top_hooks", [])
            mutation_engine = HookMutationEngine()
            mutated = mutation_engine.run(
                top_hooks,
                context={
                    "angle": (
                        max(strategy.angle_weights, key=strategy.angle_weights.get)
                        if strategy.angle_weights else "unknown"
                    ),
                    "persona": {"style": settings.channel_name},
                    "format": (
                        max(strategy.format_weights, key=strategy.format_weights.get)
                        if strategy.format_weights else "unknown"
                    ),
                },
            )
            mutated_hooks = mutated.get("mutated_hooks", [])

            editorial_brain = EditorialBrain(config={"channel_name": settings.channel_name})
            editorial_plan = editorial_brain.run(
                self._news,
                history=self._history,
                channel_config={"persona": settings.channel_name},
                platform="youtube_long",
                strategy=strategy,
                hook_candidates=mutated_hooks,
            )
            self._summary["editorial_plan"] = {
                "selected_stories": editorial_plan.selected_stories,
                "editorial_plan": editorial_plan.editorial_plan,
                "global_style": editorial_plan.global_style,
            }

            script = _load_cached_script(script_cache)
            if script is None:
                script = generate_script(
                    self._news,
                    num_scenes=8,
                    data_dir=settings.data_dir,
                    editorial_plan=editorial_plan,
                )
                persona = (
                    editorial_plan.editorial_plan[0]["persona"]
                    if editorial_plan.editorial_plan else {}
                )
                script = self._humanize_script(script, persona)
                script = self._apply_micro_hooks(script, editorial_plan, persona)
                script.save(script_cache)

            self._script = script
            save_for_digest(settings.data_dir, dt.date.today(), script_cache)
            self._live.log_event(
                f"Script ready: \"{script.title}\" ({len(script.scenes)} scenes)"
            )
            self._cp.mark_done("script", {"title": script.title, "num_scenes": len(script.scenes)})
            self._observer.step_done("script", title=script.title, num_scenes=len(script.scenes))
        else:
            logger.info("Step 2 (script) already done — loading cached script.json")
            script = _load_cached_script(script_cache)
            if script is None:
                raise RuntimeError("Checkpoint says script is done but script.json is missing")
            self._script = script
            save_for_digest(settings.data_dir, dt.date.today(), script_cache)
            self._observer.step_skip("script", title=script.title, num_scenes=len(script.scenes))

        self._summary["title"] = self._script.title
        self._summary["num_scenes"] = len(self._script.scenes)

    def _step_voice(self) -> None:
        audio_dir = self._run_dir / "audio"
        self._observer.step_start("voice")
        if not self._cp.is_done("voice"):
            if not audio_dir.exists() or len(list(audio_dir.glob("*.mp3"))) < len(self._script.scenes):
                synthesize_script(self._script, self._run_dir)
                self._script.save(self._run_dir / "script.json")
            else:
                logger.info("Reusing cached audio; measuring durations")
                _load_audio_durations(self._script, audio_dir)
            total_dur = sum(s.duration_sec for s in self._script.scenes)
            self._live.log_event(f"Audio synthesized: {total_dur:.0f}s total")
            self._cp.mark_done("voice", {"total_duration_sec": total_dur})
            self._observer.step_done("voice", total_duration_sec=total_dur)
        else:
            logger.info("Step 3 (voice) already done — measuring cached audio durations")
            _load_audio_durations(self._script, audio_dir)
            self._observer.step_skip(
                "voice",
                total_duration_sec=sum(s.duration_sec for s in self._script.scenes),
            )

        self._summary["total_duration_sec"] = sum(s.duration_sec for s in self._script.scenes)

    def _step_subtitles(self) -> None:
        en_intro = settings.source_dir / "ai-news-intro.mp4"
        self._en_intro = en_intro if en_intro.exists() else None
        self._en_outro = _get_shared_outro()

        cached_srt = self._run_dir / "subtitles.srt"
        self._subtitle_path = cached_srt if cached_srt.exists() else None

        self._observer.step_start("subtitles")
        if not self._cp.is_done("subtitles"):
            try:
                self._subtitle_path = generate_subtitles(
                    self._script,
                    self._run_dir / "audio",
                    self._run_dir / "subtitles.srt",
                    intro_duration=_get_intro_duration(self._en_intro),
                )
                self._cp.mark_done("subtitles")
                self._observer.step_done("subtitles")
            except Exception as exc:
                logger.warning(f"Subtitle generation failed (non-fatal): {exc}")
                self._observer.step_fail("subtitles", exc)
        else:
            logger.info("Step 4 (subtitles) already done")
            self._observer.step_skip("subtitles")

    def _step_video(self) -> None:
        self._long_video = self._run_dir / "final_video.mp4"
        self._observer.step_start("video")
        rebuild_video = not self._cp.is_done("video") or _needs_video_rebuild(self._long_video)
        if rebuild_video:
            build_video(
                self._script, self._run_dir,
                intro_path=self._en_intro,
                outro_path=self._en_outro,
                use_presenter=settings.presenter_enabled,
            )
            self._seen.mark_featured(self._news)
            self._live.log_event(f"Video built: {self._long_video.name}")
            self._cp.mark_done("video", {"presenter": settings.presenter_enabled})
            self._observer.step_done(
                "video",
                file=self._long_video.name,
                presenter=settings.presenter_enabled,
            )
        else:
            logger.info(f"Step 5 (video) already done — reusing {self._long_video.name}")
            self._observer.step_skip("video")

        self._short_video = self._run_dir / "shorts.mp4"
        if rebuild_video and self._short_video.exists():
            self._short_video.unlink(missing_ok=True)
        if not self._short_video.exists():
            from src.shorts_generator import build_short
            build_short(self._script, self._long_video, self._run_dir)
        else:
            logger.info(f"Reusing cached {self._short_video.name}")

        from src.broll_fetcher import get_pexels_credits
        credits = get_pexels_credits(self._run_dir)
        if credits:
            self._script.description += (
                "\n\nVideo clips provided by Pexels:\n" + "\n".join(credits)
            )

    def _step_thumbnail(self) -> None:
        self._observer.step_start("thumbnail")
        if not self._cp.is_done("thumbnail"):
            thumb_variants = generate_thumbnail_variants(
                self._long_video, self._script.title, self._run_dir
            )
            self._thumbnail = pick_thumbnail(thumb_variants, settings.data_dir, "daily")
            _style = self._thumbnail.stem.removeprefix("thumbnail_")
            self._cp.mark_done("thumbnail", {"style": _style})
            self._observer.step_done("thumbnail", style=_style, variants=len(thumb_variants))
        else:
            logger.info("Step 6 (thumbnail) already done")
            _style = self._cp.metadata("thumbnail").get("style", "default")
            self._thumbnail = self._run_dir / f"thumbnail_{_style}.jpg"
            if not self._thumbnail.exists():
                thumb_variants = generate_thumbnail_variants(
                    self._long_video, self._script.title, self._run_dir
                )
                self._thumbnail = pick_thumbnail(thumb_variants, settings.data_dir, "daily")
            self._observer.step_skip("thumbnail", style=_style)

        self._summary["thumbnail_style"] = self._thumbnail.stem.removeprefix("thumbnail_")

    def _step_quality_gate(self) -> None:
        quality_report = run_gate(
            self._script, self._run_dir / "audio", self._feedback_history
        )
        self._summary["quality_score"] = quality_report.score
        if quality_report.warnings:
            self._summary["quality_warnings"] = quality_report.warnings

    def _step_upload_en(self) -> None:
        self._observer.step_start("upload_en")
        if not self._cp.is_done("upload_en"):
            if self._skip_upload:
                logger.info("--skip-upload specified; leaving files on disk")
                self._summary["status"] = "built_not_uploaded"
                self._cp.mark_done("upload_en", {"skipped": True})
                self._observer.step_done("upload_en", skipped=True)
            else:
                ids = publish_episode(
                    self._script, self._long_video, self._short_video,
                    thumbnail=self._thumbnail,
                    subtitle_path=self._subtitle_path,
                )
                self._summary.update(ids)
                self._summary["status"] = "published"
                self._summary["analytics"] = get_recommendations()
                if video_id := ids.get("video_id"):
                    record_usage(self._script.hook, video_id, settings.data_dir, "daily")
                    record_thumbnail_usage(self._thumbnail, video_id, settings.data_dir, "daily")
                    _ep = self._summary.get("editorial_plan", {}).get("editorial_plan") or [{}]
                    save_result(
                        video_id, self._script.hook, self._script.title, self._script.description,
                        angle=_ep[0].get("angle", ""),
                        format=_ep[0].get("format", ""),
                        platform="youtube",
                    )
                    self._live.log_event(f"Uploaded to YouTube EN: {video_id}")
                    self._live.set_metrics({"video_id": video_id, "views": 0, "ctr": 0.0})
                self._cp.mark_done("upload_en", {k: v for k, v in ids.items()})
                self._observer.step_done(
                    "upload_en",
                    **{k: v for k, v in ids.items() if isinstance(v, (str, int, float, bool))},
                )
        else:
            logger.info("Step 7 (upload_en) already done")
            meta = self._cp.metadata("upload_en")
            self._summary["status"] = (
                "built_not_uploaded" if meta.get("skipped") else "published"
            )
            if not meta.get("skipped"):
                self._summary.update(meta)
            self._observer.step_skip("upload_en")

    def _step_upload_ru(self) -> None:
        self._observer.step_start("upload_ru")
        if settings.ru_enabled and not self._cp.is_done("upload_ru"):
            ru_intro = settings.source_dir / "ai-novosti-intro.mp4"
            ru = self._run_language_variant(
                english_script=self._script,
                lang_code="ru",
                lang_name="Russian",
                voice_id=settings.ru_elevenlabs_voice_id,
                voice_model=settings.ru_elevenlabs_model,
                client_secrets=settings.ru_youtube_client_secrets,
                token_file=settings.ru_youtube_token_file,
                intro_path=ru_intro if ru_intro.exists() else None,
                outro_path=_get_shared_outro(),
                include_short=True,
            )
            self._summary["ru"] = ru
            self._cp.mark_done("upload_ru", ru)
            self._observer.step_done("upload_ru", status=ru.get("status"))
        elif settings.ru_enabled:
            logger.info("Step 8 (upload_ru) already done")
            self._summary["ru"] = self._cp.metadata("upload_ru")
            self._observer.step_skip("upload_ru")
        else:
            self._observer.step_skip("upload_ru")

    def _finalize(self) -> None:
        cost_report = self._ledger.save(self._run_dir / "cost_report.json")
        self._ledger.log_summary()
        self._summary["cost_usd"] = cost_report["total_usd"]
        self._live.set_cost({"total_usd": cost_report["total_usd"]})
        self._live.log_event(f"Pipeline complete! Cost: ${cost_report['total_usd']:.3f}")
        self._observer.finish(status="success", cost_usd=cost_report["total_usd"])
        notify_success(self._summary, "daily")
        logger.info(f"=== DONE ===  {json.dumps(self._summary, indent=2)}")

    def _run_language_variant(
        self,
        english_script: VideoScript,
        lang_code: str,
        lang_name: str,
        voice_id: str,
        voice_model: str,
        client_secrets: Path,
        token_file: Path,
        intro_path: Path | None = None,
        outro_path: Path | None = None,
        include_short: bool = True,
    ) -> dict:
        return _run_language_variant(
            english_script=english_script,
            run_dir=self._run_dir,
            lang_code=lang_code,
            lang_name=lang_name,
            voice_id=voice_id,
            voice_model=voice_model,
            client_secrets=client_secrets,
            token_file=token_file,
            skip_upload=self._skip_upload,
            intro_path=intro_path,
            outro_path=outro_path,
            include_short=include_short,
        )

    # ── Script post-processing helpers ────────────────────────────────────────

    def _humanize_script(self, script: VideoScript, persona: dict) -> VideoScript:
        """Run HumanizerAgent and apply changes back to scene narrations."""
        SCENE_MARKER = "<<<SCENE_{idx}>>>"
        marked = "\n\n".join(
            f"{SCENE_MARKER.format(idx=s.idx)}\n{s.narration}"
            for s in script.scenes
        )
        humanizer = HumanizerAgent()
        humanized = humanizer.run(
            script=marked,
            editorial_plan=self._summary["editorial_plan"],
            persona=persona,
        )
        logger.info(
            f"Humanized script with {len(humanized.changes)} changes: {humanized.changes}"
        )

        scene_texts: dict[int, str] = {}
        parts = re.split(r"<<<SCENE_(\d+)>>>", humanized.final_script)
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
            words = humanized.final_script.split()
            original_counts = [len(s.narration.split()) for s in script.scenes]
            total_original = sum(original_counts) or 1
            pos = 0
            for scene, orig_count in zip(script.scenes, original_counts):
                share = round(len(words) * orig_count / total_original)
                chunk = words[pos: pos + share]
                if chunk:
                    scene.narration = " ".join(chunk)
                pos += share

        return script

    def _apply_micro_hooks(
        self, script: VideoScript, editorial_plan, persona: dict
    ) -> VideoScript:
        """Inject micro-hooks into every scene narration."""
        micro_hook_agent = MicroHookAgent()
        scene_plan = (
            editorial_plan.editorial_plan[0]["scene_plan"]
            if editorial_plan.editorial_plan else []
        )
        for scene in script.scenes:
            hooked = micro_hook_agent.run(
                script=scene.narration,
                scene_plan=scene_plan,
                persona=persona,
            )
            if hooked.final_script:
                scene.narration = hooked.final_script
        logger.info(f"Applied micro-hooks to {len(script.scenes)} scenes")
        return script

    def _setup_logging(self) -> None:
        _setup_logging(self._run_dir)
