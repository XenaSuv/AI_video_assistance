"""Daily AI news pipeline — orchestration class.

    scrape → deduplicate → script → voice → video → thumbnail → upload
    └─ if RU_ENABLED: translate → ru-voice → reassemble → ru-thumbnail → ru-upload

Safe to re-run: cached artifacts in output/YYYY-MM-DD/ are reused.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import sys
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.budget_guard import BudgetGuard
from src.checkpoint import PipelineCheckpoint
from src.cost_tracker import reset_ledger
from src.deduplicator import SeenStories
from src.deferred_feedback import _DeferredFeedbackCollector
from src.latency_tracker import LatencyTracker
from src.live_state import LiveState
from src.media_builder import _MediaBuilder
from src.pipeline_helpers import _setup_logging as _setup_logging_helper
from src.pipeline_observer import PipelineObserver
from src.publish_step import _PublishStep
from src.quality_gate import QualityGateError
from src.scraper import NewsItem, scrape_all
from src.script_step import _ScriptStep
from src.slack_notifier import (
    notify_budget_alert,
    notify_budget_summary,
    notify_failure,
    notify_slow_steps,
    notify_success,
)
from src.viral_selector import pick_viral_news

if TYPE_CHECKING:
    from src.script_generator import VideoScript


class PipelineOrchestrator:
    """Orchestrates the full daily AI news pipeline end-to-end.

    Per-run state is stored as instance attributes so step methods share context
    without passing large argument lists. A new instance should be created for
    each pipeline run.
    """

    def run(self, dry_run: bool = False, skip_upload: bool = False) -> dict[str, Any]:
        """Execute the full pipeline. Returns a summary dict."""
        date_str = dt.date.today().isoformat()
        self._run_dir = settings.output_dir / date_str
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        logger.info(f"=== AI News Pipeline: {date_str} ===")

        self._dry_run = dry_run
        self._skip_upload = skip_upload
        self._summary: dict[str, Any] = {"date": date_str, "run_dir": str(self._run_dir)}
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
        self._feedback_history: list[Any] = []
        self._subtitle_path: Path | None = None
        self._en_intro: Path | None = None
        self._en_outro: Path | None = None
        self._long_video: Path | None = None
        self._short_video: Path | None = None
        self._thumbnail: Path | None = None
        self._auto_strategy: dict[str, Any] | None = None
        self._auto_actions: list[dict[str, Any]] = []
        self._auto_insights: list[dict[str, Any]] = []
        self._thompson_preferred_type: str | None = None

        try:
            self._step_scrape()
            if dry_run:
                logger.info("Dry run — stopping after scrape")
                return self._summary
            self._step_script()
            self._builder = _MediaBuilder(
                script=self._script,
                run_dir=self._run_dir,
                cp=self._cp,
                observer=self._observer,
                live=self._live,
                seen=self._seen,
                news=self._news,
                summary=self._summary,
                thompson_preferred_type=self._thompson_preferred_type,
            )
            self._step_voice()
            self._step_subtitles()
            self._step_video()
            self._step_thumbnail()
            self._publisher = _PublishStep(
                script=self._script,
                run_dir=self._run_dir,
                cp=self._cp,
                observer=self._observer,
                live=self._live,
                seen=self._seen,
                news=self._news,
                summary=self._summary,
                skip_upload=self._skip_upload,
                auto_strategy=self._auto_strategy,
                auto_actions=self._auto_actions,
                long_video=self._long_video,
                short_video=self._short_video,
                thumbnail=self._thumbnail,
                subtitle_path=self._subtitle_path,
                feedback_history=self._feedback_history,
            )
            self._step_quality_gate()
            self._step_upload_en()
            self._step_shorts_and_ru()
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
        # Deferred feedback: update Thompson/Sequence/Cross/Retention stores from
        # recent YouTube metrics and get the preferred hook type for this run.
        collector = _DeferredFeedbackCollector(current_run_dir=self._run_dir)
        self._thompson_preferred_type = collector.collect()
        if self._thompson_preferred_type:
            logger.info(
                f"ThompsonBandit: preferred hook type from recent history: "
                f"{self._thompson_preferred_type!r}"
            )

        step = _ScriptStep(
            news=self._news,
            history=self._history,
            run_dir=self._run_dir,
            cp=self._cp,
            observer=self._observer,
            live=self._live,
            summary=self._summary,
            thompson_preferred_type=self._thompson_preferred_type,
        )
        step.run()
        self._script = step.script
        self._feedback_history = step.feedback_history
        self._auto_strategy = step.auto_strategy
        self._auto_actions = step.auto_actions
        self._auto_insights = step.auto_insights
        self._thompson_preferred_type = step.thompson_preferred_type

    def _step_voice(self) -> None:
        self._builder.build_voice()

    def _step_subtitles(self) -> None:
        self._builder.build_subtitles()
        self._subtitle_path = self._builder.subtitle_path
        self._en_intro = self._builder.en_intro
        self._en_outro = self._builder.en_outro

    def _step_video(self) -> None:
        self._builder.build_video()
        self._long_video = self._builder.long_video
        self._short_video = self._builder.short_video

    def _step_thumbnail(self) -> None:
        self._builder.build_thumbnail()
        self._thumbnail = self._builder.thumbnail

    def _step_quality_gate(self) -> None:
        self._publisher.run_quality_gate()

    def _step_upload_en(self) -> None:
        self._publisher.upload_en()

    def _step_shorts_experiments(self) -> None:
        self._publisher.run_shorts_experiments()

    def _step_upload_ru(self) -> None:
        self._publisher.upload_ru()

    def _step_shorts_and_ru(self) -> None:
        """Run Shorts experiments and RU upload concurrently.

        Mirrors the _voice_and_images pattern from media_builder: both tasks
        are dispatched to a thread-pool executor so the event loop can overlap
        their blocking I/O (YouTube API calls, ElevenLabs TTS, FFmpeg encoding)
        without needing native async code in the uploaders themselves.
        """
        async def _run() -> None:
            loop = asyncio.get_running_loop()
            await asyncio.gather(
                loop.run_in_executor(None, self._step_shorts_experiments),
                loop.run_in_executor(None, self._step_upload_ru),
            )

        asyncio.run(_run())

    def _finalize(self) -> None:
        cost_report = self._ledger.save(self._run_dir / "cost_report.json")
        self._ledger.log_summary()
        self._summary["cost_usd"] = cost_report["total_usd"]
        self._live.set_cost({"total_usd": cost_report["total_usd"]})
        self._live.log_event(f"Pipeline complete! Cost: ${cost_report['total_usd']:.3f}")

        trace = self._observer.finish(status="success", cost_usd=cost_report["total_usd"])

        tracker = LatencyTracker(settings.data_dir)
        tracker.record(trace)

        slow = tracker.slow_steps(trace)
        if slow:
            notify_slow_steps(slow, pipeline="daily", date=self._summary.get("date", ""))

        guard = BudgetGuard(settings.output_dir, settings.daily_budget_usd, settings.monthly_budget_usd)
        budget_status = guard.check()
        notify_budget_summary(budget_status, pipeline="daily")
        if budget_status.any_exceeded:
            notify_budget_alert(budget_status, pipeline="daily")

        step_timings = tracker.step_summary(trace)
        notify_success(self._summary, "daily", step_timings=step_timings)
        logger.info(f"=== DONE ===  {json.dumps(self._summary, indent=2)}")

    def _setup_logging(self) -> None:
        _setup_logging_helper(self._run_dir)
