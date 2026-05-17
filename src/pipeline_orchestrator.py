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
from src.auto_action_engine import AutoActionEngine
from src.deduplicator import SeenStories
from src.digest_script_generator import save_for_digest
from src.decision_engine_v3 import DecisionEngineV3, UnifiedStrategy, PerformanceStore
from src.shared_types import ContentStrategy
from src.feedback_analyzer import FeedbackAnalyzer
from src.hook_mutation_engine import HookMutationEngine
from src.scraper import scrape_all, NewsItem
from src.viral_selector import pick_viral_news
from src.editorial_brain import EditorialBrain
from src.humanizer_agent import HumanizerAgent
from src.micro_hook_agent import MicroHookAgent
from src.presentation_engine import PresentationEngine
from src.script_generator import Scene, VideoScript, generate_script
from src.translator import translate_script
from src.slack_notifier import notify_success, notify_failure, notify_slow_steps, notify_budget_summary, notify_budget_alert
from src.latency_tracker import LatencyTracker
from src.budget_guard import BudgetGuard
from src.checkpoint import PipelineCheckpoint
from src.quality_gate import QualityGateError
from src.cost_tracker import reset_ledger
from src.pipeline_observer import PipelineObserver
from src.live_state import LiveState
from src.youtube_analytics import get_video_metrics, get_retention_curve
from src.sequence_learning_engine import SequenceLearningEngine
from src.cross_learning_engine import CrossLearningEngine
from src.retention_correction_engine import (
    RetentionCorrectionEngine,
    Correction,
    FixType,
    DropPoint,
)
from src.shorts_experiment_engine import ShortsExperimentEngine
from src.pipeline_helpers import (
    _build_scene_map,
    _build_v3_context,
    _classify_hook_type,
    _load_cached_script,
    _load_retention_state,
    _save_retention_state,
    _setup_logging as _setup_logging_helper,
    _unified_strategy_to_content_strategy,
)
from src.analytics import get_recommendations
from src.thompson_bandit import ThompsonBandit
from src.media_builder import _MediaBuilder
from src.publish_step import _PublishStep



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
        self._auto_strategy: dict | None = None
        self._auto_actions: list[dict] = []
        self._auto_insights: list[dict] = []
        self._thompson_preferred_type: str | None = None
        self._v3_strategy: UnifiedStrategy | None = None

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
            self._step_shorts_experiments()
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

        # Update per-video Thompson bandits from YouTube metrics and get the
        # preferred hook type to guide this run's hook selection.
        self._thompson_preferred_type = self._collect_thompson_strategy()
        if self._thompson_preferred_type:
            logger.info(
                f"ThompsonBandit: preferred hook type from recent history: "
                f"{self._thompson_preferred_type!r}"
            )

        script_cache = self._run_dir / "script.json"
        self._observer.step_start("script")
        if not self._cp.is_done("script"):
            auto_action_engine = AutoActionEngine(run_type="daily")
            self._auto_strategy, self._auto_actions, self._auto_insights = (
                auto_action_engine.prepare_strategy()
            )
            self._summary["strategy"] = self._auto_strategy
            self._summary["auto_actions"] = self._auto_actions
            self._summary["insights"] = self._auto_insights

            # Load persisted retention corrections from the most recent deferred
            # analysis — feeds DecisionEngineV3 as predicted risks and is also
            # applied scene-level after script generation.
            _rce_state = _load_retention_state(settings.data_dir)
            _rce_adjustments = _rce_state.get("adjustments", {})
            _rce_corrections_raw: list[dict] = _rce_state.get("corrections", [])
            _predicted_risks = [
                {
                    "scene_idx":   c["scene_idx"],
                    "probability": min(1.0, round(
                        c["confidence"] * c["drop"]["delta"], 4
                    )),
                }
                for c in _rce_corrections_raw
            ]
            if _rce_corrections_raw:
                logger.info(
                    f"RetentionCorrectionEngine: loaded {len(_rce_corrections_raw)} "
                    f"prior correction(s) → feeding v3 as predicted_risks"
                )

            # Build context for DecisionEngineV3: channel metrics + bandit signals
            _perf_store = PerformanceStore(data_dir=settings.data_dir)
            _v3_engine = DecisionEngineV3(data_dir=settings.data_dir)
            _v3_context = _build_v3_context(
                _perf_store, _v3_engine,
                self._thompson_preferred_type, _predicted_risks,
            )
            v3_strategy: UnifiedStrategy = _v3_engine.decide(_v3_context)
            # Derive ContentStrategy for editorial_brain (angle/format weights are
            # handled by editorial_brain's own FeedbackAnalyzer, so empty is fine).
            strategy = _unified_strategy_to_content_strategy(v3_strategy)
            self._v3_strategy = v3_strategy  # kept for PerformanceStore in deferred feedback

            logger.info(
                f"Strategic decisions: mode={v3_strategy.mode}, "
                f"pace={v3_strategy.pace}, "
                f"hook_aggressiveness={v3_strategy.hook_aggressiveness:.2f}, "
                f"packaging={v3_strategy.packaging_style}, "
                f"persona={v3_strategy.persona}, "
                f"actions={v3_strategy.actions}"
            )
            self._live.set_strategy({
                "mode": v3_strategy.mode,
                "pace": v3_strategy.pace,
                "hook_aggressiveness": round(v3_strategy.hook_aggressiveness, 2),
                "packaging_style": v3_strategy.packaging_style,
                "persona": v3_strategy.persona,
                "actions": v3_strategy.actions,
                "mode_locked": v3_strategy.mode_locked,
                "exploration_rate": round(strategy.exploration_rate, 2),
                "confidence": round(strategy.confidence, 2),
            })
            # Pull the top-N retention-proven intent patterns from history and
            # inject them into the strategy dict so generate_script() can pass
            # them as ordering guidance to the GPT prompt.
            _seq_engine = SequenceLearningEngine(data_dir=settings.data_dir)
            _sequence_bias = _seq_engine.get_sequence_bias(n=3)
            if _sequence_bias:
                if self._auto_strategy is None:
                    self._auto_strategy = {}
                self._auto_strategy["sequence_bias"] = [list(p) for p in _sequence_bias]
                logger.info(
                    f"SequenceLearningEngine: injecting {len(_sequence_bias)} "
                    f"sequence bias pattern(s): {_sequence_bias}"
                )

            # Ask CrossLearningEngine for the best known combo for daily AI news.
            # The recommendation (pace, persona, hook_type) is merged into
            # auto_strategy so generate_script() and editorial_brain both see it.
            _cross_engine = CrossLearningEngine(data_dir=settings.data_dir)
            _cross_rec = _cross_engine.recommend(context={"topic": "AI news"})
            if _cross_rec:
                if self._auto_strategy is None:
                    self._auto_strategy = {}
                self._auto_strategy["pace"]    = _cross_rec.get("pace",    v3_strategy.pace)
                self._auto_strategy["persona"] = _cross_rec.get("persona", v3_strategy.persona)
                # Use cross-learning hook type when Thompson has no preference yet
                if not self._thompson_preferred_type:
                    self._thompson_preferred_type = _cross_rec.get("hook_type")
                logger.info(
                    f"CrossLearningEngine: recommendation → "
                    f"pace={_cross_rec.get('pace')!r}  "
                    f"persona={_cross_rec.get('persona')!r}  "
                    f"hook_type={_cross_rec.get('hook_type')!r}"
                )
            # Fall back to v3's packaging style as hook type signal when neither
            # Thompson nor cross-engine has accumulated enough data yet.
            if not self._thompson_preferred_type and v3_strategy.packaging_style:
                self._thompson_preferred_type = v3_strategy.packaging_style
                logger.info(
                    f"DecisionEngineV3: using packaging_style={v3_strategy.packaging_style!r} "
                    f"as initial Thompson hook type signal"
                )

            self._live.log_event(
                f"Strategy: {v3_strategy.mode} mode, "
                f"pace={v3_strategy.pace}, "
                f"hook={v3_strategy.hook_aggressiveness:.1f}"
            )

            top_recs = get_recommendations()
            top_hooks = top_recs.get("top_hooks", [])
            mutation_engine = HookMutationEngine()
            mutated = mutation_engine.run(
                top_hooks,
                context={
                    "angle": (
                        max(strategy.angle_weights, key=lambda k: strategy.angle_weights[k])
                        if strategy.angle_weights else "unknown"
                    ),
                    "persona": {"style": settings.channel_name},
                    "format": (
                        max(strategy.format_weights, key=lambda k: strategy.format_weights[k])
                        if strategy.format_weights else "unknown"
                    ),
                },
            )
            mutated_hooks = mutated.get("mutated_hooks", [])

            # Inject high-performing hooks from past Shorts experiments so the
            # editorial brain can prioritise proven openers.
            _shorts_engine = ShortsExperimentEngine(data_dir=settings.data_dir)
            _shorts_best_hooks = _shorts_engine.get_best_hooks(n=5)
            if _shorts_best_hooks:
                mutated_hooks = _shorts_best_hooks + mutated_hooks
                logger.info(
                    f"ShortsExperimentEngine: prepended {len(_shorts_best_hooks)} "
                    f"proven hook(s) to hook_candidates"
                )
            # Boost hook_aggressiveness when Shorts signal is strong enough.
            _shorts_signal = _shorts_engine.get_strong_signal()
            if _shorts_signal and self._auto_strategy is not None:
                self._auto_strategy["hook_aggressiveness"] = min(
                    1.0,
                    float(self._auto_strategy.get("hook_aggressiveness", 0.5))
                    + _shorts_signal,
                )
                logger.info(
                    f"ShortsExperimentEngine: boosted hook_aggressiveness by "
                    f"{_shorts_signal} → "
                    f"{self._auto_strategy['hook_aggressiveness']:.2f}"
                )

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
                    strategy=self._auto_strategy,
                    editorial_plan=editorial_plan,
                )
                persona = (
                    editorial_plan.editorial_plan[0]["persona"]
                    if editorial_plan.editorial_plan else {}
                )
                script = self._humanize_script(script, persona)
                script = self._apply_micro_hooks(script, editorial_plan, persona)
                script = self._apply_thompson_hook(script)
                corrections = _seq_engine.auto_correct_sequence(script.scenes)
                if corrections:
                    logger.info(
                        f"SequenceLearningEngine: corrected {corrections} "
                        f"scene intent(s) to avoid known-bad patterns"
                    )
                # Apply retention corrections from the previous video's drop
                # analysis.  Matched by scene_intent rather than position so
                # they transfer across different news topics.
                if _rce_corrections_raw:
                    _rce_engine = RetentionCorrectionEngine(data_dir=settings.data_dir)
                    # Best correction per intent (highest confidence wins)
                    _intent_to_c: dict[str, dict] = {}
                    for _rc in _rce_corrections_raw:
                        _ri = _rc.get("intent", "explain")
                        if _ri not in _intent_to_c or _rc.get("confidence", 1.0) > _intent_to_c[_ri].get("confidence", 1.0):
                            _intent_to_c[_ri] = _rc
                    _rce_matched: list[Correction] = []
                    for _scene in script.scenes:
                        _match = _intent_to_c.get(_scene.scene_intent)
                        if _match is not None:
                            _rce_matched.append(Correction(
                                scene_idx  = _scene.idx,
                                fix        = FixType(_match["fix"]),
                                drop       = DropPoint(**_match["drop"]),
                                scene_type = _match["scene_type"],
                                intent     = _match["intent"],
                                confidence = _match["confidence"],
                            ))
                    if _rce_matched:
                        _rce_engine.apply_corrections(script.scenes, _rce_matched)
                        logger.info(
                            f"RetentionCorrectionEngine: applied {len(_rce_matched)} "
                            f"intent-matched correction(s) to current script scenes"
                        )
                # If the previous video dropped at scene 0, anchor the intro
                # with an avatar scene to maximise hook retention.
                if _rce_adjustments.get("force_avatar_intro") and script.scenes:
                    script.scenes[0].scene_type = "avatar"
                    logger.info(
                        "RetentionCorrectionEngine: forced avatar intro "
                        "(drop at scene 0 detected in prior video)"
                    )

                # Apply presentation direction (voice SSML + visual + pacing)
                _persona_str = v3_strategy.persona if hasattr(v3_strategy, "persona") else "direct"
                _pres_strategy: dict[str, str] = {
                    "format": v3_strategy.packaging_style if hasattr(v3_strategy, "packaging_style") else "long",
                }
                script = PresentationEngine().apply(script, _pres_strategy, persona=_persona_str)

                try:
                    from src.emotion_engine import EmotionEngine
                    script = EmotionEngine().apply(script, conflict_intensity=0.0)
                except Exception as _emo_exc:
                    logger.debug(f"EmotionEngine skipped: {_emo_exc}")

                try:
                    from src.emotional_arc_engine import EmotionalArcEngine
                    _arc_engine = EmotionalArcEngine()
                    _arc_type   = _arc_engine.select_arc_from_editorial(
                        self._editorial if hasattr(self, "_editorial") else {}
                    )
                    script = _arc_engine.apply(script, arc_type=_arc_type)
                except Exception as _arc_exc:
                    logger.debug(f"EmotionalArcEngine skipped: {_arc_exc}")

                script.save(script_cache)

            self._script = script
            save_for_digest(settings.data_dir, dt.date.today(), script_cache)
            self._live.log_event(
                f"Script ready: \"{script.title}\" ({len(script.scenes)} scenes)"
            )
            # Compute dominant scene type for cross-learning combo storage.
            _scene_types = [s.scene_type for s in script.scenes]
            _dominant_scene_type = (
                max(set(_scene_types), key=_scene_types.count)
                if _scene_types else "image"
            )
            self._cp.mark_done("script", {
                "title": script.title,
                "num_scenes": len(script.scenes),
                "v3_mode": self._v3_strategy.mode if self._v3_strategy else "stable",
                "v3_pace": self._v3_strategy.pace if self._v3_strategy else "normal",
                # Stored so _collect_thompson_strategy() can feed CrossLearningEngine
                # with the actual combo that ran once YouTube metrics arrive.
                "cross_combo": {
                    "hook_type":  _classify_hook_type(script.hook),
                    "scene_type": _dominant_scene_type,
                    "persona":    (self._auto_strategy or {}).get("persona", "balanced"),
                    "pace":       self._v3_strategy.pace if self._v3_strategy else "normal",
                    "topic":      "AI news",
                },
            })
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
        if "thumbnail_style" in (self._builder._summary or {}):
            pass  # already updated by builder via shared summary ref

    def _step_quality_gate(self) -> None:
        self._publisher.run_quality_gate()

    def _step_upload_en(self) -> None:
        self._publisher.upload_en()

    def _step_shorts_experiments(self) -> None:
        self._publisher.run_shorts_experiments()

    def _step_upload_ru(self) -> None:
        self._publisher.upload_ru()

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

    # ── Script post-processing helpers ────────────────────────────────────────

    def _humanize_script(self, script: VideoScript, persona: dict) -> VideoScript:
        """Run HumanizerAgent with PersonaEngine-enriched persona."""
        from src.persona_engine import PersonaEngine
        pe = PersonaEngine()
        story_text = script.title + " " + (script.description or "")
        enriched_persona = {**persona, **pe.as_humanizer_persona(story_text=story_text)}

        SCENE_MARKER = "<<<SCENE_{idx}>>>"
        marked = "\n\n".join(
            f"{SCENE_MARKER.format(idx=s.idx)}\n{s.narration}"
            for s in script.scenes
        )
        humanizer = HumanizerAgent()
        humanized = humanizer.run(
            script=marked,
            editorial_plan=self._summary["editorial_plan"],
            persona=enriched_persona,
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

    # ── Thompson Bandit integration ───────────────────────────────────────────

    def _collect_thompson_strategy(self) -> str | None:
        """Update per-video Thompson bandits with fresh YouTube metrics.

        Scans the last 7 run directories that contain a thompson_state.json.
        For each, fetches current YouTube view metrics, updates the arm that
        was active, and collects the suggested preferred hook type.

        Returns the most common preferred hook type across recent videos, or
        None if no data is available (e.g. first run, no credentials).
        """
        preferred_types: list[str] = []

        run_dirs = sorted(
            [p.parent for p in settings.output_dir.glob("*/thompson_state.json")],
            key=lambda p: p.name,
            reverse=True,
        )[:7]

        for run_dir in run_dirs:
            if run_dir == self._run_dir:
                continue
            try:
                cp_path = run_dir / "checkpoint.json"
                if not cp_path.exists():
                    continue
                cp_data = json.loads(cp_path.read_text())
                video_id = (
                    cp_data.get("steps", {})
                    .get("upload_en", {})
                    .get("video_id")
                )
                if not video_id:
                    continue

                bandit = ThompsonBandit(data_dir=run_dir)
                if not bandit.get_state().arms:
                    continue

                yt_metrics = get_video_metrics(video_id)
                if yt_metrics:
                    impressions = int(yt_metrics.get("views", 0))
                    # avg_view_percentage from YouTube API is 0-100; normalise to 0-1
                    retention = float(yt_metrics.get("avg_view_percentage", 0.0)) / 100.0
                    # Use impressions as both views and potential "clicks" so the
                    # quality-adjusted formula (clicks × retention) acts as a
                    # retention-weighted success signal.
                    clicks = impressions

                    for arm in bandit.get_state().arms:
                        new_imps = max(0, impressions - arm.impressions)
                        new_clicks = max(0, clicks - arm.clicks)
                        if new_imps > 0:
                            bandit.update(
                                arm.variant_id,
                                impressions=new_imps,
                                clicks=new_clicks,
                                retention_30s=retention,
                            )
                            logger.info(
                                f"ThompsonBandit: updated arm {arm.variant_id!r} "
                                f"for {video_id} (+{new_imps} impressions, "
                                f"retention={retention:.2f})"
                            )
                            break  # only the active arm needs updating per video

                    # Feed PerformanceStore with real metrics so DecisionEngineV3's
                    # next decide() call has accurate channel-level data.
                    prior_mode = (
                        cp_data.get("steps", {})
                        .get("script", {})
                        .get("v3_mode", "stable")
                    )
                    PerformanceStore(data_dir=settings.data_dir).save(
                        video_id=video_id,
                        metrics={
                            "avg_watch_pct":  retention,
                            "hook_retention": retention,
                            "ctr":            0.0,   # not available from basic metrics
                            "drop_points":    [],
                        },
                        strategy_mode=prior_mode,
                    )
                    logger.info(
                        f"PerformanceStore: recorded metrics for {video_id} "
                        f"(mode={prior_mode!r}, retention={retention:.2f})"
                    )

                    # Feed SequenceLearningEngine with this video's scene-intent
                    # sequence and its real retention so future runs can favour
                    # orderings that historically retain viewers.
                    script_cache = run_dir / "script.json"
                    if script_cache.exists():
                        try:
                            script_data = json.loads(script_cache.read_text())
                            scene_dicts = [
                                {"intent": s.get("scene_intent", s.get("intent", "explain"))}
                                for s in script_data.get("scenes", [])
                            ]
                            SequenceLearningEngine(data_dir=settings.data_dir).store_sequence({
                                "scenes": scene_dicts,
                                "performance": {
                                    "avg_watch_pct": retention,
                                    "ctr": 0.0,
                                    "drops": [],
                                },
                            })
                        except Exception as _seq_exc:
                            logger.warning(
                                f"SequenceLearningEngine: store_sequence failed "
                                f"for {run_dir.name}: {_seq_exc}"
                            )

                    # Feed CrossLearningEngine with the full combo that ran for
                    # this video + real retention.  Combo is stored in the script
                    # checkpoint step so it's available here without re-reading
                    # the full script.
                    cross_combo = (
                        cp_data.get("steps", {})
                        .get("script", {})
                        .get("cross_combo")
                    )
                    if cross_combo:
                        try:
                            CrossLearningEngine(data_dir=settings.data_dir).learn({
                                "strategy": {
                                    "pace":    cross_combo.get("pace",    "normal"),
                                    "persona": cross_combo.get("persona", "balanced"),
                                },
                                "packaging": {
                                    "hook_type": cross_combo.get("hook_type", "curiosity"),
                                },
                                "editorial": {
                                    "topic": cross_combo.get("topic", "AI news"),
                                },
                                "scenes": [{"type": cross_combo.get("scene_type", "image")}],
                                "performance": {
                                    "avg_watch_pct": retention,
                                    "ctr": 0.0,
                                },
                            })
                            logger.info(
                                f"CrossLearningEngine: learned combo "
                                f"{cross_combo} → retention={retention:.2f}"
                            )
                        except Exception as _cx_exc:
                            logger.warning(
                                f"CrossLearningEngine: learn() failed "
                                f"for {run_dir.name}: {_cx_exc}"
                            )

                    # Analyse the full retention curve for this past video so
                    # the next run can apply scene-level corrections and get
                    # accurate predicted_risks for DecisionEngineV3.
                    try:
                        _rce_curve = get_retention_curve(video_id)
                        if _rce_curve and script_cache.exists():
                            _rce_script_data = json.loads(script_cache.read_text())
                            _rce_scene_map = _build_scene_map(
                                _rce_script_data.get("scenes", []), len(_rce_curve)
                            )
                            _rce = RetentionCorrectionEngine(data_dir=settings.data_dir)
                            _rce_found = _rce.analyze(
                                {"curve": _rce_curve}, _rce_scene_map
                            )
                            _rce_adj = _rce.suggest_strategy_adjustments(
                                _rce_found, _rce_scene_map, _rce_curve
                            )
                            _save_retention_state(settings.data_dir, _rce_found, _rce_adj)
                            logger.info(
                                f"RetentionCorrectionEngine: {len(_rce_found)} drop(s) "
                                f"for {video_id}, state persisted for next run"
                            )
                    except Exception as _rce_exc:
                        logger.warning(
                            f"RetentionCorrectionEngine: analysis failed for "
                            f"{run_dir.name}: {_rce_exc}"
                        )

                adjustments = bandit.suggest_strategy_adjustments()
                preferred = adjustments.get("preferred_variant_type")
                if preferred:
                    preferred_types.append(preferred)

            except Exception as exc:
                logger.warning(
                    f"ThompsonBandit: deferred update failed for "
                    f"{run_dir.name}: {exc}"
                )

        # Collect analytics for past Shorts experiments and learn from winners.
        try:
            _shorts_engine = ShortsExperimentEngine(data_dir=settings.data_dir)
            _new_results = _shorts_engine.collect_analytics(min_age_hours=4.0)
            if _new_results:
                _analysis = _shorts_engine.analyze()
                logger.info(
                    f"ShortsExperimentEngine: collected {_new_results} result(s), "
                    f"winners={_analysis['winners']}, "
                    f"best_hook={_analysis['best_hook_type']!r}"
                )
        except Exception as _shorts_exc:
            logger.warning(f"ShortsExperimentEngine: deferred feedback failed: {_shorts_exc}")

        if not preferred_types:
            return None
        # Most common preferred type wins
        return max(set(preferred_types), key=preferred_types.count)

    def _apply_thompson_hook(self, script: VideoScript) -> VideoScript:
        """Swap the active hook to match Thompson Bandit's preferred variant type.

        Only acts when there is a clear preference and at least one hook_variant
        matches that type.  Leaves the script unchanged if no match is found.
        """
        if not self._thompson_preferred_type or not script.hook_variants:
            return script
        preferred = self._thompson_preferred_type
        match = next(
            (h for h in script.hook_variants if _classify_hook_type(h) == preferred),
            None,
        )
        if match and match != script.hook:
            logger.info(
                f"ThompsonBandit: hook swapped to {preferred!r} type — "
                f"{match[:80]!r}"
            )
            script.hook = match
        return script

    def _setup_logging(self) -> None:
        _setup_logging_helper(self._run_dir)
