"""Script generation step — editorial planning, humanization, and hook tuning.

Extracted from PipelineOrchestrator._step_script() to reduce coupling.
Instantiate once per pipeline run; call run() which populates all output attributes.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings
from src.analytics import get_recommendations
from src.auto_action_engine import AutoActionEngine
from src.cross_learning_engine import CrossLearningEngine
from src.decision_engine_v3 import DecisionEngineV3, PerformanceStore, UnifiedStrategy
from src.digest_script_generator import save_for_digest
from src.editorial_brain import EditorialBrain, EditorialPlan
from src.feedback_analyzer import FeedbackAnalyzer
from src.hook_mutation_engine import HookMutationEngine
from src.humanizer_agent import HumanizerAgent
from src.micro_hook_agent import MicroHookAgent
from src.pipeline_helpers import (
    _build_v3_context,
    _classify_hook_type,
    _load_cached_script,
    _load_retention_state,
    _unified_strategy_to_content_strategy,
)
from src.presentation_engine import PresentationEngine
from src.retention_correction_engine import (
    Correction,
    DropPoint,
    FixType,
    RetentionCorrectionEngine,
)
from src.scraper import NewsItem
from src.script_generator import VideoScript, generate_script
from src.sequence_learning_engine import SequenceLearningEngine
from src.shorts_experiment_engine import ShortsExperimentEngine

from src.pipeline_context import PipelineContext


class _ScriptStep:
    """Runs editorial planning, script generation, humanization, and hook tuning.

    Constructor args mirror what PipelineOrchestrator passes via _step_script().
    After run() returns the following attributes are populated:
        script, feedback_history, auto_strategy, auto_actions, auto_insights,
        thompson_preferred_type.
    """

    def __init__(
        self,
        news: list[NewsItem],
        history: list[str],
        ctx: PipelineContext,
        summary: dict[str, Any],
        thompson_preferred_type: str | None,
    ) -> None:
        self._news = news
        self._history = history
        self._run_dir = ctx.run_dir
        self._cp = ctx.cp
        self._observer = ctx.observer
        self._live = ctx.live
        self._summary = summary
        # Input; may be updated during run via cross-engine / v3 fallback
        self.thompson_preferred_type = thompson_preferred_type

        # Output attributes populated by run()
        self.script: VideoScript | None = None
        self.feedback_history: list[Any] = []
        self.auto_strategy: dict[str, Any] | None = None
        self.auto_actions: list[dict[str, Any]] = []
        self.auto_insights: list[dict[str, Any]] = []

    def run(self) -> None:
        feedback_analyzer = FeedbackAnalyzer()  # type: ignore[no-untyped-call]
        feedback_analyzer.collect_deferred_feedback(min_age_hours=24.0)
        self.feedback_history = feedback_analyzer.load_feedback_history()

        script_cache = self._run_dir / "script.json"
        self._observer.step_start("script")
        if not self._cp.is_done("script"):
            self._run_new(script_cache)
        else:
            self._run_cached(script_cache)

        assert self.script is not None
        self._summary["title"] = self.script.title
        self._summary["num_scenes"] = len(self.script.scenes)

    # ── fresh-run path ────────────────────────────────────────────────────────

    def _run_new(self, script_cache: Path) -> None:
        auto_action_engine = AutoActionEngine(run_type="daily")
        self.auto_strategy, self.auto_actions, self.auto_insights = (
            auto_action_engine.prepare_strategy()
        )
        self._summary["strategy"] = self.auto_strategy
        self._summary["auto_actions"] = self.auto_actions
        self._summary["insights"] = self.auto_insights

        _rce_state = _load_retention_state(settings.data_dir)
        _rce_adjustments = _rce_state.get("adjustments", {})
        _rce_corrections_raw: list[dict[str, Any]] = _rce_state.get("corrections", [])
        _predicted_risks = [
            {
                "scene_idx": c["scene_idx"],
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

        _perf_store = PerformanceStore(data_dir=settings.data_dir)
        _v3_engine = DecisionEngineV3(data_dir=settings.data_dir)
        _v3_context = _build_v3_context(
            _perf_store, _v3_engine,
            self.thompson_preferred_type, _predicted_risks,
        )
        v3_strategy: UnifiedStrategy = _v3_engine.decide(_v3_context)
        strategy = _unified_strategy_to_content_strategy(v3_strategy)

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

        _seq_engine = SequenceLearningEngine(data_dir=settings.data_dir)
        _sequence_bias = _seq_engine.get_sequence_bias(n=3)
        if _sequence_bias:
            if self.auto_strategy is None:
                self.auto_strategy = {}
            self.auto_strategy["sequence_bias"] = [list(p) for p in _sequence_bias]
            logger.info(
                f"SequenceLearningEngine: injecting {len(_sequence_bias)} "
                f"sequence bias pattern(s): {_sequence_bias}"
            )

        _cross_engine = CrossLearningEngine(data_dir=settings.data_dir)
        _cross_rec = _cross_engine.recommend(context={"topic": "AI news"})
        if _cross_rec:
            if self.auto_strategy is None:
                self.auto_strategy = {}
            self.auto_strategy["pace"]    = _cross_rec.get("pace",    v3_strategy.pace)
            self.auto_strategy["persona"] = _cross_rec.get("persona", v3_strategy.persona)
            if not self.thompson_preferred_type:
                self.thompson_preferred_type = _cross_rec.get("hook_type")
            logger.info(
                f"CrossLearningEngine: recommendation → "
                f"pace={_cross_rec.get('pace')!r}  "
                f"persona={_cross_rec.get('persona')!r}  "
                f"hook_type={_cross_rec.get('hook_type')!r}"
            )
        if not self.thompson_preferred_type and v3_strategy.packaging_style:
            self.thompson_preferred_type = v3_strategy.packaging_style
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

        _shorts_engine = ShortsExperimentEngine(data_dir=settings.data_dir)
        _shorts_best_hooks = _shorts_engine.get_best_hooks(n=5)
        if _shorts_best_hooks:
            mutated_hooks = _shorts_best_hooks + mutated_hooks
            logger.info(
                f"ShortsExperimentEngine: prepended {len(_shorts_best_hooks)} "
                f"proven hook(s) to hook_candidates"
            )
        _shorts_signal = _shorts_engine.get_strong_signal()
        if _shorts_signal and self.auto_strategy is not None:
            self.auto_strategy["hook_aggressiveness"] = min(
                1.0,
                float(self.auto_strategy.get("hook_aggressiveness", 0.5))
                + _shorts_signal,
            )
            logger.info(
                f"ShortsExperimentEngine: boosted hook_aggressiveness by "
                f"{_shorts_signal} → "
                f"{self.auto_strategy['hook_aggressiveness']:.2f}"
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
                strategy=self.auto_strategy,
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

            if _rce_corrections_raw:
                _rce_engine = RetentionCorrectionEngine(data_dir=settings.data_dir)
                _intent_to_c: dict[str, dict[str, Any]] = {}
                for _rc in _rce_corrections_raw:
                    _ri = _rc.get("intent", "explain")
                    if (
                        _ri not in _intent_to_c
                        or _rc.get("confidence", 1.0)
                        > _intent_to_c[_ri].get("confidence", 1.0)
                    ):
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

            if _rce_adjustments.get("force_avatar_intro") and script.scenes:
                script.scenes[0].scene_type = "avatar"
                logger.info(
                    "RetentionCorrectionEngine: forced avatar intro "
                    "(drop at scene 0 detected in prior video)"
                )

            _persona_str = (
                v3_strategy.persona if hasattr(v3_strategy, "persona") else "direct"
            )
            _pres_strategy: dict[str, str] = {
                "format": (
                    v3_strategy.packaging_style
                    if hasattr(v3_strategy, "packaging_style")
                    else "long"
                ),
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
                _arc_type = _arc_engine.select_arc_from_editorial({})
                script = _arc_engine.apply(script, arc_type=_arc_type)
            except Exception as _arc_exc:
                logger.debug(f"EmotionalArcEngine skipped: {_arc_exc}")

            script.save(script_cache)

        self.script = script
        save_for_digest(settings.data_dir, dt.date.today(), script_cache)
        self._live.log_event(
            f"Script ready: \"{script.title}\" ({len(script.scenes)} scenes)"
        )
        _scene_types = [s.scene_type for s in script.scenes]
        _dominant_scene_type = (
            max(set(_scene_types), key=_scene_types.count)
            if _scene_types else "image"
        )
        self._cp.mark_done("script", {
            "title": script.title,
            "num_scenes": len(script.scenes),
            "v3_mode": v3_strategy.mode,
            "v3_pace": v3_strategy.pace,
            "cross_combo": {
                "hook_type":  _classify_hook_type(script.hook),
                "scene_type": _dominant_scene_type,
                "persona":    (self.auto_strategy or {}).get("persona", "balanced"),
                "pace":       v3_strategy.pace,
                "topic":      "AI news",
            },
        })
        self._observer.step_done("script", title=script.title, num_scenes=len(script.scenes))

    # ── cached-run path ───────────────────────────────────────────────────────

    def _run_cached(self, script_cache: Path) -> None:
        logger.info("Step 2 (script) already done — loading cached script.json")
        script = _load_cached_script(script_cache)
        if script is None:
            raise RuntimeError("Checkpoint says script is done but script.json is missing")
        self.script = script
        save_for_digest(settings.data_dir, dt.date.today(), script_cache)
        self._observer.step_skip("script", title=script.title, num_scenes=len(script.scenes))

    # ── script post-processing ────────────────────────────────────────────────

    def _humanize_script(self, script: VideoScript, persona: dict[str, Any]) -> VideoScript:
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
            for scene, orig_count in zip(script.scenes, original_counts, strict=False):
                share = round(len(words) * orig_count / total_original)
                chunk = words[pos: pos + share]
                if chunk:
                    scene.narration = " ".join(chunk)
                pos += share
        return script

    def _apply_micro_hooks(
        self, script: VideoScript, editorial_plan: EditorialPlan, persona: dict[str, Any]
    ) -> VideoScript:
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

    def _apply_thompson_hook(self, script: VideoScript) -> VideoScript:
        if not self.thompson_preferred_type or not script.hook_variants:
            return script
        preferred = self.thompson_preferred_type
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
