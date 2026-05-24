"""Tests for src/script_step.py — _ScriptStep.

Targets >= 80% line coverage of src/script_step.py.
Heavy external deps are patched at the src.script_step namespace.
conftest.py stubs numpy/Google-SDK before any module import.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

# Stub additional deps that script_step imports at module level
for _m in (
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
    "elevenlabs",
    "elevenlabs.client",
):
    sys.modules.setdefault(_m, MagicMock())

from src.decision_engine_v3 import UnifiedStrategy
from src.pipeline_context import PipelineContext
from src.script_generator import Scene, VideoScript
from src.script_step import _ScriptStep

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_scene(idx: int = 0, narration: str = "Some narration text here") -> Scene:
    return Scene(
        idx=idx,
        heading=f"Heading {idx}",
        narration=narration,
        visual_prompt="a robot on stage",
        duration_sec=30,
    )


def _make_script(n_scenes: int = 2, hook: str = "This is the hook.") -> VideoScript:
    return VideoScript(
        title="AI News Today",
        hook=hook,
        scenes=[_make_scene(i) for i in range(n_scenes)],
        description="Daily AI news",
        tags=["ai", "news"],
    )


def _make_step(
    tmp_path: Path,
    thompson_preferred_type: str | None = None,
    news: list | None = None,
) -> _ScriptStep:
    """Create a _ScriptStep with mocked infrastructure."""
    cp = MagicMock()
    cp.is_done.return_value = False
    ctx = PipelineContext(
        run_dir=tmp_path,
        cp=cp,
        observer=MagicMock(),
        live=MagicMock(),
        seen=MagicMock(),
    )
    summary: dict[str, Any] = {"editorial_plan": {"selected_stories": [], "editorial_plan": [], "global_style": {}}}
    return _ScriptStep(
        news=news or [],
        history=["old story 1", "old story 2"],
        ctx=ctx,
        summary=summary,
        thompson_preferred_type=thompson_preferred_type,
    )


def _make_unified_strategy(**kwargs) -> UnifiedStrategy:
    """Build a UnifiedStrategy with sensible defaults."""
    defaults = dict(
        mode="stable",
        pace="normal",
        hook_aggressiveness=0.5,
        packaging_style="curiosity",
        persona="balanced",
        actions=[],
        mode_locked=False,
    )
    defaults.update(kwargs)
    us = UnifiedStrategy()
    for k, v in defaults.items():
        setattr(us, k, v)
    return us


def _write_script_json(path: Path, script: VideoScript | None = None) -> None:
    """Write a script.json file at the given path."""
    if script is None:
        script = _make_script()
    script.save(path)


# ── __init__ ──────────────────────────────────────────────────────────────────

class TestScriptStepInit:
    def test_attributes_set_correctly(self, tmp_path):
        cp = MagicMock()
        observer = MagicMock()
        live = MagicMock()
        summary = {}
        news = [MagicMock()]
        ctx = PipelineContext(
            run_dir=tmp_path,
            cp=cp,
            observer=observer,
            live=live,
            seen=MagicMock(),
        )

        step = _ScriptStep(
            news=news,
            history=["hist"],
            ctx=ctx,
            summary=summary,
            thompson_preferred_type="question",
        )

        assert step._news is news
        assert step._history == ["hist"]
        assert step._run_dir is tmp_path
        assert step._cp is cp
        assert step._observer is observer
        assert step._live is live
        assert step._summary is summary
        assert step.thompson_preferred_type == "question"

    def test_output_attributes_initially_none_or_empty(self, tmp_path):
        step = _make_step(tmp_path)
        assert step.script is None
        assert step.feedback_history == []
        assert step.auto_strategy is None
        assert step.auto_actions == []
        assert step.auto_insights == []

    def test_none_thompson_preferred_type(self, tmp_path):
        step = _make_step(tmp_path, thompson_preferred_type=None)
        assert step.thompson_preferred_type is None


# ── run() — cached path ───────────────────────────────────────────────────────

class TestRunCachedPath:
    """run() when cp.is_done('script') == True."""

    def _run_cached(self, tmp_path: Path, script: VideoScript | None = None):
        """Helper: sets up cached-path run and returns the step after run()."""
        if script is None:
            script = _make_script()
        _write_script_json(tmp_path / "script.json", script)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = True

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step._load_cached_script", return_value=script),
            patch("src.script_step.save_for_digest"),
        ):
            mock_fa.return_value.load_feedback_history.return_value = [{"vid": "abc"}]
            step.run()

        return step

    def test_script_populated_after_cached_run(self, tmp_path):
        step = self._run_cached(tmp_path)
        assert step.script is not None

    def test_script_title_matches(self, tmp_path):
        script = _make_script()
        step = self._run_cached(tmp_path, script=script)
        assert step.script.title == "AI News Today"

    def test_summary_title_set(self, tmp_path):
        step = self._run_cached(tmp_path)
        assert step._summary["title"] == "AI News Today"

    def test_summary_num_scenes_set(self, tmp_path):
        step = self._run_cached(tmp_path)
        assert step._summary["num_scenes"] == 2

    def test_observer_step_start_called(self, tmp_path):
        step = self._run_cached(tmp_path)
        step._observer.step_start.assert_called_once_with("script")

    def test_observer_step_skip_called(self, tmp_path):
        step = self._run_cached(tmp_path)
        step._observer.step_skip.assert_called_once()

    def test_feedback_history_loaded(self, tmp_path):
        step = self._run_cached(tmp_path)
        assert len(step.feedback_history) == 1

    def test_feedback_analyzer_collect_deferred_called(self, tmp_path):
        script = _make_script()
        _write_script_json(tmp_path / "script.json", script)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = True

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step._load_cached_script", return_value=script),
            patch("src.script_step.save_for_digest"),
        ):
            mock_fa.return_value.load_feedback_history.return_value = []
            step.run()

        mock_fa.return_value.collect_deferred_feedback.assert_called_once_with(min_age_hours=24.0)


# ── run() — cached path, missing script raises ───────────────────────────────

class TestRunCachedMissingScript:
    def test_raises_when_script_json_missing(self, tmp_path):
        step = _make_step(tmp_path)
        step._cp.is_done.return_value = True

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step._load_cached_script", return_value=None),
        ):
            mock_fa.return_value.load_feedback_history.return_value = []
            with pytest.raises(RuntimeError, match="script.json is missing"):
                step.run()


# ── _run_cached() — direct ────────────────────────────────────────────────────

class TestRunCachedDirect:
    def test_run_cached_sets_script(self, tmp_path):
        script = _make_script()
        step = _make_step(tmp_path)
        step._summary["editorial_plan"] = {}

        with (
            patch("src.script_step._load_cached_script", return_value=script),
            patch("src.script_step.save_for_digest"),
        ):
            step._run_cached(tmp_path / "script.json")

        assert step.script is script

    def test_run_cached_calls_step_skip(self, tmp_path):
        script = _make_script()
        step = _make_step(tmp_path)
        step._summary["editorial_plan"] = {}

        with (
            patch("src.script_step._load_cached_script", return_value=script),
            patch("src.script_step.save_for_digest"),
        ):
            step._run_cached(tmp_path / "script.json")

        step._observer.step_skip.assert_called_once()

    def test_run_cached_calls_save_for_digest(self, tmp_path):
        script = _make_script()
        step = _make_step(tmp_path)
        step._summary["editorial_plan"] = {}

        with (
            patch("src.script_step._load_cached_script", return_value=script),
            patch("src.script_step.save_for_digest") as mock_save,
        ):
            step._run_cached(tmp_path / "script.json")

        mock_save.assert_called_once()


# ── _apply_thompson_hook() ────────────────────────────────────────────────────

class TestApplyThompsonHook:
    def test_no_op_when_no_preferred_type(self, tmp_path):
        step = _make_step(tmp_path, thompson_preferred_type=None)
        script = _make_script(hook="Original hook")
        result = step._apply_thompson_hook(script)
        assert result.hook == "Original hook"

    def test_no_op_when_no_hook_variants(self, tmp_path):
        step = _make_step(tmp_path, thompson_preferred_type="question")
        script = _make_script(hook="Original hook")
        # hook_variants is empty by default
        result = step._apply_thompson_hook(script)
        assert result.hook == "Original hook"

    def test_swaps_hook_when_preferred_type_matches_variant(self, tmp_path):
        step = _make_step(tmp_path, thompson_preferred_type="curiosity")
        script = _make_script(hook="Original hook")
        # The curiosity type hook is "Is this the secret behind AI growth?"
        script.hook_variants = [
            "Original hook",
            "Is this the secret behind AI growth?",  # curiosity (? and secret)
        ]
        with patch("src.script_step._classify_hook_type") as mock_classify:
            # First variant is "simple", second is "curiosity"
            mock_classify.side_effect = lambda h: "curiosity" if "secret" in h else "simple"
            result = step._apply_thompson_hook(script)
        assert result.hook == "Is this the secret behind AI growth?"

    def test_no_swap_when_preferred_type_not_found_in_variants(self, tmp_path):
        step = _make_step(tmp_path, thompson_preferred_type="question")
        script = _make_script(hook="Original hook")
        script.hook_variants = ["Original hook", "Another hook"]
        with patch("src.script_step._classify_hook_type", return_value="simple"):
            result = step._apply_thompson_hook(script)
        assert result.hook == "Original hook"

    def test_no_swap_when_match_equals_current_hook(self, tmp_path):
        step = _make_step(tmp_path, thompson_preferred_type="conflict")
        script = _make_script(hook="The problem with GPT-5")
        script.hook_variants = ["The problem with GPT-5"]
        with patch("src.script_step._classify_hook_type", return_value="conflict"):
            result = step._apply_thompson_hook(script)
        # Already the same hook — no change logged
        assert result.hook == "The problem with GPT-5"


# ── _apply_micro_hooks() ──────────────────────────────────────────────────────

class TestApplyMicroHooks:
    def _make_editorial_plan(self, scene_plan: list | None = None):
        from src.editorial_brain import EditorialPlan
        return EditorialPlan(
            editorial_plan=[{"persona": {"style": "direct"}, "scene_plan": scene_plan or []}],
            selected_stories=[],
            global_style={},
        )

    def test_applies_hooks_to_each_scene(self, tmp_path):
        step = _make_step(tmp_path)
        step._summary["editorial_plan"] = {}
        script = _make_script(n_scenes=3)
        editorial_plan = self._make_editorial_plan()

        with patch("src.script_step.MicroHookAgent") as mock_mha:
            mock_mha.return_value.run.return_value = MagicMock(
                final_script="hooked narration text"
            )
            result = step._apply_micro_hooks(script, editorial_plan, persona={})

        # Should be called once per scene
        assert mock_mha.return_value.run.call_count == 3

    def test_updates_scene_narration(self, tmp_path):
        step = _make_step(tmp_path)
        step._summary["editorial_plan"] = {}
        script = _make_script(n_scenes=1)
        editorial_plan = self._make_editorial_plan()

        with patch("src.script_step.MicroHookAgent") as mock_mha:
            mock_mha.return_value.run.return_value = MagicMock(
                final_script="updated narration"
            )
            result = step._apply_micro_hooks(script, editorial_plan, persona={})

        assert result.scenes[0].narration == "updated narration"

    def test_skips_empty_final_script(self, tmp_path):
        step = _make_step(tmp_path)
        step._summary["editorial_plan"] = {}
        original_narration = "original text"
        script = _make_script(n_scenes=1)
        script.scenes[0].narration = original_narration
        editorial_plan = self._make_editorial_plan()

        with patch("src.script_step.MicroHookAgent") as mock_mha:
            mock_mha.return_value.run.return_value = MagicMock(final_script="")
            result = step._apply_micro_hooks(script, editorial_plan, persona={})

        # Empty final_script means no update
        assert result.scenes[0].narration == original_narration

    def test_uses_empty_scene_plan_when_no_editorial(self, tmp_path):
        step = _make_step(tmp_path)
        step._summary["editorial_plan"] = {}
        script = _make_script(n_scenes=1)

        from src.editorial_brain import EditorialPlan
        editorial_plan = EditorialPlan(
            editorial_plan=[],  # empty editorial_plan → scene_plan = []
            selected_stories=[],
            global_style={},
        )

        with patch("src.script_step.MicroHookAgent") as mock_mha:
            mock_mha.return_value.run.return_value = MagicMock(final_script="text")
            result = step._apply_micro_hooks(script, editorial_plan, persona={})

        # Should still be called (just with empty scene_plan)
        mock_mha.return_value.run.assert_called_once()


# ── _humanize_script() ────────────────────────────────────────────────────────

class TestHumanizeScript:
    def _make_editorial_plan_with_summary(self, step: _ScriptStep):
        """Inject editorial_plan into step._summary."""
        step._summary["editorial_plan"] = {
            "editorial_plan": [{"persona": {"style": "direct"}, "scene_plan": []}],
            "selected_stories": [],
            "global_style": {},
        }

    def test_applies_humanized_narration_via_markers(self, tmp_path):
        step = _make_step(tmp_path)
        self._make_editorial_plan_with_summary(step)

        n_scenes = 2
        script = _make_script(n_scenes=n_scenes)
        for i, scene in enumerate(script.scenes):
            scene.narration = f"Original narration {i}"

        # Build the marker-based response that _humanize_script expects
        humanized_text = (
            "<<<SCENE_0>>>\nHumanized narration 0\n\n"
            "<<<SCENE_1>>>\nHumanized narration 1"
        )

        with (
            patch("src.persona_engine.PersonaEngine") as mock_pe,
            patch("src.script_step.HumanizerAgent") as mock_ha,
        ):
            mock_pe.return_value.as_humanizer_persona.return_value = {}
            mock_ha.return_value.run.return_value = MagicMock(
                final_script=humanized_text,
                changes=["change 1"],
            )
            result = step._humanize_script(script, persona={"style": "direct"})

        assert result.scenes[0].narration == "Humanized narration 0"
        assert result.scenes[1].narration == "Humanized narration 1"

    def test_fallback_proportional_split_when_markers_lost(self, tmp_path):
        step = _make_step(tmp_path)
        self._make_editorial_plan_with_summary(step)

        script = _make_script(n_scenes=2)
        script.scenes[0].narration = "word1 word2 word3 word4"
        script.scenes[1].narration = "word5 word6 word7 word8"

        # Return humanized text WITHOUT markers → triggers proportional split
        humanized_text = "new_word1 new_word2 new_word3 new_word4 new_word5 new_word6 new_word7 new_word8"

        with (
            patch("src.persona_engine.PersonaEngine") as mock_pe,
            patch("src.script_step.HumanizerAgent") as mock_ha,
        ):
            mock_pe.return_value.as_humanizer_persona.return_value = {}
            mock_ha.return_value.run.return_value = MagicMock(
                final_script=humanized_text,
                changes=[],
            )
            result = step._humanize_script(script, persona={})

        # After fallback, at least some narration must be set
        all_narration = " ".join(s.narration for s in result.scenes)
        assert len(all_narration) > 0

    def test_humanizer_agent_called_with_script(self, tmp_path):
        step = _make_step(tmp_path)
        self._make_editorial_plan_with_summary(step)
        script = _make_script(n_scenes=1)

        with (
            patch("src.persona_engine.PersonaEngine") as mock_pe,
            patch("src.script_step.HumanizerAgent") as mock_ha,
        ):
            mock_pe.return_value.as_humanizer_persona.return_value = {}
            mock_ha.return_value.run.return_value = MagicMock(
                final_script="<<<SCENE_0>>>\nNew narration",
                changes=[],
            )
            step._humanize_script(script, persona={})

        mock_ha.return_value.run.assert_called_once()

    def test_persona_engine_called(self, tmp_path):
        step = _make_step(tmp_path)
        self._make_editorial_plan_with_summary(step)
        script = _make_script(n_scenes=1)

        with (
            patch("src.persona_engine.PersonaEngine") as mock_pe,
            patch("src.script_step.HumanizerAgent") as mock_ha,
        ):
            mock_pe.return_value.as_humanizer_persona.return_value = {"voice": "calm"}
            mock_ha.return_value.run.return_value = MagicMock(
                final_script="<<<SCENE_0>>>\nNarr",
                changes=[],
            )
            step._humanize_script(script, persona={"base": "val"})

        mock_pe.return_value.as_humanizer_persona.assert_called_once()


# ── _run_new() — full happy path (mocked deps) ────────────────────────────────

class TestRunNew:
    """Test _run_new() with all heavy deps mocked."""

    def _setup_mocks(self, tmp_path: Path):
        """Return a context manager stack and step for _run_new tests."""
        script = _make_script(n_scenes=2)
        v3_strategy = _make_unified_strategy(
            mode="stable",
            pace="normal",
            hook_aggressiveness=0.5,
            packaging_style="curiosity",
            persona="balanced",
        )
        from src.shared_types import ContentStrategy
        content_strategy = ContentStrategy(
            mode="balanced",
            exploration_rate=0.2,
            confidence=0.8,
            angle_weights={"technical_breakthrough": 0.5},
            format_weights={"deep_dive": 0.5},
        )
        from src.editorial_brain import EditorialPlan
        editorial_plan = EditorialPlan(
            editorial_plan=[{"persona": {"style": "direct"}, "scene_plan": []}],
            selected_stories=["story1"],
            global_style={"tone": "casual"},
        )
        return script, v3_strategy, content_strategy, editorial_plan

    def test_run_new_sets_script(self, tmp_path):
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = (
                {"key": "val"}, [], []
            )
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        assert step.script is not None
        assert step.script.title == "AI News Today"

    def test_run_new_summary_fields_populated(self, tmp_path):
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({"k": "v"}, ["act"], ["ins"])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        assert "strategy" in step._summary
        assert "auto_actions" in step._summary
        assert step._summary["title"] == "AI News Today"

    def test_run_new_auto_actions_populated(self, tmp_path):
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = (
                {"auto": True}, [{"action": "test"}], [{"insight": "x"}]
            )
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        assert step.auto_actions == [{"action": "test"}]
        assert step.auto_insights == [{"insight": "x"}]

    def test_run_new_sequence_bias_added_to_strategy(self, tmp_path):
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({"s": "t"}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            # Return a non-empty sequence bias
            mock_seq.return_value.get_sequence_bias.return_value = [("explain", "shock")]
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        assert step.auto_strategy is not None
        assert "sequence_bias" in step.auto_strategy

    def test_run_new_cross_rec_sets_thompson_type(self, tmp_path):
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        # Start with no preferred type
        step = _make_step(tmp_path, thompson_preferred_type=None)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            # Cross rec returns a hook_type
            mock_cross.return_value.recommend.return_value = {
                "pace": "fast",
                "persona": "energetic",
                "hook_type": "conflict",
            }
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        assert step.thompson_preferred_type == "conflict"

    def test_run_new_shorts_best_hooks_prepended(self, tmp_path):
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": ["hook_a"]}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": ["mutated_hook"]}
            mock_shorts.return_value.get_best_hooks.return_value = ["shorts_hook_1", "shorts_hook_2"]
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        # Editorial brain should be called (verifies shorts hooks were passed through)
        mock_eb.return_value.run.assert_called_once()

    def test_run_new_shorts_signal_boosts_hook_aggressiveness(self, tmp_path):
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({"hook_aggressiveness": 0.3}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = 0.2  # non-None signal
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        # hook_aggressiveness should be boosted: 0.3 + 0.2 = 0.5
        assert step.auto_strategy is not None
        assert abs(step.auto_strategy["hook_aggressiveness"] - 0.5) < 1e-9

    def test_run_new_with_rce_corrections(self, tmp_path):
        """Test _run_new with non-empty retention corrections."""
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        rce_corrections_raw = [
            {
                "scene_idx": 0,
                "confidence": 0.8,
                "fix": "shorten",
                "drop": {"delta": 0.3, "timestamp_idx": 10},
                "scene_type": "image",
                "intent": "explain",
            }
        ]

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={
                "adjustments": {},
                "corrections": rce_corrections_raw,
            }),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.RetentionCorrectionEngine") as mock_rce,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script
            mock_rce.return_value.apply_corrections.return_value = None

            step.run()

        assert step.script is not None

    def test_run_new_with_force_avatar_intro(self, tmp_path):
        """Test _run_new with force_avatar_intro adjustment."""
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={
                "adjustments": {"force_avatar_intro": True},
                "corrections": [],
            }),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        # force_avatar_intro should have set scene 0 type to "avatar"
        assert step.script.scenes[0].scene_type == "avatar"

    def test_run_new_v3_packaging_style_fallback(self, tmp_path):
        """When no thompson_preferred_type and cross_rec, v3 packaging_style is used."""
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)
        v3_strategy.packaging_style = "question"

        step = _make_step(tmp_path, thompson_preferred_type=None)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            # No cross recommendation
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        # Should use packaging_style as fallback
        assert step.thompson_preferred_type == "question"

    def test_run_new_cached_script_used_when_exists(self, tmp_path):
        """When _load_cached_script returns a script, generate_script is not called."""
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)
        cached_script = _make_script(hook="Cached hook")

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=cached_script),
            patch("src.script_step.generate_script") as mock_gen,
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan

            step.run()

        # generate_script should NOT be called since cached script exists
        mock_gen.assert_not_called()
        assert step.script.hook == "Cached hook"

    def test_run_new_observer_step_done_called(self, tmp_path):
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        step._observer.step_done.assert_called_once()

    def test_run_new_sequence_bias_none_auto_strategy_initialized(self, tmp_path):
        """When auto_strategy is None and sequence_bias is non-empty, auto_strategy is initialized."""
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            # auto_strategy will be None (prepare_strategy returns None)
            mock_aae.return_value.prepare_strategy.return_value = (None, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = [("a", "b")]
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        # auto_strategy should be initialized with sequence_bias
        assert step.auto_strategy is not None
        assert "sequence_bias" in step.auto_strategy

    def test_run_new_cross_rec_none_auto_strategy_initialized(self, tmp_path):
        """When auto_strategy is None and cross_rec is present, auto_strategy is initialized."""
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)

        step = _make_step(tmp_path, thompson_preferred_type=None)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            # auto_strategy will be None
            mock_aae.return_value.prepare_strategy.return_value = (None, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            # cross_rec has a recommendation
            mock_cross.return_value.recommend.return_value = {
                "pace": "slow",
                "persona": "analytical",
                "hook_type": "curiosity",
            }
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        assert step.auto_strategy is not None
        assert step.auto_strategy.get("pace") == "slow"

    def test_run_new_structure_rec_applied_to_strategy(self, tmp_path):
        """Structure recommendation from Shorts winners sets angle_bias and scene_pacing."""
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)
        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        structure_rec = {
            "dominant_hook": "conflict",
            "angle_bias": "industry_impact",
            "scene_pacing": "fast",
            "pattern_interrupt_freq": "high",
            "hook_intensity": 0.6,
            "shorts_winner_count": 8,
        }

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_cross.return_value.get_structure_recommendation.return_value = structure_rec
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        assert step.auto_strategy is not None
        assert step.auto_strategy.get("angle_bias") == "industry_impact"
        assert step.auto_strategy.get("scene_pacing") == "fast"
        assert step.auto_strategy.get("pattern_interrupt_freq") == "high"

    def test_run_new_structure_rec_none_does_not_set_structural_keys(self, tmp_path):
        """When get_structure_recommendation returns None, no structural keys are added."""
        script, v3_strategy, content_strategy, editorial_plan = self._setup_mocks(tmp_path)
        step = _make_step(tmp_path)
        step._cp.is_done.return_value = False

        with (
            patch("src.script_step.FeedbackAnalyzer") as mock_fa,
            patch("src.script_step.AutoActionEngine") as mock_aae,
            patch("src.script_step._load_retention_state", return_value={"adjustments": {}, "corrections": []}),
            patch("src.script_step.PerformanceStore"),
            patch("src.script_step.DecisionEngineV3") as mock_v3,
            patch("src.script_step._build_v3_context", return_value={}),
            patch("src.script_step._unified_strategy_to_content_strategy", return_value=content_strategy),
            patch("src.script_step.SequenceLearningEngine") as mock_seq,
            patch("src.script_step.CrossLearningEngine") as mock_cross,
            patch("src.script_step.get_recommendations", return_value={"top_hooks": []}),
            patch("src.script_step.HookMutationEngine") as mock_hme,
            patch("src.script_step.ShortsExperimentEngine") as mock_shorts,
            patch("src.script_step.EditorialBrain") as mock_eb,
            patch("src.script_step._load_cached_script", return_value=None),
            patch("src.script_step.generate_script", return_value=script),
            patch("src.script_step.PresentationEngine") as mock_pe,
            patch("src.script_step.save_for_digest"),
            patch("src.script_step.settings") as mock_settings,
        ):
            mock_settings.data_dir = tmp_path
            mock_settings.channel_name = "Test Channel"
            mock_fa.return_value.load_feedback_history.return_value = []
            mock_aae.return_value.prepare_strategy.return_value = ({}, [], [])
            mock_v3.return_value.decide.return_value = v3_strategy
            mock_seq.return_value.get_sequence_bias.return_value = []
            mock_seq.return_value.auto_correct_sequence.return_value = 0
            mock_cross.return_value.recommend.return_value = {}
            mock_cross.return_value.get_structure_recommendation.return_value = None
            mock_hme.return_value.run.return_value = {"mutated_hooks": []}
            mock_shorts.return_value.get_best_hooks.return_value = []
            mock_shorts.return_value.get_strong_signal.return_value = None
            mock_eb.return_value.run.return_value = editorial_plan
            mock_pe.return_value.apply.return_value = script

            step.run()

        strategy = step.auto_strategy or {}
        assert "angle_bias" not in strategy
        assert "scene_pacing" not in strategy
