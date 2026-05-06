"""Tests for src/scene_variety_engine.py"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.scene_variety_engine import (
    SCENE_INTENTS,
    SCENE_TYPES,
    SceneVarietyEngine,
)
from src.script_generator import Scene


# ── helpers ───────────────────────────────────────────────────────────────────

def _scene(idx: int, heading: str = "Heading", narration: str = "Normal text.", **kwargs) -> Scene:
    return Scene(idx=idx, heading=heading, narration=narration, visual_prompt="test", **kwargs)


def _plan(fmt: str = "quick_hit", angle: str = "technical_breakthrough") -> MagicMock:
    p = MagicMock()
    p.editorial_plan = [{"format": fmt, "angle": angle}]
    return p


# ── assign_scene_types ────────────────────────────────────────────────────────

class TestAssignSceneTypes:
    def test_returns_same_list(self):
        scenes = [_scene(i) for i in range(3)]
        result = SceneVarietyEngine().assign_scene_types(scenes)
        assert result is scenes

    def test_all_scenes_get_valid_type(self):
        scenes = [_scene(i) for i in range(6)]
        SceneVarietyEngine().assign_scene_types(scenes)
        for s in scenes:
            assert s.scene_type in SCENE_TYPES

    def test_all_scenes_get_valid_intent(self):
        scenes = [_scene(i) for i in range(6)]
        SceneVarietyEngine().assign_scene_types(scenes)
        for s in scenes:
            assert s.scene_intent in SCENE_INTENTS

    def test_empty_list_no_error(self):
        SceneVarietyEngine().assign_scene_types([])

    def test_single_scene_no_error(self):
        scenes = [_scene(0)]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[0].scene_type in SCENE_TYPES

    def test_accepts_none_plan(self):
        scenes = [_scene(i) for i in range(3)]
        result = SceneVarietyEngine().assign_scene_types(scenes, editorial_plan=None)
        assert len(result) == 3

    def test_accepts_mock_plan(self):
        scenes = [_scene(i) for i in range(3)]
        result = SceneVarietyEngine().assign_scene_types(scenes, editorial_plan=_plan())
        assert len(result) == 3


# ── _decide_type ──────────────────────────────────────────────────────────────

class TestDecideType:
    def test_first_scene_is_image(self):
        scenes = [_scene(0, narration="Revenue grew 300%")]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[0].scene_type == "image"

    def test_existing_infographic_data_honoured(self):
        scenes = [
            _scene(0),
            _scene(1, infographic_data={"type": "stat_card", "value": "42%"}),
        ]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "infographic"

    def test_percent_triggers_infographic(self):
        scenes = [_scene(0), _scene(1, narration="Accuracy improved by 15%")]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "infographic"

    def test_million_triggers_infographic(self):
        scenes = [_scene(0), _scene(1, narration="The company raised 500 million in funding")]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "infographic"

    def test_billion_triggers_infographic(self):
        scenes = [_scene(0), _scene(1, narration="Market cap hit 1.2 billion dollars")]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "infographic"

    def test_explain_keyword_triggers_diagram(self):
        scenes = [
            _scene(0),
            _scene(1, narration="Here is how the architecture works under the hood"),
        ]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "diagram"

    def test_process_keyword_triggers_diagram(self):
        scenes = [_scene(0), _scene(1, narration="The training process has three steps")]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "diagram"

    def test_shock_keyword_triggers_text_overlay(self):
        scenes = [
            _scene(0),
            _scene(1, narration="Nobody expected this secretly released feature"),
        ]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "text_overlay"

    def test_hidden_keyword_triggers_text_overlay(self):
        scenes = [_scene(0), _scene(1, narration="The hidden truth that nobody mentions")]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "text_overlay"

    def test_hot_take_format_adds_cutaway_on_even_index(self):
        # Scenes with generic narration in hot_take format: even non-zero indices → cutaway
        scenes = [_scene(i, narration="just a normal sentence here") for i in range(6)]
        SceneVarietyEngine().assign_scene_types(scenes, editorial_plan=_plan(fmt="hot_take"))
        # scene 2 and 4 should be cutaway (even non-zero, generic text)
        assert scenes[2].scene_type == "cutaway" or scenes[4].scene_type == "cutaway"

    def test_generic_text_falls_back_to_image(self):
        # Scene 0 gets infographic_data so it becomes "infographic", not "image".
        # Scene 1 has generic narration → "image" (no clash with scene 0).
        scenes = [
            _scene(0, infographic_data={"type": "stat_card"}),
            _scene(1, narration="Plain sentence with no strong signal here at all"),
        ]
        SceneVarietyEngine().assign_scene_types(scenes)
        # scene 0 = infographic, scene 1 = image — no consecutive clash
        assert scenes[1].scene_type == "image"

    def test_infographic_data_beats_shock_keyword(self):
        scenes = [
            _scene(0),
            _scene(1, narration="Nobody expected this", infographic_data={"type": "bar_chart"}),
        ]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "infographic"

    def test_infographic_data_beats_explain_keyword(self):
        scenes = [
            _scene(0),
            _scene(1, narration="Here is how the process works", infographic_data={"type": "timeline"}),
        ]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "infographic"

    def test_heading_text_contributes_to_decision(self):
        # Shock keyword in heading (not narration) should trigger text_overlay
        scenes = [
            _scene(0),
            _scene(1, heading="Nobody expected this", narration="Standard content"),
        ]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_type == "text_overlay"


# ── _enforce_variety ──────────────────────────────────────────────────────────

class TestEnforceVariety:
    def test_no_consecutive_same_type_after_assign(self):
        scenes = [_scene(i) for i in range(8)]
        SceneVarietyEngine().assign_scene_types(scenes)
        for i in range(1, len(scenes)):
            prev, curr = scenes[i - 1].scene_type, scenes[i].scene_type
            if curr != "cutaway":
                assert curr != prev, (
                    f"Consecutive same type at [{i-1}]{prev} → [{i}]{curr}"
                )

    def test_enforce_variety_directly(self):
        scenes = [_scene(i) for i in range(4)]
        engine = SceneVarietyEngine()
        for s in scenes:
            s.scene_type = "image"
        engine._enforce_variety(scenes)
        assert scenes[1].scene_type == "cutaway"
        assert scenes[3].scene_type == "cutaway"

    def test_first_scene_never_gets_replaced(self):
        scenes = [_scene(i) for i in range(3)]
        engine = SceneVarietyEngine()
        for s in scenes:
            s.scene_type = "image"
        engine._enforce_variety(scenes)
        # Scene 0 is the reference — it cannot be touched by enforce_variety
        assert scenes[0].scene_type == "image"

    def test_type_changed_to_cutaway_on_override(self):
        # _enforce_variety changes scene_type to "cutaway" but preserves intent
        scenes = [_scene(0), _scene(1)]
        engine = SceneVarietyEngine()
        scenes[0].scene_type = "diagram"
        scenes[1].scene_type = "diagram"
        scenes[1].scene_intent = "explain"
        engine._enforce_variety(scenes)
        assert scenes[1].scene_type == "cutaway"
        assert scenes[1].scene_intent == "explain"  # intent unchanged

    def test_already_cutaway_not_doubled(self):
        # If two adjacent scenes are already "cutaway", enforce_variety skips them
        scenes = [_scene(0), _scene(1)]
        engine = SceneVarietyEngine()
        scenes[0].scene_type = "cutaway"
        scenes[1].scene_type = "cutaway"
        engine._enforce_variety(scenes)
        # Both remain cutaway (no infinite recursion or incorrect replacement)
        assert scenes[0].scene_type == "cutaway"
        assert scenes[1].scene_type == "cutaway"


# ── _decide_intent ────────────────────────────────────────────────────────────

class TestDecideIntent:
    def test_infographic_scene_gets_data_intent(self):
        scenes = [_scene(0), _scene(1, narration="Revenue grew 200%")]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_intent == "data"

    def test_diagram_scene_gets_explain_intent(self):
        scenes = [
            _scene(0),
            _scene(1, narration="Here is how the mechanism works step by step"),
        ]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_intent == "explain"

    def test_text_overlay_scene_gets_shock_intent(self):
        scenes = [_scene(0), _scene(1, narration="Nobody saw this coming secretly")]
        SceneVarietyEngine().assign_scene_types(scenes)
        assert scenes[1].scene_intent == "shock"

    def test_angle_overrides_type_intent(self):
        # threat_to_jobs angle → shock intent even for image-type scenes
        scenes = [_scene(0), _scene(1, narration="plain text")]
        SceneVarietyEngine().assign_scene_types(
            scenes, editorial_plan=_plan(fmt="quick_hit", angle="threat_to_jobs")
        )
        assert scenes[1].scene_intent == "shock"

    def test_industry_impact_angle_gives_data_intent(self):
        scenes = [_scene(0), _scene(1, narration="plain text")]
        SceneVarietyEngine().assign_scene_types(
            scenes, editorial_plan=_plan(angle="industry_impact")
        )
        assert scenes[1].scene_intent == "data"

    def test_what_this_means_angle_gives_reaction_intent(self):
        scenes = [_scene(0), _scene(1, narration="plain text")]
        SceneVarietyEngine().assign_scene_types(
            scenes, editorial_plan=_plan(angle="what_this_means_for_you")
        )
        assert scenes[1].scene_intent == "reaction"


# ── editorial plan parsing ────────────────────────────────────────────────────

class TestEditorialPlanParsing:
    def test_malformed_plan_falls_back_gracefully(self):
        bad_plan = MagicMock()
        bad_plan.editorial_plan = None  # will raise TypeError on indexing
        scenes = [_scene(i) for i in range(3)]
        # Must not raise
        SceneVarietyEngine().assign_scene_types(scenes, editorial_plan=bad_plan)
        for s in scenes:
            assert s.scene_type in SCENE_TYPES

    def test_empty_editorial_plan_list(self):
        plan = MagicMock()
        plan.editorial_plan = []
        scenes = [_scene(i) for i in range(3)]
        SceneVarietyEngine().assign_scene_types(scenes, editorial_plan=plan)
        for s in scenes:
            assert s.scene_type in SCENE_TYPES

    def test_unknown_angle_uses_type_intent(self):
        plan = _plan(angle="some_unknown_angle_xyz")
        scenes = [_scene(0), _scene(1, narration="plain text")]
        SceneVarietyEngine().assign_scene_types(scenes, editorial_plan=plan)
        # Falls back to type-based intent — just check it's valid
        assert scenes[1].scene_intent in SCENE_INTENTS

    def test_angle_partial_match(self):
        # "technical_breakthrough" contains "technical_breakthrough" → should match
        plan = _plan(angle="technical_breakthrough")
        scenes = [_scene(0), _scene(1, narration="plain text")]
        SceneVarietyEngine().assign_scene_types(scenes, editorial_plan=plan)
        assert scenes[1].scene_intent == "explain"
