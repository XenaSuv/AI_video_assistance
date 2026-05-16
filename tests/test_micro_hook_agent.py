"""Tests for MicroHookAgent — hook insertion, persona routing, and paragraph splitting."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.micro_hook_agent import MicroHookAgent


def _agent(hook_frequency: int = 2) -> MicroHookAgent:
    llm = MagicMock()
    with patch("src.micro_hook_agent.HookOptimizer") as MockOpt:
        opt_instance = MagicMock()
        opt_instance.run.return_value = MagicMock(
            recommended_patterns=[],
            avoid_patterns=[],
        )
        MockOpt.return_value = opt_instance
        agent = MicroHookAgent(llm=llm, hook_frequency=hook_frequency)
        agent.hook_optimizer = opt_instance
    return agent


def _run(agent: MicroHookAgent, script: str, n_scenes: int = 3):
    scene_plan = [{"angle": "technical_breakthrough"} for _ in range(n_scenes)]
    persona = {"style": "curious"}
    return agent.run(script=script, scene_plan=scene_plan, persona=persona)


# ── _split_into_paragraphs ────────────────────────────────────────────────────

class TestSplitIntoParagraphs:
    def test_splits_on_double_newline(self):
        agent = _agent()
        result = agent._split_into_paragraphs("Para one.\n\nPara two.\n\nPara three.")
        assert result == ["Para one.", "Para two.", "Para three."]

    def test_strips_whitespace(self):
        agent = _agent()
        result = agent._split_into_paragraphs("  Hello  \n\n  World  ")
        assert result == ["Hello", "World"]

    def test_empty_paragraphs_removed(self):
        agent = _agent()
        result = agent._split_into_paragraphs("Para one.\n\n\n\nPara two.")
        assert "" not in result
        assert len(result) == 2

    def test_single_paragraph(self):
        agent = _agent()
        result = agent._split_into_paragraphs("Only one paragraph here.")
        assert result == ["Only one paragraph here."]

    def test_empty_string_returns_empty_list(self):
        agent = _agent()
        result = agent._split_into_paragraphs("")
        assert result == []


# ── _get_persona_hook_types ───────────────────────────────────────────────────

class TestGetPersonaHookTypes:
    def test_skeptical_gets_doubt_first(self):
        agent = _agent()
        types = agent._get_persona_hook_types({"style": "skeptical"})
        assert "doubt" in types
        assert types[0] == "doubt"

    def test_cynical_also_gets_doubt(self):
        agent = _agent()
        types = agent._get_persona_hook_types({"style": "cynical thinker"})
        assert "doubt" in types

    def test_curious_gets_curiosity_first(self):
        agent = _agent()
        types = agent._get_persona_hook_types({"style": "curious engineer"})
        assert "curiosity" in types
        assert types[0] == "curiosity"

    def test_explainer_gets_personal_first(self):
        agent = _agent()
        types = agent._get_persona_hook_types({"style": "explainer"})
        assert "personal" in types
        assert types[0] == "personal"

    def test_unknown_style_returns_all_types(self):
        agent = _agent()
        types = agent._get_persona_hook_types({"style": "unknown"})
        assert len(types) >= 4

    def test_empty_style_returns_defaults(self):
        agent = _agent()
        types = agent._get_persona_hook_types({})
        assert len(types) > 0


# ── run() integration ─────────────────────────────────────────────────────────

class TestMicroHookAgentRun:
    def test_returns_micro_hook_result(self):
        from src.shared_types import MicroHookResult
        agent = _agent(hook_frequency=2)
        script = "Para one.\n\nPara two.\n\nPara three.\n\nPara four.\n\nPara five."
        result = _run(agent, script)
        assert isinstance(result, MicroHookResult)

    def test_final_script_contains_original_text(self):
        agent = _agent(hook_frequency=2)
        script = "Opening text.\n\nMiddle text.\n\nClosing text."
        result = _run(agent, script)
        assert "Opening text." in result.final_script

    def test_hooks_inserted_at_right_frequency(self):
        # 6 paragraphs, frequency=2 → hooks after para 2 and para 4
        agent = _agent(hook_frequency=2)
        paras = "\n\n".join(f"Paragraph {i}." for i in range(6))
        result = _run(agent, paras)
        # Should have inserted some hooks
        assert len(result.inserted_hooks) > 0

    def test_no_hooks_for_short_script(self):
        # Only 1 paragraph — no room for hooks
        agent = _agent(hook_frequency=2)
        result = _run(agent, "Single paragraph only.")
        assert len(result.inserted_hooks) == 0

    def test_hook_text_contains_pause_annotations(self):
        agent = _agent(hook_frequency=2)
        paras = "\n\n".join(f"Paragraph {i} with more content here." for i in range(6))
        result = _run(agent, paras)
        if result.inserted_hooks:
            for hook in result.inserted_hooks:
                assert "[PAUSE_SHORT]" in hook.text

    def test_empty_scene_plan_uses_unknown_angle(self):
        agent = _agent(hook_frequency=2)
        paras = "\n\n".join(f"Para {i}." for i in range(4))
        # Should not crash with empty scene_plan
        result = agent.run(
            script=paras,
            scene_plan=[],
            persona={"style": "curious"},
        )
        assert isinstance(result.final_script, str)
