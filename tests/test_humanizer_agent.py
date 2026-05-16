"""Tests for HumanizerAgent — 6-pass LLM pipeline with mocked OpenAI."""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from src.humanizer_agent import HumanizerAgent
from src.shared_types import HumanizationResult


def _make_llm(responses: list[str] | None = None) -> MagicMock:
    """Build a mock LLM that cycles through provided response strings."""
    llm = MagicMock()
    if responses is None:
        responses = [f"Pass {i} rewritten script." for i in range(6)]

    def _create(*args, **kwargs):
        resp = MagicMock()
        resp.choices[0].message.content = responses[_create.call_count % len(responses)]
        resp.usage = None
        _create.call_count += 1
        return resp

    _create.call_count = 0
    llm.chat.completions.create.side_effect = _create
    return llm


def _agent(responses: list[str] | None = None) -> HumanizerAgent:
    return HumanizerAgent(llm=_make_llm(responses))


_PLAN = {"tone": "curious", "angle": "technical_breakthrough"}
_PERSONA = {"style": "skeptical", "rules": ["be direct", "no fluff"]}


# ── run() ─────────────────────────────────────────────────────────────────────

class TestHumanizerAgentRun:
    def test_returns_humanization_result(self):
        result = _agent().run("My script text.", _PLAN, _PERSONA)
        assert isinstance(result, HumanizationResult)

    def test_makes_exactly_six_llm_calls(self):
        llm = _make_llm()
        agent = HumanizerAgent(llm=llm)
        agent.run("My script text.", _PLAN, _PERSONA)
        assert llm.chat.completions.create.call_count == 6

    def test_changes_list_populated(self):
        result = _agent().run("My script text.", _PLAN, _PERSONA)
        assert len(result.changes) == 6

    def test_final_script_is_last_pass_output(self):
        outputs = [f"Pass {i}" for i in range(6)]
        result = _agent(outputs).run("original", _PLAN, _PERSONA)
        assert result.final_script == "Pass 5"

    def test_final_script_is_string(self):
        result = _agent().run("My script.", _PLAN, _PERSONA)
        assert isinstance(result.final_script, str)

    def test_empty_script_does_not_crash(self):
        result = _agent().run("", _PLAN, _PERSONA)
        assert isinstance(result, HumanizationResult)


# ── Individual pass error handling ────────────────────────────────────────────

class TestPassErrorHandling:
    def _agent_with_failing_pass(self, fail_at: int) -> tuple[HumanizerAgent, MagicMock]:
        llm = MagicMock()
        call_count = [0]

        def _create(*args, **kwargs):
            idx = call_count[0]
            call_count[0] += 1
            if idx == fail_at:
                raise RuntimeError("LLM unavailable")
            resp = MagicMock()
            resp.choices[0].message.content = f"script after pass {idx}"
            resp.usage = None
            return resp

        llm.chat.completions.create.side_effect = _create
        return HumanizerAgent(llm=llm), llm

    def test_structure_breaker_failure_falls_back_to_original(self):
        agent, _ = self._agent_with_failing_pass(0)
        result = agent.run("original text", _PLAN, _PERSONA)
        # should not raise — continues with original
        assert isinstance(result, HumanizationResult)

    def test_sentence_variator_failure_keeps_going(self):
        agent, _ = self._agent_with_failing_pass(1)
        result = agent.run("original text", _PLAN, _PERSONA)
        assert isinstance(result, HumanizationResult)

    def test_all_passes_fail_returns_original_script(self):
        llm = MagicMock()
        llm.chat.completions.create.side_effect = RuntimeError("all fail")
        agent = HumanizerAgent(llm=llm)
        result = agent.run("original text", _PLAN, _PERSONA)
        assert result.final_script == "original text"
        assert result.changes == []


# ── Individual private pass methods ──────────────────────────────────────────

class TestPrivatePasses:
    def test_structure_breaker_returns_tuple(self):
        llm = _make_llm(["rewritten"])
        agent = HumanizerAgent(llm=llm)
        script, changes = agent._structure_breaker("input script")
        assert script == "rewritten"
        assert len(changes) == 1

    def test_structure_breaker_falls_back_on_error(self):
        llm = MagicMock()
        llm.chat.completions.create.side_effect = Exception("fail")
        agent = HumanizerAgent(llm=llm)
        script, changes = agent._structure_breaker("my input")
        assert script == "my input"
        assert changes == []

    def test_voice_tuner_uses_persona_style(self):
        llm = _make_llm(["tuned script"])
        agent = HumanizerAgent(llm=llm)
        script, changes = agent._voice_tuner("input", {"style": "skeptical", "rules": []}, _PLAN)
        call_kwargs = llm.chat.completions.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "skeptical" in prompt

    def test_reaction_injector_uses_persona_style(self):
        llm = _make_llm(["injected script"])
        agent = HumanizerAgent(llm=llm)
        _, _ = agent._reaction_injector("input", {"style": "curious"})
        call_kwargs = llm.chat.completions.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "curious" in prompt
