"""Tests for ContextAnalyzer.

Covers:
- ContentContext: to_dict / from_dict roundtrip
- ContentContext.to_mode_hint(): breaking/tutorial/high-novelty/high-urgency/none
- ContentContext.to_strategy_overrides(): per content_type base overrides
- to_strategy_overrides(): topic-signal refinements (is_research, mentions_openai)
- to_strategy_overrides(): urgency ≥ 0.8 boosts pace + aggressiveness
- to_strategy_overrides(): high complexity sets persona fallback
- ContextAnalyzer._detect_type(): explicit flags take priority
- ContextAnalyzer._detect_type(): keyword detection per type
- ContextAnalyzer._detect_type(): unknown → "news" default
- ContextAnalyzer._extract_topic_signals(): entity + domain detection
- ContextAnalyzer.analyze(): end-to-end, all fields populated
- ContextAnalyzer.analyze(): missing optional fields use defaults
- Orchestrator integration: context_analyzer wired in applies overrides
- Orchestrator integration: None context_analyzer leaves strategy unchanged
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.context_analyzer import ContentContext, ContextAnalyzer

# ── Helpers ────────────────────────────────────────────────────────────────────

def _analyzer() -> ContextAnalyzer:
    return ContextAnalyzer()


def _editorial(
    topic:      str   = "AI news",
    summary:    str   = "",
    novelty:    float = 0.5,
    complexity: float = 0.5,
    urgency:    float = 0.5,
    **flags,
) -> dict:
    return {
        "topic":         topic,
        "summary":       summary,
        "novelty_score": novelty,
        "complexity":    complexity,
        "urgency":       urgency,
        **flags,
    }


def _ctx(
    content_type:  str   = "news",
    novelty:       float = 0.5,
    complexity:    float = 0.5,
    urgency:       float = 0.5,
    topic_signals: dict  | None = None,
) -> ContentContext:
    return ContentContext(
        content_type  = content_type,
        topic         = "Test topic",
        novelty       = novelty,
        complexity    = complexity,
        urgency       = urgency,
        topic_signals = topic_signals or {},
    )


# ── ContentContext ─────────────────────────────────────────────────────────────

class TestContentContext:
    def test_to_dict_roundtrip(self):
        c = ContentContext(
            content_type  = "breaking",
            topic         = "OpenAI launches GPT-5",
            novelty       = 0.9,
            complexity    = 0.4,
            urgency       = 0.8,
            topic_signals = {"mentions_openai": True, "is_product_launch": True},
        )
        d  = c.to_dict()
        c2 = ContentContext.from_dict(d)
        assert c2.content_type     == "breaking"
        assert c2.topic            == "OpenAI launches GPT-5"
        assert abs(c2.novelty    - 0.9) < 1e-9
        assert abs(c2.complexity - 0.4) < 1e-9
        assert abs(c2.urgency    - 0.8) < 1e-9
        assert c2.topic_signals["mentions_openai"]   is True
        assert c2.topic_signals["is_product_launch"] is True

    def test_from_dict_defaults(self):
        c = ContentContext.from_dict({})
        assert c.content_type == "news"
        assert c.topic        == ""
        assert abs(c.novelty    - 0.5) < 1e-9
        assert abs(c.complexity - 0.5) < 1e-9
        assert abs(c.urgency    - 0.5) < 1e-9
        assert c.topic_signals == {}


# ── to_mode_hint ───────────────────────────────────────────────────────────────

class TestToModeHint:
    def test_breaking_returns_growth(self):
        assert _ctx("breaking").to_mode_hint() == "growth"

    def test_high_urgency_returns_growth(self):
        assert _ctx(urgency=0.8).to_mode_hint() == "growth"

    def test_tutorial_returns_stable(self):
        assert _ctx("tutorial").to_mode_hint() == "stable"

    def test_high_novelty_returns_growth(self):
        assert _ctx(novelty=0.8).to_mode_hint() == "growth"

    def test_news_no_signal_returns_none(self):
        assert _ctx("news").to_mode_hint() is None

    def test_review_no_signal_returns_none(self):
        assert _ctx("review").to_mode_hint() is None

    def test_urgency_exactly_at_threshold(self):
        # 0.8 → growth; 0.79 → None (for news)
        assert _ctx(urgency=0.79).to_mode_hint() is None


# ── to_strategy_overrides ─────────────────────────────────────────────────────

class TestToStrategyOverrides:
    def test_breaking_overrides(self):
        ov = _ctx("breaking").to_strategy_overrides()
        assert ov["pace"]                == "fast"
        assert ov["hook_aggressiveness"] == 1.0
        assert ov["persona"]             == "urgent"

    def test_tutorial_overrides(self):
        ov = _ctx("tutorial").to_strategy_overrides()
        assert ov["pace"]                == "slow"
        assert ov["hook_aggressiveness"] == 0.4
        assert ov["persona"]             == "teacher"

    def test_review_overrides(self):
        ov = _ctx("review").to_strategy_overrides()
        assert ov["pace"]                == "normal"
        assert ov["hook_aggressiveness"] == 0.6
        assert ov["persona"]             == "balanced"

    def test_opinion_overrides(self):
        ov = _ctx("opinion").to_strategy_overrides()
        assert ov["pace"]                == "fast"
        assert ov["hook_aggressiveness"] == 0.8
        assert ov["persona"]             == "curious"

    def test_news_no_base_overrides(self):
        ov = _ctx("news").to_strategy_overrides()
        assert "pace"                not in ov
        assert "hook_aggressiveness" not in ov

    def test_is_research_forces_slow_pace(self):
        ov = _ctx("breaking", topic_signals={"is_research": True}).to_strategy_overrides()
        assert ov["pace"] == "slow"   # overrides breaking's "fast"

    def test_mentions_openai_sets_insider_persona(self):
        ov = _ctx("news", topic_signals={"mentions_openai": True}).to_strategy_overrides()
        assert ov["persona"] == "insider"

    def test_openai_overrides_content_type_persona(self):
        ov = _ctx("tutorial", topic_signals={"mentions_openai": True}).to_strategy_overrides()
        assert ov["persona"] == "insider"   # overrides "teacher"

    def test_high_urgency_news_boosts_aggressiveness(self):
        ov = _ctx("news", urgency=0.9).to_strategy_overrides()
        assert ov["pace"] == "fast"
        assert ov["hook_aggressiveness"] >= 0.9

    def test_high_urgency_tutorial_not_boosted(self):
        ov = _ctx("tutorial", urgency=0.9).to_strategy_overrides()
        assert ov["pace"] == "slow"   # urgency override skipped for tutorial

    def test_high_complexity_sets_persona_fallback(self):
        ov = _ctx("news", complexity=0.9).to_strategy_overrides()
        assert ov.get("persona") == "clear_explainer"

    def test_high_complexity_does_not_override_existing_persona(self):
        ov = _ctx("breaking", complexity=0.9).to_strategy_overrides()
        assert ov["persona"] == "urgent"   # breaking wins, not clear_explainer


# ── _detect_type ──────────────────────────────────────────────────────────────

class TestDetectType:
    def _detect(self, topic="", summary="", **flags):
        return _analyzer()._detect_type(
            {"topic": topic, "summary": summary, **flags},
            f"{topic} {summary}".lower()
        )

    def test_explicit_is_breaking_flag(self):
        assert self._detect(topic="normal news", is_breaking=True) == "breaking"

    def test_explicit_is_tutorial_flag(self):
        assert self._detect(topic="random topic", is_tutorial=True) == "tutorial"

    def test_breaking_keyword_in_topic(self):
        assert self._detect(topic="Breaking: AI model shut down") == "breaking"

    def test_tutorial_keyword_in_topic(self):
        assert self._detect(topic="How to use GPT-4 for your business") == "tutorial"

    def test_tutorial_keyword_in_summary(self):
        assert self._detect(topic="AI tools", summary="A complete guide to using AI") == "tutorial"

    def test_review_keyword(self):
        assert self._detect(topic="GPT-4 vs Claude comparison") == "review"

    def test_opinion_keyword(self):
        assert self._detect(topic="Unpopular opinion: AI is overhyped") == "opinion"

    def test_unknown_defaults_to_news(self):
        assert self._detect(topic="AI update from last week") == "news"

    def test_explicit_flag_takes_priority_over_keyword(self):
        # is_breaking=True even though topic sounds like tutorial
        assert self._detect(topic="How to — BREAKING", is_breaking=True) == "breaking"


# ── _extract_topic_signals ────────────────────────────────────────────────────

class TestExtractTopicSignals:
    def _signals(self, text: str) -> dict:
        return _analyzer()._extract_topic_signals(text.lower())

    def test_openai_detected(self):
        assert self._signals("OpenAI launches new model")["mentions_openai"] is True

    def test_gpt_detected_as_openai(self):
        assert self._signals("GPT-4 benchmark results")["mentions_openai"] is True

    def test_google_gemini_detected(self):
        assert self._signals("Google Gemini beats GPT-4")["mentions_google"] is True

    def test_meta_llama_detected(self):
        assert self._signals("Meta releases Llama 3")["mentions_meta"] is True

    def test_research_paper_detected(self):
        assert self._signals("New research paper on alignment")["is_research"] is True

    def test_safety_detected(self):
        assert self._signals("AI safety risks for 2025")["is_safety"] is True

    def test_regulation_detected(self):
        assert self._signals("EU regulation for AI companies")["is_regulation"] is True

    def test_product_launch_detected(self):
        assert self._signals("New model release from Anthropic")["is_product_launch"] is True

    def test_no_signals_all_false(self):
        sigs = self._signals("weekly AI news roundup")
        assert not any(sigs.values())

    def test_multiple_signals_together(self):
        sigs = self._signals("OpenAI research paper on safety")
        assert sigs["mentions_openai"] is True
        assert sigs["is_research"]     is True
        assert sigs["is_safety"]       is True


# ── analyze() end-to-end ──────────────────────────────────────────────────────

class TestAnalyze:
    def test_basic_news(self):
        ctx = _analyzer().analyze(_editorial(topic="Weekly AI roundup"))
        assert ctx.content_type == "news"
        assert ctx.topic        == "Weekly AI roundup"

    def test_breaking_news(self):
        ctx = _analyzer().analyze(_editorial(topic="Breaking: OpenAI shuts down ChatGPT"))
        assert ctx.content_type == "breaking"
        assert ctx.topic_signals["mentions_openai"] is True

    def test_tutorial(self):
        ctx = _analyzer().analyze(_editorial(topic="How to fine-tune Llama 3"))
        assert ctx.content_type == "tutorial"

    def test_novelty_extracted(self):
        ctx = _analyzer().analyze(_editorial(novelty=0.9))
        assert abs(ctx.novelty - 0.9) < 1e-9

    def test_complexity_extracted(self):
        ctx = _analyzer().analyze(_editorial(complexity=0.7))
        assert abs(ctx.complexity - 0.7) < 1e-9

    def test_urgency_extracted(self):
        ctx = _analyzer().analyze(_editorial(urgency=0.8))
        assert abs(ctx.urgency - 0.8) < 1e-9

    def test_missing_optional_fields_use_defaults(self):
        ctx = _analyzer().analyze({"topic": "test"})
        assert abs(ctx.novelty    - 0.5) < 1e-9
        assert abs(ctx.complexity - 0.5) < 1e-9
        assert abs(ctx.urgency    - 0.5) < 1e-9

    def test_angle_used_as_fallback_for_topic(self):
        ctx = _analyzer().analyze({"angle": "technical_breakthrough"})
        assert ctx.topic == "technical_breakthrough"

    def test_returns_content_context_instance(self):
        ctx = _analyzer().analyze(_editorial())
        assert isinstance(ctx, ContentContext)

    def test_summary_contributes_to_detection(self):
        ctx = _analyzer().analyze(_editorial(
            topic="AI news", summary="A complete step-by-step guide to prompting"
        ))
        assert ctx.content_type == "tutorial"


# ── Orchestrator integration ───────────────────────────────────────────────────

class TestOrchestratorIntegration:
    """Verify that the orchestrator wires context_analyzer correctly."""

    def _make_orch(self, context_analyzer=None):
        import tempfile

        from src.system_orchestrator import SystemOrchestrator

        @dataclass
        class _Strategy:
            mode:                str   = "growth"
            pace:                str   = "normal"
            hook_aggressiveness: float = 0.5
            persona:             str   = "balanced"
            scene_policy:        object = None
            scene_mix:           dict   = field(default_factory=dict)

        @dataclass
        class _Packaging:
            hook:          str  = "Test hook"
            title:         str  = "Title"
            thumbnail:     dict = field(default_factory=dict)
            title_pattern: str  = "curiosity"

        @dataclass
        class _PolicyArm:
            id: str = "policy_A"

        @dataclass
        class _Pred:
            scene_idx:   int   = 0
            probability: float = 0.2

        dm = MagicMock(); dm.decide.return_value = _Strategy()
        pm = MagicMock(); pm.generate.return_value = _Packaging()
        sb = MagicMock(); sb.select_policy.return_value = _PolicyArm()
        dp = MagicMock(); dp.predict_scenes.return_value = [_Pred()]

        return SystemOrchestrator(
            decision_engine  = dm,
            packaging_engine = pm,
            scene_bandit     = sb,
            scene_engine     = MagicMock(),
            drop_predictor   = dp,
            retention_engine = MagicMock(),
            video_renderer   = lambda **kw: None,
            video_publisher  = lambda **kw: None,
            performance_store= MagicMock(),
            data_dir         = Path(tempfile.mkdtemp()),
            context_analyzer = context_analyzer,
        )

    def test_no_context_analyzer_runs_normally(self):
        orch = self._make_orch(context_analyzer=None)
        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments", return_value={}):
            result = orch.run_cycle({"scenes": [], "editorial_plan": {}, "metrics": {}})
        assert result.success is True

    def test_context_analyzer_overrides_applied(self):
        analyzer = ContextAnalyzer()
        orch     = self._make_orch(context_analyzer=analyzer)
        strategy_obj = orch.decision_engine.decide.return_value

        editorial = {
            "scenes": [], "editorial_plan": {}, "metrics": {},
            "topic": "Breaking: OpenAI shuts down all models",
            "is_breaking": True,
        }
        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments", return_value={}):
            orch.run_cycle(editorial)

        assert strategy_obj.pace                == "fast"
        assert strategy_obj.hook_aggressiveness == 1.0
        assert strategy_obj.persona             == "insider"   # OpenAI + breaking

    def test_context_written_to_run_history(self):
        import json
        analyzer = ContextAnalyzer()
        d        = Path(tempfile.mkdtemp())
        orch     = self._make_orch(context_analyzer=analyzer)
        orch._run_history_path = d / "run_history.json"
        orch._cycle_log_path   = d / "cycle_log.jsonl"

        editorial = {
            "scenes": [], "editorial_plan": {}, "metrics": {},
            "topic": "How to train a model — beginner guide",
            "is_tutorial": True,
        }
        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments", return_value={}):
            orch.run_cycle(editorial)

        records = json.loads((d / "run_history.json").read_text())
        ctx = records[0].get("context")
        assert ctx is not None
        assert ctx["content_type"] == "tutorial"

    def test_none_context_field_when_no_analyzer(self):
        import json
        d    = Path(tempfile.mkdtemp())
        orch = self._make_orch(context_analyzer=None)
        orch._run_history_path = d / "run_history.json"
        orch._cycle_log_path   = d / "cycle_log.jsonl"

        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments", return_value={}):
            orch.run_cycle({"scenes": [], "editorial_plan": {}, "metrics": {}})

        records = json.loads((d / "run_history.json").read_text())
        assert records[0].get("context") is None
