"""Tests for NarrativeIdentityEngine — temporal narrative layer."""
from __future__ import annotations

import pytest

from src.narrative_identity_engine import (
    _BELIEF_ANGLE_MAP,
    _BELIEF_SIGNALS,
    _THEME_SIGNALS,
    ARCS,
    BELIEFS,
    CALLBACKS,
    THEMES,
    NarrativeEntry,
    NarrativeIdentityEngine,
    NarrativeMemory,
    get_narrative_engine,
)

# ── BELIEFS ───────────────────────────────────────────────────────────────────

class TestBeliefs:
    def test_has_ai_hype(self):
        assert "ai_hype" in BELIEFS

    def test_has_ai_risk(self):
        assert "ai_risk" in BELIEFS

    def test_has_tools(self):
        assert "tools" in BELIEFS

    def test_has_future(self):
        assert "future" in BELIEFS

    def test_all_values_are_strings(self):
        for k, v in BELIEFS.items():
            assert isinstance(v, str) and len(v) > 0, k

    def test_belief_angle_map_covers_all_values(self):
        for v in BELIEFS.values():
            assert v in _BELIEF_ANGLE_MAP, f"belief value '{v}' missing from _BELIEF_ANGLE_MAP"


# ── THEMES ────────────────────────────────────────────────────────────────────

class TestThemes:
    def test_has_hype_vs_reality(self):
        assert "hype_vs_reality" in THEMES

    def test_has_tools_misuse(self):
        assert "tools_misuse" in THEMES

    def test_at_least_four_themes(self):
        assert len(THEMES) >= 4

    def test_all_themes_have_signals(self):
        for theme in THEMES:
            assert theme in _THEME_SIGNALS, theme

    def test_each_theme_has_signals(self):
        for theme, signals in _THEME_SIGNALS.items():
            assert len(signals) >= 2, theme


# ── ARCS ──────────────────────────────────────────────────────────────────────

class TestArcs:
    def test_has_benchmarks_lie(self):
        assert "benchmarks_lie" in ARCS

    def test_has_ai_hype_cycle(self):
        assert "ai_hype_cycle" in ARCS

    def test_each_arc_has_required_keys(self):
        for key, arc in ARCS.items():
            for field in ("label", "keywords", "status", "summary", "phrase"):
                assert field in arc, f"arc '{key}' missing '{field}'"

    def test_arc_keywords_are_sets(self):
        for key, arc in ARCS.items():
            assert isinstance(arc["keywords"], set), key

    def test_arc_phrase_non_empty(self):
        for key, arc in ARCS.items():
            assert len(arc["phrase"]) > 0, key


# ── CALLBACKS ─────────────────────────────────────────────────────────────────

class TestCallbacks:
    def test_at_least_four_callbacks(self):
        assert len(CALLBACKS) >= 4

    def test_all_strings_non_empty(self):
        for cb in CALLBACKS:
            assert isinstance(cb, str) and len(cb) > 0


# ── NarrativeMemory ───────────────────────────────────────────────────────────

class TestNarrativeMemory:
    def test_record_creates_file(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        mem.record("opinion", "OpenAI", "This is a take.")
        assert (tmp_path / "narrative_memory.jsonl").exists()

    def test_record_returns_entry(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        entry = mem.record("prediction", "AI agents", "They'll fail by Q3.")
        assert isinstance(entry, NarrativeEntry)
        assert entry.topic == "AI agents"
        assert entry.entry_type == "prediction"

    def test_load_recent_empty_on_new_store(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        assert mem.load_recent() == []

    def test_load_recent_returns_entries(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        mem.record("opinion", "OpenAI", "Take 1")
        mem.record("opinion", "Google", "Take 2")
        recent = mem.load_recent()
        assert len(recent) == 2

    def test_load_recent_respects_n(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        for i in range(10):
            mem.record("opinion", f"Topic {i}", f"Content {i}")
        assert len(mem.load_recent(n=3)) == 3

    def test_get_relevant_matches_topic(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        mem.record("prediction", "OpenAI GPT-5", "I predicted this.")
        matches = mem.get_relevant("OpenAI GPT-5")
        assert len(matches) == 1

    def test_get_relevant_empty_when_no_match(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        mem.record("opinion", "NVIDIA chips", "Something.")
        assert mem.get_relevant("completely unrelated xyz") == []

    def test_get_arc_episodes_matches_key(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        mem.record("arc_episode", "Benchmarks lie again", "Content.", arc_key="benchmarks_lie")
        episodes = mem.get_arc_episodes("benchmarks_lie")
        assert len(episodes) == 1

    def test_get_arc_episodes_empty_for_unknown_arc(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        mem.record("arc_episode", "Some topic", "Content.", arc_key="ai_hype_cycle")
        assert mem.get_arc_episodes("nonexistent_arc") == []

    def test_topic_count_increments(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        assert mem.topic_count("OpenAI") == 0
        mem.record("opinion", "OpenAI", "Take 1")
        assert mem.topic_count("OpenAI") == 1
        mem.record("opinion", "OpenAI GPT release", "Take 2")
        assert mem.topic_count("OpenAI") == 2

    def test_arc_episode_count(self, tmp_path):
        mem = NarrativeMemory(tmp_path)
        for i in range(3):
            mem.record("arc_episode", f"topic {i}", f"content {i}", arc_key="ai_hype_cycle")
        assert mem.arc_episode_count("ai_hype_cycle") == 3


# ── NarrativeEntry ────────────────────────────────────────────────────────────

class TestNarrativeEntry:
    def test_to_dict_round_trip(self):
        e = NarrativeEntry(entry_type="prediction", topic="AI", content="It will fail.")
        d = e.to_dict()
        e2 = NarrativeEntry.from_dict(d)
        assert e2.topic == "AI"
        assert e2.entry_type == "prediction"

    def test_recorded_at_is_set(self):
        e = NarrativeEntry(entry_type="opinion", topic="test", content="test")
        assert len(e.recorded_at) > 0


# ── NarrativeIdentityEngine._inject_belief ────────────────────────────────────

class TestInjectBelief:
    def test_breakthrough_triggers_ai_hype_belief(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": "OpenAI announces breakthrough model"}
        result = engine._inject_belief(editorial)
        assert result.get("belief_applied") == "ai_hype"

    def test_hallucination_triggers_tools_belief(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": "ChatGPT hallucination in legal brief"}
        result = engine._inject_belief(editorial)
        assert result.get("belief_applied") == "tools"

    def test_replace_triggers_ai_risk(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": "AI will replace software engineers"}
        result = engine._inject_belief(editorial)
        assert result.get("belief_applied") == "ai_risk"

    def test_unknown_topic_adds_no_belief(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": "random topic with no signals"}
        result = engine._inject_belief(editorial)
        assert "belief_applied" not in result

    def test_does_not_override_existing_belief_angle(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": "OpenAI breakthrough", "belief_angle": "already_set"}
        result = engine._inject_belief(editorial)
        assert result["belief_angle"] == "already_set"

    def test_apply_does_not_mutate_input(self, tmp_path):
        # _inject_belief mutates in-place; apply() must copy first
        engine = NarrativeIdentityEngine(tmp_path)
        original = {"topic": "breakthrough model"}
        engine.apply(original, "breakthrough model")
        assert "belief_applied" not in original


# ── NarrativeIdentityEngine._inject_theme ─────────────────────────────────────

class TestInjectTheme:
    def test_hype_keywords_give_hype_theme(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": "revolutionary breakthrough overhyped claims"}
        result = engine._inject_theme(editorial)
        assert result["theme"] == "hype_vs_reality"

    def test_hallucination_gives_tools_misuse(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": "hallucination error limitation wrong"}
        result = engine._inject_theme(editorial)
        assert result["theme"] == "tools_misuse"

    def test_open_source_gives_open_vs_closed(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": "llama open source vs closed proprietary"}
        result = engine._inject_theme(editorial)
        assert result["theme"] == "open_vs_closed"

    def test_fallback_theme_is_valid(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": "some random topic with no signals xyz"}
        result = engine._inject_theme(editorial)
        assert result["theme"] in THEMES

    def test_theme_key_always_present(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        result = engine._inject_theme({"topic": ""})
        assert "theme" in result


# ── NarrativeIdentityEngine._inject_arc ───────────────────────────────────────

class TestInjectArc:
    def test_benchmark_keyword_matches_arc(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": ""}
        result = engine._inject_arc(editorial, "GPT-4 benchmark evaluation shows SOTA")
        assert result.get("arc_key") == "benchmarks_lie"

    def test_jobs_keyword_matches_job_arc(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {"topic": ""}
        result = engine._inject_arc(editorial, "AI will replace workers and cause layoffs")
        assert result.get("arc_key") == "job_displacement"

    def test_arc_phrase_set_when_matched(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {}
        result = engine._inject_arc(editorial, "benchmark leaderboard results")
        assert len(result.get("arc_phrase", "")) > 0

    def test_unmatched_topic_adds_no_arc(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {}
        result = engine._inject_arc(editorial, "weather forecast sunny tuesday")
        assert "arc_key" not in result

    def test_arc_label_set_when_matched(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        editorial = {}
        result = engine._inject_arc(editorial, "AI will replace all jobs automation")
        assert len(result.get("arc_label", "")) > 0


# ── NarrativeIdentityEngine._inject_memory ────────────────────────────────────

class TestInjectMemory:
    def test_reference_set_when_past_opinion_exists(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        engine._memory.record("opinion", "OpenAI", "I thought this was wrong.")
        editorial = {}
        result = engine._inject_memory(editorial, "OpenAI new release")
        assert "reference" in result

    def test_callback_phrase_set_when_prediction_exists(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        engine._memory.record("prediction", "OpenAI", "This will fail by Q3.")
        editorial = {}
        result = engine._inject_memory(editorial, "OpenAI fails to deliver")
        assert result.get("callback_phrase") == "Remember what I said? This is it."

    def test_no_reference_when_no_history(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        result = engine._inject_memory({}, "completely unknown topic xyz")
        assert "reference" not in result


# ── NarrativeIdentityEngine.get_signature_callback ────────────────────────────

class TestGetSignatureCallback:
    def test_prediction_returns_i_called_this(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        result = engine.get_signature_callback("any topic", has_prediction=True)
        assert result == "Remember what I said? This is it."

    def test_recurring_topic_returns_here_we_go_again(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        engine._memory.record("opinion", "OpenAI", "Take 1")
        engine._memory.record("opinion", "OpenAI", "Take 2")
        result = engine.get_signature_callback("OpenAI", has_prediction=False)
        assert result == "Here we go again."

    def test_single_appearance_returns_callback_string(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        engine._memory.record("opinion", "OpenAI", "Take 1")
        result = engine.get_signature_callback("OpenAI", has_prediction=False)
        assert isinstance(result, str) and len(result) > 0

    def test_no_history_returns_empty(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        result = engine.get_signature_callback("brand new topic xyz", has_prediction=False)
        assert result == ""


# ── NarrativeIdentityEngine.apply ─────────────────────────────────────────────

class TestApply:
    def test_returns_dict(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        result = engine.apply({}, "OpenAI released GPT-5")
        assert isinstance(result, dict)

    def test_has_theme(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        result = engine.apply({}, "OpenAI breakthrough revolutionary")
        assert "theme" in result

    def test_has_belief_applied_for_relevant_topic(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        result = engine.apply({}, "OpenAI announces revolutionary breakthrough")
        assert result.get("belief_applied") == "ai_hype"

    def test_has_arc_key_for_benchmark_topic(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        result = engine.apply({}, "New model tops benchmark leaderboard")
        assert result.get("arc_key") == "benchmarks_lie"

    def test_does_not_mutate_input(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        original = {"angle": "industry_impact"}
        engine.apply(original, "test topic")
        assert "theme" not in original

    def test_topic_preserved_when_already_in_editorial(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        result = engine.apply({"topic": "existing topic"}, "different title")
        assert result["topic"] == "existing topic"

    def test_topic_filled_from_title_when_missing(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        result = engine.apply({}, "OpenAI news story")
        assert result["topic"] == "OpenAI news story"


# ── NarrativeIdentityEngine.record ────────────────────────────────────────────

class TestRecord:
    def test_returns_narrative_entry(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        entry = engine.record("OpenAI", "I said this would happen.", entry_type="prediction")
        assert isinstance(entry, NarrativeEntry)

    def test_persists_to_disk(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        engine.record("OpenAI", "Take.", theme="hype_vs_reality", arc_key="ai_hype_cycle")
        recent = engine._memory.load_recent()
        assert len(recent) == 1
        assert recent[0].theme == "hype_vs_reality"
        assert recent[0].arc_key == "ai_hype_cycle"


# ── NarrativeIdentityEngine.arc_status ────────────────────────────────────────

class TestArcStatus:
    def test_returns_all_arcs(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        status = engine.arc_status()
        for key in ARCS:
            assert key in status

    def test_episode_count_starts_at_zero(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        status = engine.arc_status()
        for key in ARCS:
            assert status[key]["episode_count"] == 0

    def test_episode_count_increments_on_record(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        engine.record("benchmarks topic", "SOTA lies.", arc_key="benchmarks_lie")
        status = engine.arc_status()
        assert status["benchmarks_lie"]["episode_count"] == 1

    def test_status_has_label_and_summary(self, tmp_path):
        engine = NarrativeIdentityEngine(tmp_path)
        status = engine.arc_status()
        for key, info in status.items():
            assert "label" in info, key
            assert "summary" in info, key


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_returns_narrative_identity_engine(self):
        engine = get_narrative_engine()
        assert isinstance(engine, NarrativeIdentityEngine)

    def test_same_instance_returned(self):
        assert get_narrative_engine() is get_narrative_engine()
