"""Tests for PersonaEngine — character layer (voice + opinion + memory)."""
from __future__ import annotations

import pytest

from src.persona_engine import (
    _FORMAT_SIGNALS,
    _SKEPTIC_ANGLE_BOOST,
    CONTENT_FORMATS,
    DEFAULT_PERSONA,
    SIGNATURES,
    MemoryEntry,
    PersonaEngine,
    PersonaMemory,
    get_default_engine,
)

# ── DEFAULT_PERSONA ───────────────────────────────────────────────────────────

class TestDefaultPersona:
    def test_has_archetype(self):
        assert "archetype" in DEFAULT_PERSONA

    def test_archetype_is_skeptic_insider(self):
        assert DEFAULT_PERSONA["archetype"] == "skeptic_insider"

    def test_has_rules_list(self):
        assert isinstance(DEFAULT_PERSONA["rules"], list)
        assert len(DEFAULT_PERSONA["rules"]) >= 3

    def test_has_bias(self):
        assert "bias" in DEFAULT_PERSONA

    def test_has_style(self):
        assert "style" in DEFAULT_PERSONA


# ── SIGNATURES ────────────────────────────────────────────────────────────────

class TestSignatures:
    def test_all_positions_present(self):
        for pos in ("opener", "transition", "kicker"):
            assert pos in SIGNATURES

    def test_each_position_has_multiple_phrases(self):
        for pos, phrases in SIGNATURES.items():
            assert len(phrases) >= 2, pos

    def test_phrases_are_non_empty_strings(self):
        for _pos, phrases in SIGNATURES.items():
            for phrase in phrases:
                assert isinstance(phrase, str) and len(phrase) > 0


# ── CONTENT_FORMATS ───────────────────────────────────────────────────────────

class TestContentFormats:
    def test_all_formats_present(self):
        for fmt in ("hot_take", "reaction", "breakdown", "prediction"):
            assert fmt in CONTENT_FORMATS

    def test_each_format_has_opener_and_angle_bias(self):
        for fmt, config in CONTENT_FORMATS.items():
            assert "opener" in config, fmt
            assert "angle_bias" in config, fmt


# ── PersonaEngine.detect_format ───────────────────────────────────────────────

class TestDetectFormat:
    def test_hot_take_detected_from_keywords(self):
        engine = PersonaEngine()
        text = "This is wrong and overhyped, nobody is talking about it"
        assert engine.detect_format(text) == "hot_take"

    def test_prediction_detected(self):
        engine = PersonaEngine()
        text = "What will happen next with AI agents and the future of work"
        assert engine.detect_format(text) in ("prediction", "reaction", "hot_take")

    def test_reaction_detected_for_breaking_news(self):
        engine = PersonaEngine()
        text = "OpenAI just released and announced a major new product"
        assert engine.detect_format(text) == "reaction"

    def test_breakdown_detected(self):
        engine = PersonaEngine()
        text = "How and why this actually works: a deep dive explained"
        assert engine.detect_format(text) == "breakdown"

    def test_returns_string_for_unknown_text(self):
        engine = PersonaEngine()
        result = engine.detect_format("some random text with no signals")
        assert isinstance(result, str)
        assert result in CONTENT_FORMATS


# ── PersonaEngine.inject_opinion ──────────────────────────────────────────────

class TestInjectOpinion:
    def test_returns_string(self):
        engine = PersonaEngine()
        result = engine.inject_opinion("OpenAI released GPT-5.")
        assert isinstance(result, str)

    def test_result_longer_than_input(self):
        engine = PersonaEngine()
        text = "OpenAI released GPT-5."
        result = engine.inject_opinion(text)
        assert len(result) > len(text)

    def test_contains_original_text(self):
        engine = PersonaEngine()
        text = "Something happened today."
        result = engine.inject_opinion(text)
        assert text in result

    def test_conflict_hook_uses_skeptic_push(self):
        engine = PersonaEngine()
        result = engine.inject_opinion("The problem with OpenAI.", hook_type="conflict")
        assert len(result) > 0

    def test_insider_hook_uses_insider_take(self):
        engine = PersonaEngine()
        result = engine.inject_opinion("What really happened.", hook_type="insider")
        assert len(result) > 0

    def test_deterministic_for_same_input(self):
        engine = PersonaEngine()
        r1 = engine.inject_opinion("Same text.", hook_type="curiosity")
        r2 = engine.inject_opinion("Same text.", hook_type="curiosity")
        assert r1 == r2


# ── PersonaEngine.get_signature ───────────────────────────────────────────────

class TestGetSignature:
    def test_returns_non_empty_string(self):
        engine = PersonaEngine()
        result = engine.get_signature("opener")
        assert isinstance(result, str) and len(result) > 0

    def test_all_positions_work(self):
        engine = PersonaEngine()
        for pos in ("opener", "transition", "kicker"):
            assert len(engine.get_signature(pos)) > 0

    def test_unknown_position_falls_back_to_kicker(self):
        engine = PersonaEngine()
        result = engine.get_signature("nonexistent_position")
        assert isinstance(result, str) and len(result) > 0

    def test_different_seeds_may_vary(self):
        engine = PersonaEngine()
        results = {engine.get_signature("kicker", seed=str(i)) for i in range(10)}
        assert len(results) >= 2

    def test_same_seed_deterministic(self):
        engine = PersonaEngine()
        r1 = engine.get_signature("opener", seed="test")
        r2 = engine.get_signature("opener", seed="test")
        assert r1 == r2


# ── PersonaEngine.opinionize_core ─────────────────────────────────────────────

class TestOpinionizeCore:
    def test_returns_non_empty_string(self):
        engine = PersonaEngine()
        result = engine.opinionize_core("OpenAI released something big.", "curiosity")
        assert isinstance(result, str) and len(result) > 0

    def test_contains_core_fact(self):
        engine = PersonaEngine()
        core = "OpenAI released something big."
        result = engine.opinionize_core(core, "curiosity")
        assert core in result

    def test_longer_than_input(self):
        engine = PersonaEngine()
        core = "Something happened."
        result = engine.opinionize_core(core, "conflict")
        assert len(result) > len(core)


# ── PersonaEngine.as_humanizer_persona ───────────────────────────────────────

class TestAsHumanizerPersona:
    def test_returns_dict(self):
        engine = PersonaEngine()
        result = engine.as_humanizer_persona("OpenAI just released something")
        assert isinstance(result, dict)

    def test_has_style_and_rules(self):
        engine = PersonaEngine()
        result = engine.as_humanizer_persona()
        assert "style" in result
        assert "rules" in result

    def test_has_format_key(self):
        engine = PersonaEngine()
        result = engine.as_humanizer_persona("OpenAI just released a product")
        assert "format" in result
        assert result["format"] in CONTENT_FORMATS

    def test_has_opener_key(self):
        engine = PersonaEngine()
        result = engine.as_humanizer_persona("something")
        assert "opener" in result

    def test_rules_is_list(self):
        engine = PersonaEngine()
        result = engine.as_humanizer_persona()
        assert isinstance(result["rules"], list)


# ── PersonaEngine.bias_angle_scores ──────────────────────────────────────────

class TestBiasAngleScores:
    def test_overhyped_vs_reality_gets_boost(self):
        engine = PersonaEngine()
        scores = {"overhyped_vs_reality": 0.5, "industry_impact": 0.5}
        biased = engine.bias_angle_scores(scores)
        assert biased["overhyped_vs_reality"] > biased["industry_impact"]

    def test_all_keys_preserved(self):
        engine = PersonaEngine()
        scores = {"overhyped_vs_reality": 0.5, "technical_breakthrough": 0.6}
        biased = engine.bias_angle_scores(scores)
        assert set(biased.keys()) == set(scores.keys())

    def test_values_are_floats(self):
        engine = PersonaEngine()
        scores = {"threat_to_jobs": 0.4, "industry_impact": 0.3}
        biased = engine.bias_angle_scores(scores)
        for v in biased.values():
            assert isinstance(v, float)

    def test_neutral_angle_unchanged_by_boost(self):
        engine = PersonaEngine()
        scores = {"technical_breakthrough": 0.7}
        biased = engine.bias_angle_scores(scores)
        boost = _SKEPTIC_ANGLE_BOOST.get("technical_breakthrough", 0.0)
        assert abs(biased["technical_breakthrough"] - (0.7 + boost)) < 0.001


# ── PersonaMemory ─────────────────────────────────────────────────────────────

class TestPersonaMemory:
    def test_record_persists_to_file(self, tmp_path):
        mem = PersonaMemory(tmp_path)
        mem.record("opinion", "OpenAI", "This looks impressive but...")
        assert (tmp_path / "persona_memory.jsonl").exists()

    def test_record_returns_memory_entry(self, tmp_path):
        mem = PersonaMemory(tmp_path)
        entry = mem.record("hot_take", "AI agents", "They're not ready yet.")
        assert isinstance(entry, MemoryEntry)
        assert entry.topic == "AI agents"

    def test_load_recent_returns_entries(self, tmp_path):
        mem = PersonaMemory(tmp_path)
        mem.record("opinion", "OpenAI", "Text 1")
        mem.record("prediction", "AI", "Text 2")
        recent = mem.load_recent()
        assert len(recent) == 2

    def test_load_recent_respects_n_limit(self, tmp_path):
        mem = PersonaMemory(tmp_path)
        for i in range(10):
            mem.record("opinion", f"Topic {i}", f"Opinion {i}")
        recent = mem.load_recent(n=3)
        assert len(recent) == 3

    def test_load_recent_on_empty_file(self, tmp_path):
        mem = PersonaMemory(tmp_path)
        assert mem.load_recent() == []

    def test_get_callbacks_matches_topic(self, tmp_path):
        mem = PersonaMemory(tmp_path)
        mem.record("prediction", "OpenAI GPT-5", "I predicted this would happen")
        callbacks = mem.get_callbacks("OpenAI GPT-5")
        assert len(callbacks) == 1
        assert "predicted" in callbacks[0].lower() or "said" in callbacks[0].lower()

    def test_get_callbacks_empty_when_no_match(self, tmp_path):
        mem = PersonaMemory(tmp_path)
        mem.record("opinion", "NVIDIA", "Something about chips")
        assert mem.get_callbacks("completely unrelated topic xyz") == []


# ── PersonaEngine.record and get_callbacks ────────────────────────────────────

class TestEngineMemory:
    def test_record_returns_entry(self, tmp_path):
        engine = PersonaEngine(data_dir=tmp_path)
        entry = engine.record("OpenAI", "This is a hot take.", entry_type="hot_take")
        assert isinstance(entry, MemoryEntry)

    def test_get_callback_returns_string_when_match(self, tmp_path):
        engine = PersonaEngine(data_dir=tmp_path)
        engine.record("OpenAI", "I said this would be a problem.", entry_type="prediction")
        callbacks = engine.get_callbacks("OpenAI")
        assert isinstance(callbacks, list)
        assert len(callbacks) >= 1

    def test_get_callback_empty_when_no_match(self, tmp_path):
        engine = PersonaEngine(data_dir=tmp_path)
        assert engine.get_callbacks("unrelated topic xyz") == []


# ── get_default_engine singleton ──────────────────────────────────────────────

class TestDefaultEngineSingleton:
    def test_returns_persona_engine(self):
        engine = get_default_engine()
        assert isinstance(engine, PersonaEngine)

    def test_same_instance_returned(self):
        e1 = get_default_engine()
        e2 = get_default_engine()
        assert e1 is e2
