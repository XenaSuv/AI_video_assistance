"""Tests for NarrativeConflictEngine — tension and drama layer."""
from __future__ import annotations

import pytest

from src.narrative_conflict_engine import (
    _CONFLICT_TEMPLATES,
    _HIGH_THRESHOLD,
    _LOW_THRESHOLD,
    CONFLICT_TYPES,
    ConflictEntry,
    ConflictMemory,
    NarrativeConflictEngine,
    get_conflict_engine,
)

# ── Constants ─────────────────────────────────────────────────────────────────

class TestConflictTypes:
    def test_has_external(self):
        assert "external" in CONFLICT_TYPES

    def test_has_internal(self):
        assert "internal" in CONFLICT_TYPES

    def test_has_temporal(self):
        assert "temporal" in CONFLICT_TYPES

    def test_exactly_three_types(self):
        assert len(CONFLICT_TYPES) == 3


class TestConflictTemplates:
    def test_all_types_present(self):
        for t in CONFLICT_TYPES:
            assert t in _CONFLICT_TEMPLATES, t

    def test_each_type_has_three_bands(self):
        for t, bands in _CONFLICT_TEMPLATES.items():
            for band in ("high", "medium", "low"):
                assert band in bands, f"{t} missing band '{band}'"

    def test_each_band_has_templates(self):
        for t, bands in _CONFLICT_TEMPLATES.items():
            for band, templates in bands.items():
                assert len(templates) >= 2, f"{t}/{band} needs ≥2 templates"

    def test_all_templates_are_non_empty_strings(self):
        for _t, bands in _CONFLICT_TEMPLATES.items():
            for _band, templates in bands.items():
                for tmpl in templates:
                    assert isinstance(tmpl, str) and len(tmpl) > 0


class TestThresholds:
    def test_high_above_low(self):
        assert _HIGH_THRESHOLD > _LOW_THRESHOLD

    def test_thresholds_in_unit_range(self):
        assert 0.0 < _LOW_THRESHOLD < _HIGH_THRESHOLD < 1.0


# ── ConflictMemory ────────────────────────────────────────────────────────────

class TestConflictMemory:
    def test_record_creates_file(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        mem.record("external", "OpenAI", 0.7, "Everyone's wrong about this.")
        assert (tmp_path / "conflict_memory.jsonl").exists()

    def test_record_returns_entry(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        entry = mem.record("internal", "AI agents", 0.5, "Something feels off.")
        assert isinstance(entry, ConflictEntry)
        assert entry.conflict_type == "internal"
        assert entry.intensity == 0.5

    def test_load_recent_empty_on_new_store(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        assert mem.load_recent() == []

    def test_load_recent_returns_entries(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        mem.record("external", "OpenAI", 0.8, "Line 1")
        mem.record("internal", "Google", 0.4, "Line 2")
        assert len(mem.load_recent()) == 2

    def test_load_recent_respects_n(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        for i in range(10):
            mem.record("external", f"Topic {i}", 0.5, f"Line {i}")
        assert len(mem.load_recent(n=3)) == 3

    def test_get_topic_history_matches(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        mem.record("external", "OpenAI GPT-5", 0.8, "Everyone's wrong.")
        history = mem.get_topic_history("OpenAI GPT-5")
        assert len(history) == 1

    def test_get_topic_history_empty_for_no_match(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        mem.record("external", "NVIDIA chips", 0.5, "Something wrong.")
        assert mem.get_topic_history("unrelated topic xyz") == []

    def test_stats_returns_dict(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        stats = mem.stats()
        assert isinstance(stats, dict)
        assert "total" in stats
        assert "type_counts" in stats
        assert "avg_intensity" in stats
        assert "dominant_type" in stats

    def test_stats_total_zero_on_empty(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        assert mem.stats()["total"] == 0

    def test_stats_counts_types(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        mem.record("external", "T1", 0.8, "L1")
        mem.record("external", "T2", 0.7, "L2")
        mem.record("internal", "T3", 0.4, "L3")
        stats = mem.stats()
        assert stats["type_counts"]["external"] == 2
        assert stats["type_counts"]["internal"] == 1
        assert stats["dominant_type"] == "external"

    def test_stats_avg_intensity(self, tmp_path):
        mem = ConflictMemory(tmp_path)
        mem.record("external", "T1", 0.8, "L1")
        mem.record("external", "T2", 0.2, "L2")
        stats = mem.stats()
        assert abs(stats["avg_intensity"] - 0.5) < 0.01


# ── ConflictEntry ─────────────────────────────────────────────────────────────

class TestConflictEntry:
    def test_to_dict_round_trip(self):
        e = ConflictEntry(
            conflict_type="external", topic="OpenAI",
            intensity=0.75, line="Everyone's wrong."
        )
        d = e.to_dict()
        e2 = ConflictEntry.from_dict(d)
        assert e2.conflict_type == "external"
        assert e2.intensity == 0.75

    def test_recorded_at_is_set(self):
        e = ConflictEntry(conflict_type="internal", topic="t", intensity=0.5, line="l")
        assert len(e.recorded_at) > 0


# ── NarrativeConflictEngine._uncertainty_score ────────────────────────────────

class TestUncertaintyScore:
    def test_high_novelty_low_controversy_gives_high_uncertainty(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        editorial = {"novelty_score": 0.9, "controversy_score": 0.1}
        assert engine._uncertainty_score(editorial) > 0.6

    def test_low_novelty_high_controversy_gives_low_uncertainty(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        editorial = {"novelty_score": 0.1, "controversy_score": 0.9}
        assert engine._uncertainty_score(editorial) < 0.4

    def test_default_scores_give_midrange(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        score = engine._uncertainty_score({})
        assert 0.3 < score < 0.7


# ── NarrativeConflictEngine._intensity_score ──────────────────────────────────

class TestIntensityScore:
    def test_high_controversy_gives_high_intensity(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        editorial = {"controversy_score": 0.9, "belief_applied": ""}
        assert engine._intensity_score(editorial) > 0.6

    def test_belief_ai_hype_boosts_intensity(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        base  = engine._intensity_score({"controversy_score": 0.5, "belief_applied": ""})
        boosted = engine._intensity_score({"controversy_score": 0.5, "belief_applied": "ai_hype"})
        assert boosted > base

    def test_intensity_clamped_to_unit_range(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        score = engine._intensity_score({"controversy_score": 1.0, "belief_applied": "ai_hype"})
        assert 0.0 <= score <= 1.0


# ── NarrativeConflictEngine._select_conflict ──────────────────────────────────

class TestSelectConflict:
    def test_past_reference_with_moderate_uncertainty_gives_temporal(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        editorial = {
            "reference": "I predicted this: OpenAI would fail.",
            "novelty_score": 0.5,
            "controversy_score": 0.5,
        }
        conflict_type, _ = engine._select_conflict(editorial)
        assert conflict_type == "temporal"

    def test_no_past_high_uncertainty_gives_internal(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        editorial = {
            "novelty_score": 0.95,
            "controversy_score": 0.05,
        }
        conflict_type, _ = engine._select_conflict(editorial)
        assert conflict_type == "internal"

    def test_no_past_low_uncertainty_gives_external(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        editorial = {
            "novelty_score": 0.1,
            "controversy_score": 0.8,
        }
        conflict_type, _ = engine._select_conflict(editorial)
        assert conflict_type == "external"

    def test_returns_tuple_with_intensity(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        conflict_type, intensity = engine._select_conflict({})
        assert conflict_type in CONFLICT_TYPES
        assert 0.0 <= intensity <= 1.0


# ── NarrativeConflictEngine.generate_conflict_line ────────────────────────────

class TestGenerateConflictLine:
    def test_returns_non_empty_string(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        line = engine.generate_conflict_line("external", "OpenAI", 0.7)
        assert isinstance(line, str) and len(line) > 0

    def test_high_intensity_gives_strong_language(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        line = engine.generate_conflict_line("external", "OpenAI", 0.9)
        templates = _CONFLICT_TEMPLATES["external"]["high"]
        assert line in templates

    def test_low_intensity_gives_soft_language(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        line = engine.generate_conflict_line("external", "OpenAI", 0.1)
        templates = _CONFLICT_TEMPLATES["external"]["low"]
        assert line in templates

    def test_temporal_with_reference_uses_past_content(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        line = engine.generate_conflict_line(
            "temporal", "OpenAI", 0.8,
            reference="I predicted this: OpenAI would fall behind."
        )
        assert "OpenAI would fall behind" in line

    def test_temporal_medium_with_reference_uses_last_time(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        line = engine.generate_conflict_line(
            "temporal", "OpenAI", 0.5,
            reference="I said this before: this model is overrated."
        )
        assert "Last time I said" in line

    def test_temporal_no_reference_uses_template(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        line = engine.generate_conflict_line("temporal", "OpenAI", 0.7)
        templates = _CONFLICT_TEMPLATES["temporal"]["high"]
        assert line in templates

    def test_unknown_type_falls_back_to_external_medium(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        line = engine.generate_conflict_line("unknown_type", "OpenAI", 0.5)
        assert isinstance(line, str) and len(line) > 0

    def test_deterministic_for_same_inputs(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        l1 = engine.generate_conflict_line("external", "OpenAI GPT", 0.7)
        l2 = engine.generate_conflict_line("external", "OpenAI GPT", 0.7)
        assert l1 == l2


# ── NarrativeConflictEngine.apply ─────────────────────────────────────────────

class TestApply:
    def test_returns_dict(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        result = engine.apply({"topic": "OpenAI"})
        assert isinstance(result, dict)

    def test_has_conflict_type(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        result = engine.apply({"topic": "OpenAI"})
        assert result["conflict_type"] in CONFLICT_TYPES

    def test_has_conflict_line(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        result = engine.apply({"topic": "OpenAI"})
        assert isinstance(result["conflict_line"], str)
        assert len(result["conflict_line"]) > 0

    def test_has_conflict_intensity(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        result = engine.apply({"topic": "OpenAI"})
        intensity = result["conflict_intensity"]
        assert isinstance(intensity, float)
        assert 0.0 <= intensity <= 1.0

    def test_does_not_mutate_input(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        original = {"topic": "OpenAI", "hook": "test hook"}
        engine.apply(original)
        assert "conflict_type" not in original

    def test_past_reference_triggers_temporal(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        editorial = {
            "topic": "OpenAI",
            "reference": "I predicted this: OpenAI would stumble.",
            "novelty_score": 0.5,
            "controversy_score": 0.5,
        }
        result = engine.apply(editorial)
        assert result["conflict_type"] == "temporal"

    def test_high_uncertainty_triggers_internal(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        editorial = {
            "topic": "novel research paper",
            "novelty_score": 0.95,
            "controversy_score": 0.05,
        }
        result = engine.apply(editorial)
        assert result["conflict_type"] == "internal"


# ── NarrativeConflictEngine.record ────────────────────────────────────────────

class TestRecord:
    def test_returns_conflict_entry(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        entry = engine.record(
            topic="OpenAI", conflict_type="external",
            intensity=0.7, line="Everyone's wrong."
        )
        assert isinstance(entry, ConflictEntry)

    def test_persists_to_disk(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        engine.record("OpenAI", "external", 0.7, "Everyone's wrong.")
        recent = engine._memory.load_recent()
        assert len(recent) == 1
        assert recent[0].conflict_type == "external"


# ── NarrativeConflictEngine.stats ─────────────────────────────────────────────

class TestStats:
    def test_returns_dict_with_required_keys(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        stats = engine.stats()
        for key in ("total", "type_counts", "avg_intensity", "dominant_type"):
            assert key in stats

    def test_total_zero_when_empty(self, tmp_path):
        engine = NarrativeConflictEngine(tmp_path)
        assert engine.stats()["total"] == 0


# ── PresentationEngine INTENT_MAP ─────────────────────────────────────────────

class TestPresentationEngineIntentMap:
    def test_conflict_in_intent_map(self):
        from src.presentation_engine import INTENT_MAP
        assert "conflict" in INTENT_MAP

    def test_conflict_has_strong_voice(self):
        from src.presentation_engine import INTENT_MAP
        assert INTENT_MAP["conflict"]["voice"] == "strong"

    def test_conflict_has_cut_fast_visual(self):
        from src.presentation_engine import INTENT_MAP
        assert INTENT_MAP["conflict"]["visual"] == "cut_fast"

    def test_mistake_in_intent_map(self):
        from src.presentation_engine import INTENT_MAP
        assert "mistake" in INTENT_MAP

    def test_insider_in_intent_map(self):
        from src.presentation_engine import INTENT_MAP
        assert "insider" in INTENT_MAP


# ── Singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_returns_engine(self):
        assert isinstance(get_conflict_engine(), NarrativeConflictEngine)

    def test_same_instance(self):
        assert get_conflict_engine() is get_conflict_engine()
