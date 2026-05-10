"""Tests for HookPatternEngine — structural hook learning."""
from __future__ import annotations

import pytest

from src.hook_pattern_engine import (
    HookPatternEngine,
    HookPatternRecord,
    PatternScore,
    ComboScore,
    extract_features,
    _most_common,
)


# ── extract_features ──────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_returns_all_four_keys(self):
        feats = extract_features("Nobody is talking about this")
        assert set(feats.keys()) == {"pattern", "emotion", "structure", "tone"}

    def test_hidden_info_pattern(self):
        feats = extract_features("Nobody is talking about this secret")
        assert feats["pattern"] == "hidden_info"

    def test_conflict_pattern(self):
        feats = extract_features("This is the real problem and it's getting worse")
        assert feats["pattern"] == "conflict"

    def test_mistake_pattern(self):
        feats = extract_features("You're using this completely wrong")
        assert feats["pattern"] == "mistake"

    def test_warning_pattern(self):
        feats = extract_features("Warning: this could backfire badly")
        assert feats["pattern"] == "warning"

    def test_insider_pattern(self):
        feats = extract_features("What insiders know about this")
        assert feats["pattern"] == "insider"

    def test_general_fallback_when_no_signals(self):
        feats = extract_features("a quick brown fox jumps")
        assert feats["pattern"] == "general"

    def test_neutral_emotion_fallback(self):
        feats = extract_features("a quick brown fox jumps")
        assert feats["emotion"] == "neutral"

    def test_direct_callout_structure(self):
        feats = extract_features("You're doing this wrong")
        assert feats["structure"] == "direct_callout"

    def test_question_structure(self):
        feats = extract_features("Is this the real problem?")
        assert feats["structure"] == "question"

    def test_fear_emotion(self):
        feats = extract_features("This is scary and could collapse")
        assert feats["emotion"] == "fear"

    def test_intrigue_emotion(self):
        feats = extract_features("Nobody is talking about the secret")
        assert feats["emotion"] == "intrigue"

    def test_urgency_emotion(self):
        feats = extract_features("Warning: this might actually backfire")
        assert feats["emotion"] == "urgency"


# ── HookPatternRecord ─────────────────────────────────────────────────────────

class TestHookPatternRecord:
    def test_round_trip(self):
        rec = HookPatternRecord(
            hook_text="Nobody talks about this",
            hook_type="curiosity",
            pattern="hidden_info",
            emotion="intrigue",
            structure="gap",
            tone="curious",
            retention_3s=0.75,
            short_id="abc123",
        )
        rec2 = HookPatternRecord.from_dict(rec.to_dict())
        assert rec2.hook_text == rec.hook_text
        assert rec2.retention_3s == 0.75
        assert rec2.pattern == "hidden_info"

    def test_to_dict_has_recorded_at(self):
        rec = HookPatternRecord(
            hook_text="test", hook_type="conflict",
            pattern="conflict", emotion="fear",
            structure="reveal", tone="urgent",
            retention_3s=0.60,
        )
        d = rec.to_dict()
        assert "recorded_at" in d

    def test_short_id_defaults_empty(self):
        rec = HookPatternRecord(
            hook_text="test", hook_type="conflict",
            pattern="conflict", emotion="fear",
            structure="reveal", tone="urgent",
            retention_3s=0.60,
        )
        assert rec.short_id == ""


# ── HookPatternEngine.record ──────────────────────────────────────────────────

class TestRecord:
    def test_record_persists_to_file(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        engine.record("Nobody talks about this secret", "curiosity", 0.78, "s1")
        assert (tmp_path / "hook_patterns.jsonl").exists()

    def test_record_returns_correct_type(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        rec = engine.record("You're using AI completely wrong", "mistake", 0.70)
        assert isinstance(rec, HookPatternRecord)

    def test_record_extracts_correct_pattern(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        rec = engine.record("Nobody is talking about this secret", "curiosity", 0.72)
        assert rec.pattern == "hidden_info"

    def test_record_roundtrips_retention(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        rec = engine.record("test hook", "conflict", 0.6789)
        assert rec.retention_3s == round(0.6789, 4)

    def test_multiple_records_append(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        engine.record("hook one", "conflict", 0.60)
        engine.record("hook two", "curiosity", 0.70)
        engine.record("hook three", "warning", 0.80)
        lines = (tmp_path / "hook_patterns.jsonl").read_text().splitlines()
        assert len(lines) == 3


# ── HookPatternEngine.get_top_patterns ───────────────────────────────────────

class TestGetTopPatterns:
    def test_empty_when_no_data(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        assert engine.get_top_patterns() == []

    def test_returns_pattern_score_objects(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        engine.record("Nobody is talking about this", "curiosity", 0.75)
        result = engine.get_top_patterns()
        assert isinstance(result[0], PatternScore)

    def test_sorted_descending(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        engine.record("Nobody is talking about this secret", "curiosity", 0.80)
        engine.record("This is the real problem here", "conflict", 0.50)
        engine.record("Warning this could backfire", "warning", 0.65)
        top = engine.get_top_patterns()
        scores = [p.avg_score for p in top]
        assert scores == sorted(scores, reverse=True)

    def test_respects_n_limit(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        for i in range(10):
            engine.record(f"Nobody is talking about thing {i}", "curiosity", 0.70 + i * 0.01)
        assert len(engine.get_top_patterns(n=3)) <= 3

    def test_averages_multiple_records_per_pattern(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        engine.record("Nobody is talking about this", "curiosity", 0.60)
        engine.record("Nobody knows the secret", "curiosity", 0.80)
        top = engine.get_top_patterns()
        hidden_info = next((p for p in top if p.pattern == "hidden_info"), None)
        assert hidden_info is not None
        assert abs(hidden_info.avg_score - 0.70) < 0.01
        assert hidden_info.count == 2


# ── HookPatternEngine.get_best_combo ─────────────────────────────────────────

class TestGetBestCombo:
    def test_empty_when_no_data(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        assert engine.get_best_combo() == []

    def test_empty_when_fewer_than_min_points(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        engine.record("Nobody is talking about this secret", "curiosity", 0.80)
        # Only 1 data point — below _MIN_COMBO_POINTS (2)
        assert engine.get_best_combo() == []

    def test_returns_combo_when_enough_data(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        engine.record("Nobody is talking about this secret", "curiosity", 0.80)
        engine.record("Nobody is talking about the hidden info", "curiosity", 0.75)
        combos = engine.get_best_combo()
        assert len(combos) >= 1
        assert isinstance(combos[0], ComboScore)

    def test_sorted_descending(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        for _ in range(2):
            engine.record("Nobody is talking about this secret", "curiosity", 0.80)
        for _ in range(2):
            engine.record("Warning this could backfire badly", "warning", 0.55)
        combos = engine.get_best_combo()
        scores = [c.avg_score for c in combos]
        assert scores == sorted(scores, reverse=True)

    def test_combo_has_correct_fields(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        engine.record("Nobody is talking about this secret", "curiosity", 0.80)
        engine.record("Nobody is talking about the hidden info", "curiosity", 0.75)
        combo = engine.get_best_combo()[0]
        assert hasattr(combo, "pattern")
        assert hasattr(combo, "tone")
        assert hasattr(combo, "structure")
        assert hasattr(combo, "avg_score")
        assert hasattr(combo, "count")


# ── HookPatternEngine.generate_from_pattern ──────────────────────────────────

class TestGenerateFromPattern:
    def test_returns_string(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        result = engine.generate_from_pattern("hidden_info", "ChatGPT")
        assert isinstance(result, str)

    def test_contains_topic(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        result = engine.generate_from_pattern("conflict", "OpenAI")
        assert "OpenAI" in result

    def test_unknown_pattern_falls_back_to_general(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        result = engine.generate_from_pattern("nonexistent_pattern", "AI")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_different_seeds_may_vary(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        results = set()
        for seed in ["a", "b", "c", "d"]:
            results.add(engine.generate_from_pattern("hidden_info", "AI", seed=seed))
        # At least 2 different hooks generated from 4 seeds
        assert len(results) >= 2


# ── HookPatternEngine.generate_best_hooks ────────────────────────────────────

class TestGenerateBestHooks:
    def test_returns_list_of_dicts(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        hooks = engine.generate_best_hooks("ChatGPT")
        assert isinstance(hooks, list)
        for h in hooks:
            assert isinstance(h, dict)

    def test_each_hook_has_text_and_pattern(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        hooks = engine.generate_best_hooks("ChatGPT", n=5)
        for h in hooks:
            assert "text" in h
            assert "pattern" in h

    def test_respects_n_limit(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        hooks = engine.generate_best_hooks("AI", n=3)
        assert len(hooks) <= 3

    def test_learned_hooks_come_first(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        # Seed two observations for hidden_info pattern
        engine.record("Nobody is talking about this secret", "curiosity", 0.90)
        engine.record("Nobody knows the hidden info", "curiosity", 0.85)
        hooks = engine.generate_best_hooks("AI", n=5)
        sources = [h.get("source") for h in hooks]
        # At least one should be pattern_learned
        assert "pattern_learned" in sources

    def test_template_fallback_when_no_data(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        hooks = engine.generate_best_hooks("AI", n=5)
        sources = {h.get("source") for h in hooks}
        assert "template" in sources

    def test_hooks_contain_topic(self, tmp_path):
        engine = HookPatternEngine(data_dir=tmp_path)
        hooks = engine.generate_best_hooks("OpenAI", n=5)
        texts = [h["text"] for h in hooks]
        assert any("OpenAI" in t for t in texts)


# ── _most_common ──────────────────────────────────────────────────────────────

class TestMostCommon:
    def test_returns_most_frequent(self):
        assert _most_common(["a", "b", "a", "c", "a"]) == "a"

    def test_single_element(self):
        assert _most_common(["x"]) == "x"

    def test_empty_returns_empty_string(self):
        assert _most_common([]) == ""

    def test_generator_input(self):
        result = _most_common(x for x in ["a", "b", "a"])
        assert result == "a"
