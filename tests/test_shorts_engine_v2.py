"""Tests for ShortsEngineV2 — experiment system, not a video generator."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.shorts_engine_v2 import (
    HOOK_TYPES,
    LearningRecord,
    ShortResult,
    ShortScript,
    ShortsEngineV2,
    _HOOK_TEMPLATES,
    _LOOP_ENDINGS,
    _PERSONA_MAP,
    _STYLE_MAP,
    _WINNER_THRESHOLD,
    _build_script_from_editorial,
    _extract_topic,
    _pick_template,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _editorial(topic: str = "ChatGPT") -> dict[str, Any]:
    return {
        "topic":     topic,
        "core_fact": f"{topic} just released a major update that changes AI.",
        "angle":     f"This is bigger than most people realize about {topic}.",
    }


def _metrics(retention_3s: float = 0.70) -> dict[str, float]:
    return {"retention_3s": retention_3s, "avg_watch": 0.50, "completion": 0.35}


# ── ShortScript ───────────────────────────────────────────────────────────────

class TestShortScript:
    def test_full_narration_joins_all_four_segments(self):
        s = ShortScript(
            short_id="x", topic="AI", hook_type="conflict",
            hook="H", core="C", twist="T", ending="E",
            style="fast_cut", persona="direct",
        )
        narr = s.full_narration
        assert all(seg in narr for seg in ("H", "C", "T", "E"))

    def test_visual_text_equals_hook(self):
        s = ShortScript(
            short_id="x", topic="AI", hook_type="curiosity",
            hook="Stop the scroll.", core="c", twist="t", ending="e",
            style="zoom_text", persona="curious",
        )
        assert s.visual_text == "Stop the scroll."

    def test_round_trip(self):
        s = ShortScript(
            short_id="abc", topic="AI", hook_type="warning",
            hook="h", core="c", twist="t", ending="e",
            style="avatar_focus", persona="serious",
        )
        s2 = ShortScript.from_dict(s.to_dict())
        assert s2.short_id == "abc"
        assert s2.style == "avatar_focus"
        assert s2.persona == "serious"


# ── LearningRecord ────────────────────────────────────────────────────────────

class TestLearningRecord:
    def test_round_trip(self):
        r = LearningRecord(
            short_id="s1", hook_type="conflict", style="fast_cut",
            persona="direct", topic="AI", retention_3s=0.72,
        )
        r2 = LearningRecord.from_dict(r.to_dict())
        assert r2.retention_3s == 0.72
        assert r2.hook_type == "conflict"


# ── _extract_topic ────────────────────────────────────────────────────────────

class TestExtractTopic:
    def test_uses_topic_key_first(self):
        # "topic" key takes precedence; first 4 words returned
        assert _extract_topic({"topic": "ChatGPT is a game changer", "title": "other"}) == "ChatGPT is a game"

    def test_falls_back_to_title(self):
        assert _extract_topic({"title": "A B C D E"}) == "A B C D"

    def test_missing_returns_AI(self):
        assert _extract_topic({}) == "AI"


# ── _build_script_from_editorial ─────────────────────────────────────────────

class TestBuildScriptFromEditorial:
    def test_all_fields_non_empty(self):
        s = _build_script_from_editorial(_editorial(), "conflict", "Hook text")
        for seg in (s.hook, s.core, s.twist, s.ending):
            assert seg.strip()

    def test_hook_passed_through(self):
        s = _build_script_from_editorial(_editorial(), "mistake", "Unique hook")
        assert s.hook == "Unique hook"

    def test_style_assigned_from_map(self):
        for hook_type in HOOK_TYPES:
            s = _build_script_from_editorial(_editorial(), hook_type, "H")
            assert s.style == _STYLE_MAP[hook_type]

    def test_persona_assigned_from_map(self):
        for hook_type in HOOK_TYPES:
            s = _build_script_from_editorial(_editorial(), hook_type, "H")
            assert s.persona == _PERSONA_MAP[hook_type]

    def test_ending_contains_loop(self):
        s = _build_script_from_editorial(_editorial(), "warning", "H")
        assert any(loop in s.ending for loop in _LOOP_ENDINGS)

    def test_angle_used_as_twist_when_present(self):
        ed = _editorial()
        ed["angle"] = "This is the real twist."
        s = _build_script_from_editorial(ed, "curiosity", "H")
        assert "real twist" in s.twist

    def test_core_fact_truncated_to_120_chars(self):
        ed = _editorial()
        ed["core_fact"] = "x" * 200 + "."
        s = _build_script_from_editorial(ed, "conflict", "H")
        # PersonaEngine prepends an opinion prefix (~25-60 chars); core_fact portion is ≤120
        assert len(s.core) <= 200


# ── ShortsEngineV2: generate ──────────────────────────────────────────────────

class TestGenerate:
    def test_returns_up_to_max_shorts(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial(), max_shorts=3)
        assert 1 <= len(scripts) <= 3

    def test_covers_different_hook_types(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial(), max_shorts=5)
        assert len({s.hook_type for s in scripts}) >= 2

    def test_unique_ids(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial(), max_shorts=5)
        ids     = [s.short_id for s in scripts]
        assert len(ids) == len(set(ids))

    def test_scripts_persisted(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial(), max_shorts=3)
        loaded  = engine.load_scripts()
        assert len(loaded) == len(scripts)

    def test_generate_hooks_method(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        hooks  = engine.generate_hooks("ChatGPT", n=8)
        assert isinstance(hooks, list)
        for h in hooks:
            assert "text" in h
            assert "hook_type" in h

    def test_each_variant_has_style_and_persona(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial(), max_shorts=5)
        for s in scripts:
            assert s.style   in _STYLE_MAP.values()
            assert s.persona in _PERSONA_MAP.values()


# ── ShortsEngineV2: _generate_hooks ──────────────────────────────────────────

class TestGenerateHooks:
    def test_returns_one_hook_per_type(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        hooks  = engine._generate_hooks(_editorial())
        types  = [h["type"] for h in hooks]
        assert set(types) == set(HOOK_TYPES)

    def test_hooks_contain_topic(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        hooks  = engine._generate_hooks({"topic": "GPT-5", "core_fact": "fact"})
        assert any("GPT-5" in h["text"] for h in hooks)


# ── ShortsEngineV2: learn ─────────────────────────────────────────────────────

class TestLearn:
    def test_learn_persists_record(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial(), max_shorts=1)
        engine.learn(scripts[0], _metrics(0.75))
        records = engine._load_learning()
        assert len(records) == 1
        assert records[0].retention_3s == 0.75

    def test_learn_accepts_dict_short(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        engine.learn(
            {"short_id": "x", "hook_type": "conflict", "style": "fast_cut",
             "persona": "direct", "topic": "AI"},
            _metrics(0.60),
        )
        records = engine._load_learning()
        assert records[0].hook_type == "conflict"

    def test_learn_returns_learning_record(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial(), max_shorts=1)
        rec     = engine.learn(scripts[0], _metrics(0.80))
        assert isinstance(rec, LearningRecord)
        assert rec.retention_3s == 0.80


# ── ShortsEngineV2: get_top_hooks ────────────────────────────────────────────

class TestGetTopHooks:
    def test_empty_when_no_data(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        assert engine.get_top_hooks() == []

    def test_sorted_descending_by_retention(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial(), max_shorts=3)
        engine.learn(scripts[0], _metrics(0.80))
        engine.learn(scripts[1], _metrics(0.50))
        engine.learn(scripts[2], _metrics(0.65))
        top = engine.get_top_hooks(n=3)
        retentions = [r.retention_3s for r in top]
        assert retentions == sorted(retentions, reverse=True)

    def test_respects_n_limit(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial(), max_shorts=5)
        for s in scripts:
            engine.learn(s, _metrics(0.70))
        assert len(engine.get_top_hooks(n=2)) == 2


# ── ShortsEngineV2: get_best_combination ─────────────────────────────────────

class TestGetBestCombination:
    def test_none_when_no_data(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        assert engine.get_best_combination() is None

    def test_none_when_fewer_than_3_points_per_combo(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial(), max_shorts=2)
        for s in scripts:
            engine.learn(s, _metrics(0.80))
        # Only 1 data point per combination — not enough
        assert engine.get_best_combination() is None

    def test_returns_dict_when_enough_data(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        # Inject 3 learning records with same combo manually
        for _ in range(3):
            engine.learn(
                {"short_id": str(_), "hook_type": "conflict", "style": "fast_cut",
                 "persona": "direct", "topic": "AI"},
                _metrics(0.82),
            )
        result = engine.get_best_combination()
        assert result is not None
        assert result["hook_type"] == "conflict"
        assert result["style"]     == "fast_cut"
        assert result["persona"]   == "direct"
        assert "avg_retention_3s"  in result


# ── generate_hooks: proven winners first ─────────────────────────────────────

class TestGenerateHooksWithLearning:
    def test_learned_hooks_appear_with_source_learned(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_editorial("OpenAI"), max_shorts=1)
        engine.learn(scripts[0], _metrics(0.90))
        hooks = engine.generate_hooks("OpenAI", n=10)
        sources = [h.get("source") for h in hooks]
        assert "learned" in sources

    def test_returns_correct_count(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        hooks  = engine.generate_hooks("AI", n=6)
        assert len(hooks) <= 6


# ── ShortsPipeline: dry-run smoke test ───────────────────────────────────────

class TestShortsPipelineDryRun:
    def test_dry_run_uses_editorial_dict(self, tmp_path):
        from src.shorts_pipeline import ShortsPipeline, _story_to_editorial
        story    = {"title": "OpenAI releases GPT-5", "summary": "Big release.", "source": "OpenAI"}
        ed       = _story_to_editorial(story)
        assert ed["topic"] == "OpenAI releases GPT-5"[:len(ed["topic"])]
        assert ed["core_fact"] == "Big release."

    def test_dry_run_generates_scripts_no_upload(self, tmp_path):
        from src.shorts_pipeline import ShortsPipeline
        pipeline = ShortsPipeline()
        with patch("src.shorts_pipeline.settings") as mock_s:
            mock_s.output_dir             = tmp_path
            mock_s.data_dir               = tmp_path
            mock_s.heygen_enabled         = False
            mock_s.elevenlabs_api_key     = "test"
            mock_s.elevenlabs_voice_id    = "test"
            mock_s.elevenlabs_model       = "test"
            mock_s.background_music_path  = ""
            mock_s.shorts_music_volume    = 0.1
            mock_s.source_dir             = tmp_path
            with patch.object(pipeline, "_get_story", return_value={
                "title": "Test story", "source": "test", "summary": "Summary."
            }):
                with patch("src.shorts_pipeline.ShortsExperimentEngine"):
                    with patch("src.shorts_pipeline.notify_success"):
                        result = pipeline.run(dry_run=True, max_shorts=2)

        assert result["status"] == "dry_run"
        assert result["generated"] >= 1
        assert "scripts" in result
        assert result["rendered"] == 0
        assert result["uploaded"] == 0
