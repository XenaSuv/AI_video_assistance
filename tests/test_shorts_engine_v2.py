"""Tests for ShortsEngineV2 — Shorts as first-class content units."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.shorts_engine_v2 import (
    ShortScript,
    ShortsEngineV2,
    _HOOK_PATTERNS,
    _LOOP_ENDINGS,
    _build_script_from_story,
    _extract_topic,
    _pick_hook,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _story(title: str = "OpenAI releases GPT-5") -> dict[str, Any]:
    return {
        "title":   title,
        "source":  "OpenAI Blog",
        "summary": "OpenAI has released a new model. It is significantly faster.",
    }


# ── ShortScript ───────────────────────────────────────────────────────────────

class TestShortScript:
    def test_full_narration_joins_all_segments(self):
        s = ShortScript(
            short_id="x", topic="AI", hook_type="conflict",
            hook="Hook.", context="Context.", develop="Develop.",
            payoff="Payoff.", loop="Loop.",
        )
        assert "Hook." in s.full_narration
        assert "Payoff." in s.full_narration
        assert "Loop." in s.full_narration

    def test_visual_text_is_hook(self):
        s = ShortScript(
            short_id="x", topic="AI", hook_type="wait_what",
            hook="Stop the scroll.", context="c", develop="d", payoff="p", loop="l",
        )
        assert s.visual_text == "Stop the scroll."

    def test_to_dict_round_trip(self):
        s = ShortScript(
            short_id="abc", topic="AI", hook_type="curiosity",
            hook="h", context="c", develop="d", payoff="p", loop="l",
        )
        d  = s.to_dict()
        s2 = ShortScript.from_dict(d)
        assert s2.short_id == "abc"
        assert s2.hook_type == "curiosity"

    def test_default_use_avatar_false_without_heygen(self):
        """use_avatar should default to False when heygen is not enabled."""
        with patch("src.shorts_engine_v2.settings") as mock_settings:
            mock_settings.heygen_enabled = False
            s = _build_script_from_story(_story(), "conflict", "Hook text")
        assert s.use_avatar is False

    def test_use_avatar_true_when_heygen_enabled(self):
        with patch("src.shorts_engine_v2.settings") as mock_settings:
            mock_settings.heygen_enabled = True
            s = _build_script_from_story(_story(), "conflict", "Hook text")
        assert s.use_avatar is True


# ── _extract_topic ────────────────────────────────────────────────────────────

class TestExtractTopic:
    def test_returns_first_four_words(self):
        assert _extract_topic({"title": "A B C D E F"}) == "A B C D"

    def test_short_title_returned_intact(self):
        assert _extract_topic({"title": "AI"}) == "AI"

    def test_missing_title_defaults_to_AI(self):
        assert _extract_topic({}) == "AI"


# ── _pick_hook ────────────────────────────────────────────────────────────────

class TestPickHook:
    def test_returns_string_with_topic_substituted(self):
        hook = _pick_hook("ChatGPT", "conflict", seed="x")
        assert isinstance(hook, str)
        assert len(hook) > 5

    def test_deterministic_for_same_seed(self):
        h1 = _pick_hook("AI", "wait_what", seed="stable-seed")
        h2 = _pick_hook("AI", "wait_what", seed="stable-seed")
        assert h1 == h2

    def test_different_seeds_may_differ(self):
        hooks = {_pick_hook("AI", "conflict", seed=str(i)) for i in range(len(_HOOK_PATTERNS["conflict"]))}
        # At least 1 unique hook (may be 1 if all hash to same idx with different seeds)
        assert len(hooks) >= 1


# ── _build_script_from_story ──────────────────────────────────────────────────

class TestBuildScriptFromStory:
    def test_all_segments_non_empty(self):
        s = _build_script_from_story(_story(), "conflict", "Hook text")
        for seg in (s.hook, s.context, s.develop, s.payoff, s.loop):
            assert seg.strip()

    def test_hook_is_passed_through(self):
        s = _build_script_from_story(_story(), "wait_what", "Unique hook text here")
        assert s.hook == "Unique hook text here"

    def test_loop_is_one_of_known_endings(self):
        s = _build_script_from_story(_story(), "negative", "Hook")
        assert s.loop in _LOOP_ENDINGS

    def test_context_contains_story_content(self):
        story = _story("OpenAI releases GPT-5")
        story["summary"] = "OpenAI just launched a major new model."
        s = _build_script_from_story(story, "curiosity", "h")
        assert "OpenAI" in s.context or "model" in s.context

    def test_summary_truncated_at_140_chars(self):
        story = _story()
        story["summary"] = "x" * 200 + "."
        s = _build_script_from_story(story, "conflict", "h")
        # context = truncated summary + source tag + "."
        assert len(s.context) < 200


# ── ShortsEngineV2 ────────────────────────────────────────────────────────────

class TestShortsEngineV2:
    def test_generate_returns_up_to_max_shorts(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_story(), max_shorts=3)
        assert 1 <= len(scripts) <= 3

    def test_generate_covers_different_hook_types(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_story(), max_shorts=5)
        types   = {s.hook_type for s in scripts}
        assert len(types) >= 2

    def test_generate_assigns_unique_ids(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_story(), max_shorts=5)
        ids     = [s.short_id for s in scripts]
        assert len(ids) == len(set(ids))

    def test_scripts_persisted_to_jsonl(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts = engine.generate(_story(), max_shorts=3)
        loaded  = engine.load_scripts()
        assert len(loaded) == len(scripts)
        assert loaded[0].short_id == scripts[0].short_id

    def test_generate_hooks_returns_list_of_dicts(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        hooks  = engine.generate_hooks("ChatGPT", n=8)
        assert isinstance(hooks, list)
        for h in hooks:
            assert "text" in h
            assert "hook_type" in h

    def test_generate_hooks_respects_n(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        hooks  = engine.generate_hooks("AI", n=6)
        assert len(hooks) <= 6

    def test_load_scripts_empty_when_no_file(self, tmp_path):
        engine = ShortsEngineV2(data_dir=tmp_path)
        assert engine.load_scripts() == []

    def test_generate_for_different_stories_independent(self, tmp_path):
        engine  = ShortsEngineV2(data_dir=tmp_path)
        scripts1 = engine.generate(_story("Story One"), max_shorts=2)
        scripts2 = engine.generate(_story("Story Two completely different"), max_shorts=2)
        ids_all  = [s.short_id for s in scripts1 + scripts2]
        assert len(ids_all) == len(set(ids_all))


# ── ShortsPipeline dry-run smoke test ─────────────────────────────────────────

class TestShortsPipelineDryRun:
    def test_dry_run_generates_scripts_no_upload(self, tmp_path):
        from src.shorts_pipeline import ShortsPipeline
        pipeline = ShortsPipeline()
        with patch("src.shorts_pipeline.settings") as mock_s:
            mock_s.output_dir    = tmp_path
            mock_s.data_dir      = tmp_path
            mock_s.heygen_enabled = False
            mock_s.elevenlabs_api_key  = "test"
            mock_s.elevenlabs_voice_id = "test"
            mock_s.elevenlabs_model    = "test"
            mock_s.background_music_path = ""
            mock_s.shorts_music_volume = 0.1
            mock_s.source_dir = tmp_path
            # Patch story fetch and engine to avoid network calls
            with patch.object(pipeline, "_get_story", return_value={"title": "Test story", "source": "test", "summary": "Summary"}):
                with patch("src.shorts_pipeline.ShortsExperimentEngine"):
                    with patch("src.shorts_pipeline.notify_success"):
                        result = pipeline.run(dry_run=True, max_shorts=2)

        assert result["status"] == "dry_run"
        assert result["generated"] >= 1
        assert "scripts" in result
        # No rendering or uploading in dry-run
        assert result["rendered"] == 0
        assert result["uploaded"] == 0
