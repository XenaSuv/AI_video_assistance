"""Tests for src/script_generator.py.

Covers:
- Scene dataclass (creation, from_dict via __dict__ round-trip)
- VideoScript.save() / load() / full_narration / to_dict
- _build_items_block()
- _build_strategy_block()
- _build_recommendation_block()
- _log_usage()
- _fill_missing_short_narrations()
- generate_script() — happy path + edge cases (empty items, no hook_variants, etc.)
- generate_short_script()
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import src.script_generator as _sg_module
from src.scraper import NewsItem

# ── env vars must be set BEFORE any project module is imported ──────────────
# conftest.py already handles this, but be explicit for clarity.
# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------
from src.script_generator import (
    _DAILY_SHORTS_FILL_PROMPT,
    Scene,
    VideoScript,
    _build_items_block,
    _build_recommendation_block,
    _build_strategy_block,
    _fill_missing_short_narrations,
    _log_usage,
    generate_script,
    generate_short_script,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_news_item(
    title: str = "Test AI Story",
    source: str = "OpenAI",
    url: str = "https://example.com/story",
    summary: str = "A test summary about AI breakthroughs.",
    authors: list[str] | None = None,
    published: str = "2026-05-16",
) -> NewsItem:
    return NewsItem(
        source=source,
        title=title,
        url=url,
        summary=summary,
        authors=authors or [],
        published=published,
    )


def _make_scene(idx: int = 0, narration: str = "Test narration.", short_narration: str | None = None) -> Scene:
    return Scene(
        idx=idx,
        heading=f"Scene {idx}",
        narration=narration,
        visual_prompt="A cinematic AI newsroom scene",
        short_narration=short_narration,
    )


def _make_openai_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like an openai ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50
    usage.prompt_tokens_details = None
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# Scene tests
# ---------------------------------------------------------------------------

class TestScene:
    def test_defaults(self):
        scene = Scene(idx=0, heading="Test", narration="Hello", visual_prompt="img")
        assert scene.duration_sec == 0
        assert scene.screenshot_key is None
        assert scene.screenshot_url is None
        assert scene.short_narration is None
        assert scene.infographic_data is None
        assert scene.video_query is None
        assert scene.source_quote is None
        assert scene.quote_attribution is None
        assert scene.scene_type == "image"
        assert scene.scene_intent == "explain"
        assert scene.voice_direction == ""
        assert scene.visual_direction == ""
        assert scene.pause_after == ""
        assert scene.subtitle_style == ""
        assert scene.emotion == "neutral"
        assert scene.emotion_intensity == 0.0
        assert scene.avatar_style == "normal"

    def test_dict_roundtrip(self):
        """__dict__ should contain all fields — used by VideoScript.save()."""
        scene = Scene(
            idx=1,
            heading="My heading",
            narration="This is narration.",
            visual_prompt="An AI image.",
            short_narration="Short version.",
            duration_sec=15,
            video_query="robot arm factory",
            source_quote="We achieved SOTA",
            quote_attribution="Jane Doe, CEO",
        )
        d = scene.__dict__
        assert d["idx"] == 1
        assert d["heading"] == "My heading"
        assert d["narration"] == "This is narration."
        assert d["short_narration"] == "Short version."
        assert d["duration_sec"] == 15
        assert d["video_query"] == "robot arm factory"
        assert d["source_quote"] == "We achieved SOTA"
        assert d["quote_attribution"] == "Jane Doe, CEO"

    def test_from_dict_via_kwargs(self):
        """Simulate from-dict construction (as done in VideoScript load pattern)."""
        d = {
            "idx": 3,
            "heading": "Loaded",
            "narration": "Loaded narration.",
            "visual_prompt": "Loaded prompt.",
            "duration_sec": 20,
            "screenshot_key": None,
            "screenshot_url": None,
            "short_narration": "Loaded short.",
            "infographic_data": {"type": "stat_card", "value": "5B"},
            "video_query": "server room",
            "source_quote": None,
            "quote_attribution": None,
            "scene_type": "image",
            "scene_intent": "data",
            "voice_direction": "strong",
            "visual_direction": "zoom_in",
            "pause_after": "short",
            "subtitle_style": "bold_highlight",
            "emotion": "excitement",
            "emotion_intensity": 0.8,
            "avatar_style": "energetic",
        }
        scene = Scene(**d)
        assert scene.idx == 3
        assert scene.emotion == "excitement"
        assert scene.infographic_data == {"type": "stat_card", "value": "5B"}


# ---------------------------------------------------------------------------
# VideoScript tests
# ---------------------------------------------------------------------------

class TestVideoScript:
    def _make_script(self, scenes: list[Scene] | None = None) -> VideoScript:
        if scenes is None:
            scenes = [
                _make_scene(0, "First narration."),
                _make_scene(1, "Second narration."),
            ]
        return VideoScript(
            title="Test Title",
            description="A test description.",
            tags=["ai", "news"],
            hook="This is the hook.",
            hook_variants=["Hook A", "Hook B"],
            scenes=scenes,
        )

    def test_full_narration_empty(self):
        script = VideoScript(title="T", description="D", tags=[], hook="H")
        assert script.full_narration == ""

    def test_full_narration_single_scene(self):
        script = self._make_script(scenes=[_make_scene(0, "Only scene.")])
        assert script.full_narration == "Only scene."

    def test_full_narration_multiple_scenes(self):
        script = self._make_script()
        assert script.full_narration == "First narration.\n\nSecond narration."

    def test_save_creates_valid_json(self, tmp_path):
        script = self._make_script()
        path = tmp_path / "script.json"
        script.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["title"] == "Test Title"
        assert data["description"] == "A test description."
        assert data["tags"] == ["ai", "news"]
        assert data["hook"] == "This is the hook."
        assert data["hook_variants"] == ["Hook A", "Hook B"]
        assert len(data["scenes"]) == 2
        assert data["scenes"][0]["narration"] == "First narration."

    def test_save_scene_fields_present(self, tmp_path):
        scene = _make_scene(0, "Narration here.", short_narration="Short!")
        script = self._make_script(scenes=[scene])
        path = tmp_path / "script.json"
        script.save(path)
        data = json.loads(path.read_text())
        s = data["scenes"][0]
        assert s["short_narration"] == "Short!"
        assert s["visual_prompt"] == "A cinematic AI newsroom scene"

    def test_save_and_load_roundtrip(self, tmp_path):
        """Save then manually parse to verify integrity."""
        scene = Scene(
            idx=0,
            heading="Breaking AI",
            narration="GPT-5 was released today.",
            visual_prompt="A dramatic newsroom.",
            infographic_data={"type": "stat_card", "value": "5B", "label": "Parameters"},
        )
        script = VideoScript(
            title="Breaking AI News",
            description="Today's top stories",
            tags=["ai", "gpt-5"],
            hook="GPT-5 just changed everything.",
            hook_variants=["Variant A", "Variant B", "Variant C"],
            scenes=[scene],
        )
        path = tmp_path / "out.json"
        script.save(path)
        loaded = json.loads(path.read_text())
        assert loaded["title"] == "Breaking AI News"
        assert loaded["scenes"][0]["infographic_data"]["type"] == "stat_card"


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestBuildItemsBlock:
    def test_single_item_no_authors(self):
        item = _make_news_item(title="AI Beats Humans", url="https://example.com/ai")
        block = _build_items_block([item])
        assert "AI Beats Humans" in block
        assert "https://example.com/ai" in block
        assert "OpenAI" in block
        assert "A test summary" in block

    def test_multiple_items_numbered(self):
        items = [
            _make_news_item(title="Story One"),
            _make_news_item(title="Story Two", source="HuggingFace"),
        ]
        block = _build_items_block(items)
        assert "1. [OpenAI] Story One" in block
        assert "2. [HuggingFace] Story Two" in block

    def test_authors_truncated_at_three(self):
        item = _make_news_item(authors=["Alice", "Bob", "Carol", "Dave", "Eve"])
        block = _build_items_block([item])
        assert "et al." in block
        assert "Alice" in block

    def test_few_authors_no_etal(self):
        item = _make_news_item(authors=["Alice", "Bob"])
        block = _build_items_block([item])
        assert "et al." not in block

    def test_summary_truncated_at_500(self):
        long_summary = "X" * 600
        item = _make_news_item(summary=long_summary)
        block = _build_items_block([item])
        # The block should include at most 500 chars of the summary
        assert "X" * 500 in block
        assert "X" * 501 not in block

    def test_empty_list(self):
        block = _build_items_block([])
        assert block == ""


class TestBuildStrategyBlock:
    def test_none_returns_empty(self):
        assert _build_strategy_block(None) == ""

    def test_empty_dict_returns_empty(self):
        # Empty dict is falsy → returns "" (same as None)
        block = _build_strategy_block({})
        assert block == ""

    def test_strategy_with_persona(self):
        block = _build_strategy_block({"persona": "skeptic"})
        assert "persona: skeptic" in block

    def test_strategy_with_sequence_bias(self):
        block = _build_strategy_block({
            "sequence_bias": [["hook", "twist", "data"], ["hook", "data"]],
        })
        assert "hook → twist → data" in block
        assert "hook → data" in block

    def test_strategy_with_all_fields(self):
        block = _build_strategy_block({
            "pace": "fast",
            "hook_aggressiveness": 0.8,
            "packaging_style": "bold",
            "persona": "enthusiast",
            "sequence_bias": [["hook", "shock"]],
        })
        assert "pace: fast" in block
        assert "hook_aggressiveness: 0.80" in block
        assert "packaging_style: bold" in block
        assert "persona: enthusiast" in block
        assert "hook → shock" in block
        assert "Use this as light guidance" in block


class TestBuildRecommendationBlock:
    def test_empty_analytics_returns_empty(self):
        with patch("src.script_generator.get_proven_hooks", return_value=[]), \
             patch("src.script_generator.get_proven_titles", return_value=[]), \
             patch("src.script_generator.top_keywords", return_value=[]):
            result = _build_recommendation_block()
        assert result == ""

    def test_with_hooks_only(self):
        with patch("src.script_generator.get_proven_hooks", return_value=["Hook A", "Hook B"]), \
             patch("src.script_generator.get_proven_titles", return_value=[]), \
             patch("src.script_generator.top_keywords", return_value=[]):
            result = _build_recommendation_block()
        assert "ANALYTICS INSIGHTS" in result
        assert "Hook A" in result
        assert "Hook B" in result

    def test_with_titles_only(self):
        with patch("src.script_generator.get_proven_hooks", return_value=[]), \
             patch("src.script_generator.get_proven_titles", return_value=["Title X"]), \
             patch("src.script_generator.top_keywords", return_value=[]):
            result = _build_recommendation_block()
        assert "Title X" in result

    def test_with_keywords_only(self):
        with patch("src.script_generator.get_proven_hooks", return_value=[]), \
             patch("src.script_generator.get_proven_titles", return_value=[]), \
             patch("src.script_generator.top_keywords", return_value=[("gpt", 100), ("llm", 80)]):
            result = _build_recommendation_block()
        assert "gpt" in result
        assert "llm" in result

    def test_with_all_data(self):
        with patch("src.script_generator.get_proven_hooks", return_value=["Great Hook"]), \
             patch("src.script_generator.get_proven_titles", return_value=["Top Title"]), \
             patch("src.script_generator.top_keywords", return_value=[("ai", 500)]):
            result = _build_recommendation_block()
        assert "Great Hook" in result
        assert "Top Title" in result
        assert "ai" in result



class TestLogUsage:
    def test_no_usage_returns_none(self):
        result = _log_usage(None)
        assert result is None

    def test_logs_usage_with_cached_tokens(self):
        usage = MagicMock()
        usage.prompt_tokens = 200
        usage.completion_tokens = 50
        details = MagicMock()
        details.cached_tokens = 100
        usage.prompt_tokens_details = details

        with patch("src.script_generator.get_ledger") as mock_ledger:
            mock_record = MagicMock()
            mock_ledger.return_value = mock_record
            _log_usage(usage, label="test", model="gpt-4o")
            mock_record.record_llm.assert_called_once_with(
                tag="test",
                model="gpt-4o",
                prompt_tokens=200,
                completion_tokens=50,
                cached_tokens=100,
            )

    def test_logs_usage_no_details(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 40
        usage.prompt_tokens_details = None

        with patch("src.script_generator.get_ledger") as mock_ledger:
            mock_record = MagicMock()
            mock_ledger.return_value = mock_record
            _log_usage(usage)
            mock_record.record_llm.assert_called_once()
            call_kwargs = mock_record.record_llm.call_args[1]
            assert call_kwargs["cached_tokens"] == 0


# ---------------------------------------------------------------------------
# _fill_missing_short_narrations tests
# ---------------------------------------------------------------------------

class TestFillMissingShortNarrations:
    def test_already_enough_short_narrations(self):
        scenes = [
            _make_scene(0, short_narration="Short 1"),
            _make_scene(1, short_narration="Short 2"),
            _make_scene(2),
        ]
        client = MagicMock()
        _fill_missing_short_narrations(scenes, client, min_count=2)
        # client.chat.completions.create should NOT be called
        client.chat.completions.create.assert_not_called()

    def test_fills_one_missing(self):
        scenes = [
            _make_scene(0, short_narration="Already has it"),
            _make_scene(1),
        ]
        fill_response = _make_openai_response(json.dumps({
            "scenes": [{"idx": 1, "short_narration": "Auto-filled short narration."}]
        }))
        client = MagicMock()
        client.chat.completions.create.return_value = fill_response
        _fill_missing_short_narrations(scenes, client, min_count=2)
        assert scenes[1].short_narration == "Auto-filled short narration."

    def test_fills_when_none_have_short(self):
        scenes = [_make_scene(i) for i in range(3)]
        fill_response = _make_openai_response(json.dumps({
            "scenes": [
                {"idx": 0, "short_narration": "Short 0"},
                {"idx": 1, "short_narration": "Short 1"},
            ]
        }))
        client = MagicMock()
        client.chat.completions.create.return_value = fill_response
        _fill_missing_short_narrations(scenes, client, min_count=2)
        # idx 0 and 1 are candidates (skipping first and last based on slicing)
        # At least one should be filled
        client.chat.completions.create.assert_called_once()

    def test_api_failure_is_non_fatal(self):
        scenes = [_make_scene(0), _make_scene(1)]
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")
        # Should not raise — just log a warning
        _fill_missing_short_narrations(scenes, client, min_count=2)
        # Narrations remain None
        assert scenes[0].short_narration is None
        assert scenes[1].short_narration is None

    def test_empty_scenes_no_crash(self):
        client = MagicMock()
        _fill_missing_short_narrations([], client, min_count=2)
        client.chat.completions.create.assert_not_called()

    def test_bad_json_in_fill_response(self):
        scenes = [_make_scene(0), _make_scene(1)]
        fill_response = _make_openai_response("not json {{{")
        fill_response.choices[0].message.content = "not json {{{"
        client = MagicMock()
        client.chat.completions.create.return_value = fill_response
        # Should not crash — json.loads failure caught in except block
        _fill_missing_short_narrations(scenes, client, min_count=2)


# ---------------------------------------------------------------------------
# generate_script tests
# ---------------------------------------------------------------------------

def _make_generate_script_response(
    title: str = "AI Takes Over",
    num_scenes: int = 3,
    hook_variants: list[str] | None = None,
) -> str:
    if hook_variants is None:
        hook_variants = ["Hook fact variant", "Hook question variant", "Hook bold variant"]
    scenes = []
    for i in range(num_scenes):
        scenes.append({
            "heading": f"Scene {i} heading",
            "narration": f"This is narration for scene {i}. " * 30,
            "visual_prompt": f"Visual prompt for scene {i}",
            "video_query": None,
            "infographic_data": None,
            "short_narration": "Short: AI news. Subscribe for daily AI news." if i == 0 else None,
            "source_quote": None,
            "quote_attribution": None,
        })
    data = {
        "title": title,
        "description": "Episode description. (TIMESTAMPS_AUTOFILL)\nSources: https://example.com",
        "tags": ["ai", "news", "gpt"],
        "hook_variants": hook_variants,
        "scenes": scenes,
    }
    return json.dumps(data)


class TestGenerateScript:
    def _patch_client(self, script_content: str, fill_content: str | None = None):
        """Return a mock client whose completions.create returns appropriate responses."""
        fill_content = fill_content or json.dumps({
            "scenes": [{"idx": 1, "short_narration": "Auto-filled."}]
        })
        script_resp = _make_openai_response(script_content)
        fill_resp = _make_openai_response(fill_content)

        client = MagicMock()
        # First call is the main script; subsequent calls are fill_missing
        client.chat.completions.create.side_effect = [script_resp, fill_resp, fill_resp]
        return client

    def test_raises_on_empty_items(self):
        with pytest.raises(ValueError, match="No news items"):
            with patch("src.script_generator.make_openai_client"):
                generate_script([])

    def test_basic_generation(self):
        items = [_make_news_item()]
        script_content = _make_generate_script_response(num_scenes=4)
        fill_content = json.dumps({"scenes": [{"idx": 1, "short_narration": "Auto."}]})

        with patch("src.script_generator.make_openai_client") as mock_factory:
            client = self._patch_client(script_content, fill_content)
            mock_factory.return_value = client
            result = generate_script(items)

        assert isinstance(result, VideoScript)
        assert result.title == "AI Takes Over"
        assert len(result.scenes) == 4
        assert result.hook in result.hook_variants

    def test_first_hook_variant_chosen_without_data_dir(self):
        items = [_make_news_item()]
        script_content = _make_generate_script_response()
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory:
            client = self._patch_client(script_content, fill_content)
            mock_factory.return_value = client
            result = generate_script(items, data_dir=None)

        # Without data_dir, first hook_variant should be chosen
        assert result.hook == "Hook fact variant"

    def test_with_strategy_block(self):
        items = [_make_news_item()]
        script_content = _make_generate_script_response()
        strategy = {"pace": "fast", "hook_aggressiveness": 0.9, "packaging_style": "bold"}
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory:
            client = self._patch_client(script_content, fill_content)
            mock_factory.return_value = client
            result = generate_script(items, strategy=strategy)

        assert result.title == "AI Takes Over"

    def test_with_data_dir_analytics(self, tmp_path):
        items = [_make_news_item()]
        script_content = _make_generate_script_response()
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory, \
             patch("src.script_generator.get_proven_hooks", return_value=["Test hook"]), \
             patch("src.script_generator.get_proven_titles", return_value=[]), \
             patch("src.script_generator.top_keywords", return_value=[]):
            client = self._patch_client(script_content, fill_content)
            mock_factory.return_value = client
            result = generate_script(items, data_dir=tmp_path)

        assert result is not None

    def test_with_editorial_plan_dict(self):
        items = [_make_news_item()]
        script_content = _make_generate_script_response()
        fill_content = json.dumps({"scenes": []})
        editorial_plan = {
            "selected_stories": [],
            "editorial_plan": [],
            "global_style": {"tone": "serious"},
        }

        with patch("src.script_generator.make_openai_client") as mock_factory, \
             patch("src.script_generator.editorial_plan_to_prompt", return_value="PLAN PROMPT"):
            client = self._patch_client(script_content, fill_content)
            mock_factory.return_value = client
            result = generate_script(items, editorial_plan=editorial_plan)

        assert result is not None

    def test_with_editorial_plan_object(self):
        from src.editorial_brain import EditorialPlan
        items = [_make_news_item()]
        script_content = _make_generate_script_response()
        fill_content = json.dumps({"scenes": []})
        plan = EditorialPlan(selected_stories=[], editorial_plan=[], global_style={})

        with patch("src.script_generator.make_openai_client") as mock_factory, \
             patch("src.script_generator.editorial_plan_to_prompt", return_value="PLAN PROMPT"):
            client = self._patch_client(script_content, fill_content)
            mock_factory.return_value = client
            result = generate_script(items, editorial_plan=plan)

        assert result is not None

    def test_missing_visual_prompt_falls_back_to_heading(self):
        items = [_make_news_item()]
        # Make a scene with no visual_prompt
        raw_data = {
            "title": "Test",
            "description": "Desc",
            "tags": [],
            "hook_variants": ["Hook"],
            "scenes": [{
                "heading": "From Heading",
                "narration": "Narration text.",
                "visual_prompt": "",  # empty
                "video_query": None,
                "infographic_data": None,
                "short_narration": "Short. Subscribe for daily AI news.",
                "source_quote": None,
                "quote_attribution": None,
            }],
        }
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory:
            script_resp = _make_openai_response(json.dumps(raw_data))
            fill_resp = _make_openai_response(fill_content)
            client = MagicMock()
            client.chat.completions.create.side_effect = [script_resp, fill_resp]
            mock_factory.return_value = client
            result = generate_script(items)

        assert result.scenes[0].visual_prompt == "From Heading"

    def test_missing_heading_and_visual_prompt_falls_back(self):
        items = [_make_news_item()]
        raw_data = {
            "title": "Test",
            "description": "Desc",
            "tags": [],
            "hook_variants": ["Hook"],
            "scenes": [{
                "heading": "",
                "narration": "First sentence. Second sentence.",
                "visual_prompt": "",
                "video_query": None,
                "infographic_data": None,
                "short_narration": "Short. Subscribe for daily AI news.",
                "source_quote": None,
                "quote_attribution": None,
            }],
        }
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory:
            script_resp = _make_openai_response(json.dumps(raw_data))
            fill_resp = _make_openai_response(fill_content)
            client = MagicMock()
            client.chat.completions.create.side_effect = [script_resp, fill_resp]
            mock_factory.return_value = client
            result = generate_script(items)

        # Falls back to first sentence of narration
        assert result.scenes[0].visual_prompt == "First sentence"

    def test_invalid_infographic_data_rejected(self):
        items = [_make_news_item()]
        raw_data = {
            "title": "Test",
            "description": "Desc",
            "tags": [],
            "hook_variants": ["Hook"],
            "scenes": [{
                "heading": "Scene",
                "narration": "Narration.",
                "visual_prompt": "Image",
                "video_query": None,
                "infographic_data": "not a dict",  # should be rejected
                "short_narration": "Short. Subscribe for daily AI news.",
                "source_quote": None,
                "quote_attribution": None,
            }],
        }
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory:
            script_resp = _make_openai_response(json.dumps(raw_data))
            fill_resp = _make_openai_response(fill_content)
            client = MagicMock()
            client.chat.completions.create.side_effect = [script_resp, fill_resp]
            mock_factory.return_value = client
            result = generate_script(items)

        assert result.scenes[0].infographic_data is None

    def test_valid_infographic_data_accepted(self):
        items = [_make_news_item()]
        raw_data = {
            "title": "Test",
            "description": "Desc",
            "tags": [],
            "hook_variants": ["Hook"],
            "scenes": [{
                "heading": "Scene",
                "narration": "Narration.",
                "visual_prompt": "Image",
                "video_query": None,
                "infographic_data": {"type": "stat_card", "value": "5B"},
                "short_narration": "Short. Subscribe for daily AI news.",
                "source_quote": None,
                "quote_attribution": None,
            }],
        }
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory:
            script_resp = _make_openai_response(json.dumps(raw_data))
            fill_resp = _make_openai_response(fill_content)
            client = MagicMock()
            client.chat.completions.create.side_effect = [script_resp, fill_resp]
            mock_factory.return_value = client
            result = generate_script(items)

        assert result.scenes[0].infographic_data == {"type": "stat_card", "value": "5B"}

    def test_old_style_single_hook_fallback(self):
        """When hook_variants is absent, uses data['hook'] as fallback."""
        items = [_make_news_item()]
        raw_data = {
            "title": "Fallback Hook Test",
            "description": "Desc",
            "tags": [],
            # No hook_variants key
            "hook": "Old style single hook",
            "scenes": [{
                "heading": "Scene",
                "narration": "Narration.",
                "visual_prompt": "Image",
                "video_query": None,
                "infographic_data": None,
                "short_narration": "Short. Subscribe for daily AI news.",
                "source_quote": None,
                "quote_attribution": None,
            }],
        }
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory:
            script_resp = _make_openai_response(json.dumps(raw_data))
            fill_resp = _make_openai_response(fill_content)
            client = MagicMock()
            client.chat.completions.create.side_effect = [script_resp, fill_resp]
            mock_factory.return_value = client
            result = generate_script(items)

        assert result.hook == "Old style single hook"

    def test_analytics_block_failure_is_non_fatal(self, tmp_path):
        items = [_make_news_item()]
        script_content = _make_generate_script_response()
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory, \
             patch("src.script_generator.get_proven_hooks", side_effect=RuntimeError("DB error")):
            client = self._patch_client(script_content, fill_content)
            mock_factory.return_value = client
            # Should not raise
            result = generate_script(items, data_dir=tmp_path)

        assert result is not None

    def test_editorial_plan_failure_is_non_fatal(self):
        items = [_make_news_item()]
        script_content = _make_generate_script_response()
        fill_content = json.dumps({"scenes": []})
        # Pass a badly-formed editorial_plan dict that causes a failure
        bad_plan = {"bad": "data"}

        with patch("src.script_generator.make_openai_client") as mock_factory, \
             patch("src.script_generator.editorial_plan_to_prompt", side_effect=Exception("Bad plan")):
            client = self._patch_client(script_content, fill_content)
            mock_factory.return_value = client
            result = generate_script(items, editorial_plan=bad_plan)

        assert result is not None

    def test_source_quote_and_attribution(self):
        items = [_make_news_item()]
        raw_data = {
            "title": "Test",
            "description": "Desc",
            "tags": [],
            "hook_variants": ["Hook"],
            "scenes": [{
                "heading": "Scene",
                "narration": "Narration.",
                "visual_prompt": "Image",
                "video_query": "developer laptop",
                "infographic_data": None,
                "short_narration": "Short. Subscribe for daily AI news.",
                "source_quote": "We achieved SOTA on all benchmarks",
                "quote_attribution": "Sam Altman, CEO · OpenAI",
            }],
        }
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory:
            script_resp = _make_openai_response(json.dumps(raw_data))
            fill_resp = _make_openai_response(fill_content)
            client = MagicMock()
            client.chat.completions.create.side_effect = [script_resp, fill_resp]
            mock_factory.return_value = client
            result = generate_script(items)

        scene = result.scenes[0]
        assert scene.source_quote == "We achieved SOTA on all benchmarks"
        assert scene.quote_attribution == "Sam Altman, CEO · OpenAI"
        assert scene.video_query == "developer laptop"

    def test_hook_selector_used_with_data_dir(self, tmp_path):
        items = [_make_news_item()]
        script_content = _make_generate_script_response(
            hook_variants=["Fact hook", "Question hook", "Bold hook"]
        )
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory, \
             patch("src.script_generator.get_proven_hooks", return_value=[]), \
             patch("src.script_generator.get_proven_titles", return_value=[]), \
             patch("src.script_generator.top_keywords", return_value=[]), \
             patch("src.hook_selector.pick_hook", return_value="Question hook") as mock_pick:
            client = self._patch_client(script_content, fill_content)
            mock_factory.return_value = client
            result = generate_script(items, data_dir=tmp_path)

        assert result.hook in ["Fact hook", "Question hook", "Bold hook"]

    def test_num_scenes_parameter(self):
        items = [_make_news_item()]
        script_content = _make_generate_script_response(num_scenes=6)
        fill_content = json.dumps({"scenes": []})

        with patch("src.script_generator.make_openai_client") as mock_factory:
            client = self._patch_client(script_content, fill_content)
            mock_factory.return_value = client
            result = generate_script(items, num_scenes=6)

        assert len(result.scenes) == 6


# ---------------------------------------------------------------------------
# generate_short_script tests
# ---------------------------------------------------------------------------

class TestGenerateShortScript:
    def test_basic_generation(self, fresh_story):
        response_text = "I tested this AI tool and it blew my mind."
        mock_resp = _make_openai_response(response_text)

        with patch("src.script_generator.make_openai_client") as mock_factory:
            client = MagicMock()
            client.chat.completions.create.return_value = mock_resp
            mock_factory.return_value = client
            result = generate_short_script(fresh_story, "Here's a hook")

        assert result == response_text

    def test_uses_story_summary(self, fresh_story):
        """Verify that the prompt is built from story.summary."""
        mock_resp = _make_openai_response("Generated short.")

        with patch("src.script_generator.make_openai_client") as mock_factory:
            client = MagicMock()
            client.chat.completions.create.return_value = mock_resp
            mock_factory.return_value = client
            generate_short_script(fresh_story, "Hook text")
            call_args = client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            # The prompt content should contain the story summary
            assert fresh_story.summary in messages[0]["content"]

    def test_uses_hook_in_prompt(self, fresh_story):
        mock_resp = _make_openai_response("Short script result.")

        with patch("src.script_generator.make_openai_client") as mock_factory:
            client = MagicMock()
            client.chat.completions.create.return_value = mock_resp
            mock_factory.return_value = client
            generate_short_script(fresh_story, "My special hook")
            call_args = client.chat.completions.create.call_args
            messages = call_args[1]["messages"]
            assert "My special hook" in messages[0]["content"]

    def test_dict_input_fallback(self):
        """If input is a dict instead of NewsItem, uses .get('summary')."""
        news_dict = {"summary": "Dict-based summary for AI story."}
        mock_resp = _make_openai_response("Generated from dict.")

        with patch("src.script_generator.make_openai_client") as mock_factory:
            client = MagicMock()
            client.chat.completions.create.return_value = mock_resp
            mock_factory.return_value = client
            result = generate_short_script(news_dict, "Hook")

        assert result == "Generated from dict."

    def test_empty_response_returns_empty_string(self, fresh_story):
        mock_resp = _make_openai_response(None)

        with patch("src.script_generator.make_openai_client") as mock_factory:
            client = MagicMock()
            client.chat.completions.create.return_value = mock_resp
            mock_factory.return_value = client
            result = generate_short_script(fresh_story, "Hook")

        assert result == ""

    def test_temperature_is_high(self, fresh_story):
        """Short scripts should use temperature=0.9."""
        mock_resp = _make_openai_response("Result.")

        with patch("src.script_generator.make_openai_client") as mock_factory:
            client = MagicMock()
            client.chat.completions.create.return_value = mock_resp
            mock_factory.return_value = client
            generate_short_script(fresh_story, "Hook")
            call_args = client.chat.completions.create.call_args
            assert call_args[1]["temperature"] == 0.9
