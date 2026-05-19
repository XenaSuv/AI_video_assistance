"""Tests for src/weekly_script_generator.py — topic rotation + tutorial generation."""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.script_generator import Scene
from src.weekly_script_generator import (
    _REUSE_COOLDOWN_WEEKS,
    _fill_missing_short_narrations,
    _format_keys_block,
    _load_used_topics,
    _log_usage,
    _topics_file,
    generate_tutorial_script,
    get_tool,
    pick_topic,
    save_used_topic,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _scene(idx: int, short_narration: str | None = None) -> Scene:
    return Scene(
        idx=idx,
        heading=f"Scene {idx}",
        narration="Narration text for scene. " * 20,
        visual_prompt="A compelling visual.",
        short_narration=short_narration,
    )


def _minimal_raw(
    n_scenes: int = 7,
    short_count: int = 3,
    hook_variants: list[str] | None = None,
) -> dict:
    scenes = []
    for i in range(n_scenes):
        scenes.append({
            "heading": f"Scene {i}",
            "narration": "Narration. " * 30,
            "visual_prompt": "A photo of something.",
            "screenshot_key": None,
            "short_narration": f"Short narration {i}." if i < short_count else None,
            "video_query": "",
            "infographic_data": None,
            "source_quote": "",
            "quote_attribution": None,
        })
    return {
        "title": "Complete Claude Tutorial",
        "description": "Learn Claude quickly.",
        "tags": ["Claude", "AI", "Tutorial"],
        "hook_variants": hook_variants if hook_variants is not None else ["Hook A", "Hook B", "Hook C"],
        "hook": "Fallback hook",
        "scenes": scenes,
    }


def _mock_client(raw: dict | None = None) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps(raw or _minimal_raw())
    resp.usage = None
    client.chat.completions.create.return_value = resp
    return client


# ── get_tool ──────────────────────────────────────────────────────────────────

class TestGetTool:
    def test_returns_claude_tool(self):
        tool = get_tool("claude")
        assert tool.key == "claude"

    def test_returns_chatgpt_tool(self):
        tool = get_tool("chatgpt")
        assert tool.key == "chatgpt"

    def test_returns_gemini_tool(self):
        tool = get_tool("gemini")
        assert tool.key == "gemini"

    def test_each_tool_has_name(self):
        assert get_tool("claude").name == "Claude"
        assert get_tool("chatgpt").name == "ChatGPT"
        assert get_tool("gemini").name == "Gemini"

    def test_each_tool_has_topics(self):
        for key in ("claude", "chatgpt", "gemini"):
            assert len(get_tool(key).topics) > 0

    def test_raises_for_unknown_key(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            get_tool("grok")


# ── _topics_file ──────────────────────────────────────────────────────────────

class TestTopicsFile:
    def test_returns_path_with_tool_key(self, tmp_path):
        p = _topics_file(tmp_path, "claude")
        assert p == tmp_path / "weekly_topics_claude.json"

    def test_different_keys_produce_different_paths(self, tmp_path):
        assert _topics_file(tmp_path, "claude") != _topics_file(tmp_path, "chatgpt")


# ── _load_used_topics ─────────────────────────────────────────────────────────

class TestLoadUsedTopics:
    def test_returns_empty_list_when_file_missing(self, tmp_path):
        assert _load_used_topics(tmp_path, "claude") == []

    def test_returns_used_list_from_file(self, tmp_path):
        data = {"used": [{"topic": "Topic A", "date": "2025-01-01"}]}
        (tmp_path / "weekly_topics_claude.json").write_text(json.dumps(data))
        result = _load_used_topics(tmp_path, "claude")
        assert len(result) == 1
        assert result[0]["topic"] == "Topic A"

    def test_handles_file_without_used_key(self, tmp_path):
        (tmp_path / "weekly_topics_claude.json").write_text("{}")
        assert _load_used_topics(tmp_path, "claude") == []


# ── save_used_topic ───────────────────────────────────────────────────────────

class TestSaveUsedTopic:
    def test_creates_file_on_first_save(self, tmp_path):
        save_used_topic(tmp_path, "claude", "Topic A")
        assert (tmp_path / "weekly_topics_claude.json").exists()

    def test_appends_to_existing_list(self, tmp_path):
        save_used_topic(tmp_path, "claude", "Topic A")
        save_used_topic(tmp_path, "claude", "Topic B")
        used = _load_used_topics(tmp_path, "claude")
        assert len(used) == 2

    def test_records_todays_date(self, tmp_path):
        save_used_topic(tmp_path, "claude", "Topic A")
        used = _load_used_topics(tmp_path, "claude")
        assert used[0]["date"] == date.today().isoformat()

    def test_topic_in_saved_file(self, tmp_path):
        save_used_topic(tmp_path, "gemini", "Gemini for coding")
        used = _load_used_topics(tmp_path, "gemini")
        assert used[0]["topic"] == "Gemini for coding"

    def test_isolated_per_tool(self, tmp_path):
        save_used_topic(tmp_path, "claude", "Claude topic")
        save_used_topic(tmp_path, "chatgpt", "ChatGPT topic")
        assert len(_load_used_topics(tmp_path, "claude")) == 1
        assert _load_used_topics(tmp_path, "claude")[0]["topic"] == "Claude topic"


# ── pick_topic ────────────────────────────────────────────────────────────────

class TestPickTopic:
    def test_returns_a_valid_tool_topic(self, tmp_path):
        topic = pick_topic(tmp_path, "claude")
        assert topic in get_tool("claude").topics

    def test_excludes_recently_used_topics(self, tmp_path):
        tool = get_tool("claude")
        # Mark all topics except the last one as recently used
        all_but_last = tool.topics[:-1]
        data = {"used": [
            {"topic": t, "date": date.today().isoformat()}
            for t in all_but_last
        ]}
        (tmp_path / "weekly_topics_claude.json").write_text(json.dumps(data))
        topic = pick_topic(tmp_path, "claude")
        assert topic == tool.topics[-1]

    def test_includes_topics_older_than_cooldown(self, tmp_path):
        tool = get_tool("claude")
        old_date = (date.today() - timedelta(weeks=_REUSE_COOLDOWN_WEEKS + 2)).isoformat()
        all_but_last = tool.topics[:-1]
        data = {"used": [
            {"topic": t, "date": old_date}
            for t in all_but_last
        ]}
        (tmp_path / "weekly_topics_claude.json").write_text(json.dumps(data))
        # Old topics are past cooldown → all are available
        topic = pick_topic(tmp_path, "claude")
        assert topic in tool.topics

    def test_falls_back_to_full_list_when_all_recently_used(self, tmp_path):
        tool = get_tool("claude")
        data = {"used": [
            {"topic": t, "date": date.today().isoformat()}
            for t in tool.topics
        ]}
        (tmp_path / "weekly_topics_claude.json").write_text(json.dumps(data))
        # All used → warning + fallback to full list → still returns a valid topic
        topic = pick_topic(tmp_path, "claude")
        assert topic in tool.topics

    def test_returns_string(self, tmp_path):
        assert isinstance(pick_topic(tmp_path, "chatgpt"), str)


# ── _format_keys_block ────────────────────────────────────────────────────────

class TestFormatKeysBlock:
    def test_returns_none_message_when_no_keys(self):
        with patch("src.weekly_script_generator.available_keys", return_value=[]):
            result = _format_keys_block("claude")
        assert "none" in result.lower()
        assert "DALL-E" in result

    def test_returns_formatted_list_when_keys_exist(self):
        with (
            patch("src.weekly_script_generator.available_keys", return_value=["api_overview"]),
            patch("src.weekly_script_generator.url_for", return_value="https://example.com"),
        ):
            result = _format_keys_block("claude")
        assert "api_overview" in result
        assert "https://example.com" in result


# ── _log_usage ────────────────────────────────────────────────────────────────

class TestLogUsage:
    def test_returns_early_when_usage_is_none(self):
        mock_ledger = MagicMock()
        with patch("src.weekly_script_generator.get_ledger", return_value=mock_ledger):
            _log_usage(None)
        mock_ledger.record_llm.assert_not_called()

    def test_calls_record_llm_with_token_counts(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.prompt_tokens_details = None
        mock_ledger = MagicMock()
        ms = MagicMock()
        ms.openai_model = "gpt-4o-mini"
        with (
            patch("src.weekly_script_generator.get_ledger", return_value=mock_ledger),
            patch("src.weekly_script_generator.settings", ms),
        ):
            _log_usage(usage, "test-label")
        mock_ledger.record_llm.assert_called_once()
        kwargs = mock_ledger.record_llm.call_args.kwargs
        assert kwargs["prompt_tokens"] == 100
        assert kwargs["completion_tokens"] == 50

    def test_computes_cached_tokens_from_details(self):
        details = MagicMock()
        details.cached_tokens = 40
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 20
        usage.prompt_tokens_details = details
        mock_ledger = MagicMock()
        ms = MagicMock(openai_model="gpt-4o-mini")
        with (
            patch("src.weekly_script_generator.get_ledger", return_value=mock_ledger),
            patch("src.weekly_script_generator.settings", ms),
        ):
            _log_usage(usage)
        kwargs = mock_ledger.record_llm.call_args.kwargs
        assert kwargs["cached_tokens"] == 40


# ── _fill_missing_short_narrations ───────────────────────────────────────────

class TestFillMissingShortNarrations:
    def test_returns_immediately_when_enough(self):
        scenes = [_scene(i, short_narration=f"Short {i}") for i in range(3)]
        client = MagicMock()
        _fill_missing_short_narrations(scenes, client, min_count=3)
        client.chat.completions.create.assert_not_called()

    def test_does_not_call_gpt_when_more_than_min(self):
        scenes = [_scene(i, short_narration=f"Short {i}") for i in range(5)]
        client = MagicMock()
        _fill_missing_short_narrations(scenes, client, min_count=3)
        client.chat.completions.create.assert_not_called()

    def test_calls_gpt_when_short_narrations_missing(self):
        scenes = [_scene(i) for i in range(5)]  # none have short_narration
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
            {"scenes": [{"idx": i, "short_narration": f"S{i}"} for i in range(1, 4)]}
        )
        client.chat.completions.create.return_value.usage = None
        ms = MagicMock(openai_model="gpt-4o-mini")
        with patch("src.weekly_script_generator.settings", ms):
            _fill_missing_short_narrations(scenes, client, min_count=3)
        client.chat.completions.create.assert_called_once()

    def test_fills_scenes_by_idx(self):
        scenes = [_scene(i) for i in range(5)]
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
            {"scenes": [
                {"idx": 1, "short_narration": "Narr for 1"},
                {"idx": 2, "short_narration": "Narr for 2"},
                {"idx": 3, "short_narration": "Narr for 3"},
            ]}
        )
        client.chat.completions.create.return_value.usage = None
        ms = MagicMock(openai_model="gpt-4o-mini")
        with patch("src.weekly_script_generator.settings", ms):
            _fill_missing_short_narrations(scenes, client, min_count=3)
        assert scenes[1].short_narration == "Narr for 1"
        assert scenes[2].short_narration == "Narr for 2"
        assert scenes[3].short_narration == "Narr for 3"

    def test_gpt_failure_is_non_fatal(self):
        scenes = [_scene(i) for i in range(5)]
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("quota exceeded")
        ms = MagicMock(openai_model="gpt-4o-mini")
        with patch("src.weekly_script_generator.settings", ms):
            _fill_missing_short_narrations(scenes, client, min_count=3)  # must not raise

    def test_prefers_middle_scenes_over_first_and_last(self):
        scenes = [_scene(i) for i in range(7)]  # none have short_narration
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
            {"scenes": []}
        )
        client.chat.completions.create.return_value.usage = None
        ms = MagicMock(openai_model="gpt-4o-mini")
        with patch("src.weekly_script_generator.settings", ms):
            _fill_missing_short_narrations(scenes, client, min_count=3)
        # The scenes block sent to GPT should only contain middle scenes (idx 1–5)
        call_content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "idx=0" not in call_content
        assert "idx=6" not in call_content

    def test_fills_only_needed_count(self):
        # 1 already filled; min_count=3 → need 2 more
        scenes = [_scene(0, short_narration="Already here")] + [_scene(i) for i in range(1, 6)]
        client = MagicMock()
        client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
            {"scenes": [
                {"idx": 1, "short_narration": "Fill 1"},
                {"idx": 2, "short_narration": "Fill 2"},
            ]}
        )
        client.chat.completions.create.return_value.usage = None
        ms = MagicMock(openai_model="gpt-4o-mini")
        with patch("src.weekly_script_generator.settings", ms):
            _fill_missing_short_narrations(scenes, client, min_count=3)
        # Only 2 scenes should be in the prompt (needed=2)
        call_content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "idx=1" in call_content
        assert "idx=2" in call_content
        assert "idx=3" not in call_content


# ── generate_tutorial_script ─────────────────────────────────────────────────

class TestGenerateTutorialScript:
    def _patch_ctx(self, client=None, raw=None, has_key_val=True):
        """Return an ExitStack with all external deps mocked."""
        from contextlib import ExitStack
        stack = ExitStack()
        ms = MagicMock(openai_model="gpt-4o-mini")
        stack.enter_context(patch("src.weekly_script_generator.settings", ms))
        stack.enter_context(
            patch("src.weekly_script_generator.make_openai_client",
                  return_value=client or _mock_client(raw))
        )
        stack.enter_context(
            patch("src.weekly_script_generator.available_keys", return_value=[])
        )
        stack.enter_context(
            patch("src.weekly_script_generator.has_key", return_value=has_key_val)
        )
        stack.enter_context(
            patch("src.weekly_script_generator._fill_missing_short_narrations")
        )
        return stack

    def test_returns_video_script(self, tmp_path):
        from src.script_generator import VideoScript
        stack = self._patch_ctx()
        with stack:
            result = generate_tutorial_script("Claude API", "claude")
        assert isinstance(result, VideoScript)

    def test_parses_title(self, tmp_path):
        raw = _minimal_raw()
        raw["title"] = "Custom Title"
        stack = self._patch_ctx(raw=raw)
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        assert script.title == "Custom Title"

    def test_parses_description(self):
        raw = _minimal_raw()
        raw["description"] = "Custom description text."
        stack = self._patch_ctx(raw=raw)
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        assert script.description == "Custom description text."

    def test_parses_all_scenes(self):
        raw = _minimal_raw(n_scenes=7)
        stack = self._patch_ctx(raw=raw)
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        assert len(script.scenes) == 7

    def test_scene_idx_is_sequential(self):
        stack = self._patch_ctx()
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        for i, scene in enumerate(script.scenes):
            assert scene.idx == i

    def test_unknown_screenshot_key_becomes_none(self):
        raw = _minimal_raw()
        raw["scenes"][0]["screenshot_key"] = "bad_key_xyz"
        stack = self._patch_ctx(raw=raw, has_key_val=False)
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        assert script.scenes[0].screenshot_key is None

    def test_known_screenshot_key_is_preserved(self):
        raw = _minimal_raw()
        raw["scenes"][0]["screenshot_key"] = "api_overview"
        stack = self._patch_ctx(raw=raw, has_key_val=True)
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        assert script.scenes[0].screenshot_key == "api_overview"

    def test_empty_video_query_becomes_none(self):
        raw = _minimal_raw()
        raw["scenes"][0]["video_query"] = "   "  # whitespace only
        stack = self._patch_ctx(raw=raw)
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        assert script.scenes[0].video_query is None

    def test_non_dict_infographic_becomes_none(self):
        raw = _minimal_raw()
        raw["scenes"][0]["infographic_data"] = "this should be a dict"
        stack = self._patch_ctx(raw=raw)
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        assert script.scenes[0].infographic_data is None

    def test_dict_infographic_is_preserved(self):
        raw = _minimal_raw()
        raw["scenes"][0]["infographic_data"] = {"type": "bar_chart", "items": []}
        stack = self._patch_ctx(raw=raw)
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        assert script.scenes[0].infographic_data == {"type": "bar_chart", "items": []}

    def test_uses_first_hook_variant_when_no_data_dir(self):
        raw = _minimal_raw(hook_variants=["First hook", "Second hook"])
        stack = self._patch_ctx(raw=raw)
        with stack:
            script = generate_tutorial_script("Claude API", "claude", data_dir=None)
        assert script.hook == "First hook"

    def test_calls_pick_hook_when_data_dir_and_multiple_variants(self, tmp_path):
        raw = _minimal_raw(hook_variants=["Hook A", "Hook B", "Hook C"])
        mock_pick = MagicMock(return_value="Hook B")
        stack = self._patch_ctx(raw=raw)
        with stack:
            with patch("src.hook_selector.pick_hook", mock_pick):
                script = generate_tutorial_script("Claude API", "claude", data_dir=tmp_path)
        mock_pick.assert_called_once()
        assert script.hook == "Hook B"

    def test_fallback_hook_from_raw_when_variants_empty(self):
        raw = _minimal_raw(hook_variants=[])
        raw["hook"] = "Fallback hook text"
        stack = self._patch_ctx(raw=raw)
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        assert script.hook == "Fallback hook text"

    def test_hook_variants_stored_on_script(self):
        raw = _minimal_raw(hook_variants=["H1", "H2", "H3"])
        stack = self._patch_ctx(raw=raw)
        with stack:
            script = generate_tutorial_script("Claude API", "claude")
        assert script.hook_variants == ["H1", "H2", "H3"]

    def test_calls_fill_missing_short_narrations(self):
        from contextlib import ExitStack
        stack = ExitStack()
        ms = MagicMock(openai_model="gpt-4o-mini")
        stack.enter_context(patch("src.weekly_script_generator.settings", ms))
        stack.enter_context(
            patch("src.weekly_script_generator.make_openai_client",
                  return_value=_mock_client())
        )
        stack.enter_context(
            patch("src.weekly_script_generator.available_keys", return_value=[])
        )
        stack.enter_context(
            patch("src.weekly_script_generator.has_key", return_value=True)
        )
        mock_fill = stack.enter_context(
            patch("src.weekly_script_generator._fill_missing_short_narrations")
        )
        with stack:
            generate_tutorial_script("Claude API", "claude")
        mock_fill.assert_called_once()
