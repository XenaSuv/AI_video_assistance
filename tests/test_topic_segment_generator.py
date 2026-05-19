"""Tests for src/topic_segment_generator.py."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.topic_segment_generator import (
    _REFILL_THRESHOLD,
    TopicIdea,
    _load_ideas,
    _load_used,
    _log_usage,
    _mark_used,
    _save_ideas,
    generate_next_topic_ideas,
    generate_topic_script,
    pick_next_topic,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _idea(**kw) -> TopicIdea:
    defaults = dict(
        format="honest_review",
        subject="GPT-5",
        angle="Tested it for 7 days",
        suggested_title="I tested GPT-5 for 7 days",
    )
    defaults.update(kw)
    return TopicIdea(**defaults)


def _gpt_client(content: str) -> MagicMock:
    """Return a mock OpenAI client whose completion returns *content*."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.usage.prompt_tokens_details = None
    client = MagicMock()
    client.chat.completions.create.return_value = resp
    return client


def _patch_files(tmp_path):
    """Return a context-manager dict patching both idea/used file paths."""
    idea_file = tmp_path / "ideas.json"
    used_file = tmp_path / "used.json"
    return idea_file, used_file


# ── TopicIdea dataclass ───────────────────────────────────────────────────────

class TestTopicIdea:
    def test_to_dict_from_dict_roundtrip(self):
        original = _idea()
        restored = TopicIdea.from_dict(original.to_dict())
        assert restored.format == original.format
        assert restored.subject == original.subject
        assert restored.angle == original.angle
        assert restored.suggested_title == original.suggested_title

    def test_created_defaults_to_today(self):
        assert _idea().created == date.today().isoformat()

    def test_from_dict_preserves_explicit_created_date(self):
        d = _idea().to_dict()
        d["created"] = "2025-01-01"
        assert TopicIdea.from_dict(d).created == "2025-01-01"

    def test_all_four_formats_accepted(self):
        for fmt in ("honest_review", "hidden_gems", "ai_project_build", "news_with_opinion"):
            assert _idea(format=fmt).format == fmt


# ── File I/O helpers ──────────────────────────────────────────────────────────

class TestLoadSaveIdeas:
    def test_load_returns_empty_when_file_absent(self, tmp_path):
        with patch("src.topic_segment_generator._IDEA_FILE", tmp_path / "ideas.json"):
            assert _load_ideas() == []

    def test_save_then_load_roundtrip(self, tmp_path):
        ideas = [_idea(), _idea(subject="Claude 4")]
        idea_file = tmp_path / "ideas.json"
        with patch("src.topic_segment_generator._IDEA_FILE", idea_file):
            _save_ideas(ideas)
            loaded = _load_ideas()
        assert len(loaded) == 2
        assert loaded[0].subject == "GPT-5"
        assert loaded[1].subject == "Claude 4"

    def test_save_creates_parent_directories(self, tmp_path):
        target = tmp_path / "nested" / "deep" / "ideas.json"
        with patch("src.topic_segment_generator._IDEA_FILE", target):
            _save_ideas([_idea()])
        assert target.exists()


class TestLoadMarkUsed:
    def test_load_used_returns_empty_when_absent(self, tmp_path):
        with patch("src.topic_segment_generator._USED_FILE", tmp_path / "used.json"):
            assert _load_used() == []

    def test_mark_used_appends_title(self, tmp_path):
        used_file = tmp_path / "used.json"
        with patch("src.topic_segment_generator._USED_FILE", used_file):
            _mark_used(_idea(suggested_title="Title A"))
            _mark_used(_idea(suggested_title="Title B"))
            used = _load_used()
        assert "Title A" in used
        assert "Title B" in used
        assert len(used) == 2

    def test_mark_used_creates_parent_directories(self, tmp_path):
        used_file = tmp_path / "nested" / "used.json"
        with patch("src.topic_segment_generator._USED_FILE", used_file):
            _mark_used(_idea())
        assert used_file.exists()


# ── _log_usage ────────────────────────────────────────────────────────────────

class TestLogUsage:
    def test_no_op_when_usage_is_none(self):
        with patch("src.topic_segment_generator.get_ledger") as mock_ledger:
            _log_usage(None)
        mock_ledger.assert_not_called()

    def test_records_tokens_to_ledger(self):
        usage = MagicMock(prompt_tokens=100, completion_tokens=50, prompt_tokens_details=None)
        with patch("src.topic_segment_generator.get_ledger") as mock_ledger:
            _log_usage(usage, label="test")
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_reads_cached_tokens_from_details(self):
        details = MagicMock(cached_tokens=25)
        usage = MagicMock(prompt_tokens=100, completion_tokens=50, prompt_tokens_details=details)
        captured = {}
        with patch("src.topic_segment_generator.get_ledger") as mock_ledger:
            _log_usage(usage, label="x")
            captured = mock_ledger.return_value.record_llm.call_args.kwargs
        assert captured["cached_tokens"] == 25

    def test_cached_zero_when_details_is_none(self):
        usage = MagicMock(prompt_tokens=10, completion_tokens=5, prompt_tokens_details=None)
        with patch("src.topic_segment_generator.get_ledger") as mock_ledger:
            _log_usage(usage)
            captured = mock_ledger.return_value.record_llm.call_args.kwargs
        assert captured["cached_tokens"] == 0


# ── generate_next_topic_ideas ─────────────────────────────────────────────────

class TestGenerateNextTopicIdeas:
    def _run(self, tmp_path, gpt_content, existing=None):
        idea_file, used_file = _patch_files(tmp_path)
        if existing:
            idea_file.write_text(json.dumps([i.to_dict() for i in existing]))
        client = _gpt_client(gpt_content)
        with (
            patch("src.topic_segment_generator.make_openai_client", return_value=client),
            patch("src.topic_segment_generator.get_ledger"),
            patch("src.topic_segment_generator._IDEA_FILE", idea_file),
            patch("src.topic_segment_generator._USED_FILE", used_file),
        ):
            return generate_next_topic_ideas(count=1), idea_file

    def test_returns_list_of_topic_ideas(self, tmp_path):
        raw = [{"format": "honest_review", "subject": "Gemini",
                "angle": "vs GPT", "suggested_title": "Gemini honest take"}]
        ideas, _ = self._run(tmp_path, json.dumps({"ideas": raw}))
        assert len(ideas) == 1
        assert isinstance(ideas[0], TopicIdea)
        assert ideas[0].subject == "Gemini"

    def test_persists_ideas_to_disk(self, tmp_path):
        raw = [{"format": "hidden_gems", "subject": "Perplexity",
                "angle": "vs Google", "suggested_title": "Perplexity hidden gem"}]
        _, idea_file = self._run(tmp_path, json.dumps({"ideas": raw}))
        saved = json.loads(idea_file.read_text())
        assert len(saved) == 1

    def test_extends_existing_queue(self, tmp_path):
        existing = [_idea(subject="Old topic")]
        raw = [{"format": "hidden_gems", "subject": "New topic",
                "angle": "a", "suggested_title": "New gems"}]
        _, idea_file = self._run(tmp_path, json.dumps({"ideas": raw}), existing)
        saved = json.loads(idea_file.read_text())
        assert len(saved) == 2

    def test_empty_gpt_ideas_field_adds_nothing(self, tmp_path):
        ideas, _ = self._run(tmp_path, json.dumps({}))
        assert ideas == []

    def test_missing_content_adds_nothing(self, tmp_path):
        ideas, _ = self._run(tmp_path, "{}")
        assert ideas == []


# ── pick_next_topic ───────────────────────────────────────────────────────────

class TestPickNextTopic:
    def _write_ideas(self, tmp_path, n: int, **extra_kwargs):
        ideas = [_idea(subject=f"Topic {i}", suggested_title=f"Title {i}", **extra_kwargs)
                 for i in range(n)]
        f = tmp_path / "ideas.json"
        f.write_text(json.dumps([i.to_dict() for i in ideas]))
        return f

    def _run(self, tmp_path, idea_file, **kwargs):
        used_file = tmp_path / "used.json"
        with (
            patch("src.topic_segment_generator._IDEA_FILE", idea_file),
            patch("src.topic_segment_generator._USED_FILE", used_file),
        ):
            return pick_next_topic(**kwargs), used_file

    def test_returns_first_idea(self, tmp_path):
        idea_file = self._write_ideas(tmp_path, 5)
        chosen, _ = self._run(tmp_path, idea_file)
        assert chosen.subject == "Topic 0"

    def test_removes_chosen_from_queue(self, tmp_path):
        idea_file = self._write_ideas(tmp_path, 5)
        self._run(tmp_path, idea_file)
        remaining = json.loads(idea_file.read_text())
        assert len(remaining) == 4
        assert "Topic 0" not in [r["subject"] for r in remaining]

    def test_marks_chosen_as_used(self, tmp_path):
        idea_file = self._write_ideas(tmp_path, 5)
        chosen, used_file = self._run(tmp_path, idea_file)
        used = json.loads(used_file.read_text())
        assert chosen.suggested_title in used

    def test_format_filter_picks_matching_format(self, tmp_path):
        ideas = [
            _idea(subject="Review", format="honest_review", suggested_title="R0"),
            _idea(subject="Gem",    format="hidden_gems",   suggested_title="G0"),
            _idea(subject="Gem2",   format="hidden_gems",   suggested_title="G1"),
            _idea(subject="Build",  format="ai_project_build", suggested_title="B0"),
            _idea(subject="News",   format="news_with_opinion", suggested_title="N0"),
        ]
        idea_file = tmp_path / "ideas.json"
        idea_file.write_text(json.dumps([i.to_dict() for i in ideas]))
        chosen, _ = self._run(tmp_path, idea_file, format_filter="hidden_gems")
        assert chosen.format == "hidden_gems"

    def test_format_filter_falls_back_when_no_match(self, tmp_path):
        ideas = [_idea(subject=f"R{i}", format="honest_review", suggested_title=f"T{i}")
                 for i in range(5)]
        idea_file = tmp_path / "ideas.json"
        idea_file.write_text(json.dumps([i.to_dict() for i in ideas]))
        # No "hidden_gems" → logs warning and falls back to first available
        chosen, _ = self._run(tmp_path, idea_file, format_filter="hidden_gems")
        assert chosen.subject == "R0"

    def test_auto_refill_triggered_when_queue_at_threshold(self, tmp_path):
        # _REFILL_THRESHOLD = 3; 3 ideas → len(ideas) <= 3 → refill
        idea_file = self._write_ideas(tmp_path, _REFILL_THRESHOLD)
        used_file = tmp_path / "used.json"
        refill_content = json.dumps({"ideas": [
            {"format": "honest_review", "subject": "Refilled",
             "angle": "a", "suggested_title": "Refilled title"}
        ]})
        resp = MagicMock()
        resp.choices[0].message.content = refill_content
        resp.usage.prompt_tokens = 5
        resp.usage.completion_tokens = 3
        resp.usage.prompt_tokens_details = None
        client = MagicMock()
        client.chat.completions.create.return_value = resp
        with (
            patch("src.topic_segment_generator._IDEA_FILE", idea_file),
            patch("src.topic_segment_generator._USED_FILE", used_file),
            patch("src.topic_segment_generator.make_openai_client", return_value=client),
            patch("src.topic_segment_generator.get_ledger"),
        ):
            pick_next_topic()
        client.chat.completions.create.assert_called_once()

    def test_no_refill_when_queue_above_threshold(self, tmp_path):
        idea_file = self._write_ideas(tmp_path, _REFILL_THRESHOLD + 2)
        used_file = tmp_path / "used.json"
        client = MagicMock()
        with (
            patch("src.topic_segment_generator._IDEA_FILE", idea_file),
            patch("src.topic_segment_generator._USED_FILE", used_file),
            patch("src.topic_segment_generator.make_openai_client", return_value=client),
        ):
            pick_next_topic()
        client.chat.completions.create.assert_not_called()


# ── generate_topic_script ─────────────────────────────────────────────────────

class TestGenerateTopicScript:
    def _resp_content(self, n_scenes: int = 3, extras: dict | None = None) -> str:
        scenes = [
            {
                "heading": f"Scene {i}",
                "narration": f"Narration for scene {i}. " * 20,
                "visual_prompt": "visual",
                "screenshot_url": None,
                "video_query": None,
                "infographic_data": None,
                "short_narration": None,
                "source_quote": None,
                "quote_attribution": None,
            }
            for i in range(n_scenes)
        ]
        if extras:
            scenes[0].update(extras)
        return json.dumps({
            "title": "Test Script Title",
            "description": "Description text. TIMESTAMPS_AUTOFILL",
            "tags": ["ai", "test"],
            "hook_variants": ["Hook A", "Hook B"],
            "scenes": scenes,
        })

    def _run(self, idea=None, num_scenes=None, scene_extras=None, n_scenes=3):
        if idea is None:
            idea = _idea(format="honest_review")
        content = self._resp_content(n_scenes=n_scenes, extras=scene_extras)
        client = _gpt_client(content)
        with (
            patch("src.topic_segment_generator.make_openai_client", return_value=client),
            patch("src.topic_segment_generator.get_ledger"),
        ):
            return generate_topic_script(idea, num_scenes=num_scenes), client

    def test_returns_videoscript_instance(self):
        from src.script_generator import VideoScript
        script, _ = self._run()
        assert isinstance(script, VideoScript)

    def test_scene_count_matches_gpt_response(self):
        script, _ = self._run(n_scenes=5)
        assert len(script.scenes) == 5

    def test_scene_idx_sequential(self):
        script, _ = self._run(n_scenes=4)
        assert [s.idx for s in script.scenes] == [0, 1, 2, 3]

    def test_hook_is_first_hook_variant(self):
        script, _ = self._run()
        assert script.hook == "Hook A"

    def test_hook_variants_preserved(self):
        script, _ = self._run()
        assert script.hook_variants == ["Hook A", "Hook B"]

    def test_title_from_gpt_response(self):
        script, _ = self._run()
        assert script.title == "Test Script Title"

    def test_non_dict_infographic_data_stripped(self):
        script, _ = self._run(scene_extras={"infographic_data": "not-a-dict"})
        assert script.scenes[0].infographic_data is None

    def test_dict_infographic_data_preserved(self):
        data = {"type": "bar_chart", "items": []}
        script, _ = self._run(scene_extras={"infographic_data": data})
        assert script.scenes[0].infographic_data == data

    def test_non_http_screenshot_url_stripped(self):
        script, _ = self._run(scene_extras={"screenshot_url": "javascript:alert(1)"})
        assert script.scenes[0].screenshot_url is None

    def test_http_screenshot_url_kept(self):
        script, _ = self._run(scene_extras={"screenshot_url": "https://example.com/img.png"})
        assert script.scenes[0].screenshot_url == "https://example.com/img.png"

    def test_empty_video_query_becomes_none(self):
        script, _ = self._run(scene_extras={"video_query": "  "})
        assert script.scenes[0].video_query is None

    def test_non_empty_video_query_preserved(self):
        script, _ = self._run(scene_extras={"video_query": "person typing on laptop"})
        assert script.scenes[0].video_query == "person typing on laptop"

    def test_num_scenes_override_in_prompt(self):
        _, client = self._run(num_scenes=4, n_scenes=4)
        user_content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "4 scenes" in user_content

    def test_default_scene_count_honest_review_is_6(self):
        _, client = self._run(idea=_idea(format="honest_review"), n_scenes=6)
        user_content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "6 scenes" in user_content

    def test_default_scene_count_hidden_gems_is_7(self):
        _, client = self._run(idea=_idea(format="hidden_gems"), n_scenes=7)
        user_content = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        assert "7 scenes" in user_content

    def test_raw_json_stored_on_script(self):
        script, _ = self._run()
        assert script.raw_json is not None
        assert "title" in script.raw_json

    def test_fallback_title_uses_idea_suggested_title(self):
        """When GPT response has no 'title' key, falls back to idea.suggested_title."""
        idea = _idea(suggested_title="Fallback Title")
        content_data = json.loads(self._resp_content(n_scenes=2))
        del content_data["title"]
        client = _gpt_client(json.dumps(content_data))
        with (
            patch("src.topic_segment_generator.make_openai_client", return_value=client),
            patch("src.topic_segment_generator.get_ledger"),
        ):
            script = generate_topic_script(idea)
        assert script.title == "Fallback Title"
