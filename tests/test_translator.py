"""Tests for translator — translate_script() mocking OpenAI."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from src.script_generator import Scene, VideoScript
from src.translator import translate_script, _log_usage


def _make_script(n_scenes: int = 2) -> VideoScript:
    return VideoScript(
        title="AI revolution",
        description="A look at AI today.",
        tags=["AI", "news"],
        hook="This will blow your mind",
        scenes=[
            Scene(
                idx=i,
                heading=f"Scene {i}",
                narration=f"Narration for scene {i}",
                visual_prompt=f"visual prompt {i}",
                duration_sec=30 + i,
            )
            for i in range(n_scenes)
        ],
    )


def _mock_response(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps(data)
    resp.usage = None
    return resp


def _translated_payload(script: VideoScript, suffix: str = "_ru") -> dict:
    """Build the JSON the mock LLM would return."""
    return {
        "title":       script.title + suffix,
        "description": script.description + suffix,
        "tags":        [t + suffix for t in script.tags],
        "hook":        script.hook + suffix,
        "scenes": [
            {
                "idx":           s.idx,
                "heading":       s.heading + suffix,
                "narration":     s.narration + suffix,
                "visual_prompt": s.visual_prompt,   # must stay unchanged
                "duration_sec":  s.duration_sec,    # must stay unchanged
            }
            for s in script.scenes
        ],
    }


class TestTranslateScript:
    def test_returns_video_script(self):
        script = _make_script()
        with patch("src.translator.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                _mock_response(_translated_payload(script))
            )
            result = translate_script(script, "Russian")
        assert isinstance(result, VideoScript)

    def test_translated_title(self):
        script = _make_script()
        with patch("src.translator.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                _mock_response(_translated_payload(script))
            )
            result = translate_script(script, "Russian")
        assert result.title == script.title + "_ru"

    def test_visual_prompt_not_translated(self):
        script = _make_script(n_scenes=2)
        with patch("src.translator.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                _mock_response(_translated_payload(script))
            )
            result = translate_script(script, "Russian")
        for orig, translated in zip(script.scenes, result.scenes):
            assert translated.visual_prompt == orig.visual_prompt

    def test_duration_sec_preserved(self):
        script = _make_script(n_scenes=2)
        with patch("src.translator.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                _mock_response(_translated_payload(script))
            )
            result = translate_script(script, "Russian")
        for orig, translated in zip(script.scenes, result.scenes):
            assert translated.duration_sec == orig.duration_sec

    def test_scene_count_preserved(self):
        script = _make_script(n_scenes=3)
        with patch("src.translator.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                _mock_response(_translated_payload(script))
            )
            result = translate_script(script, "Russian")
        assert len(result.scenes) == 3

    def test_original_script_not_mutated(self):
        script = _make_script()
        original_title = script.title
        with patch("src.translator.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                _mock_response(_translated_payload(script))
            )
            translate_script(script, "Russian")
        assert script.title == original_title

    def test_scene_idx_preserved(self):
        script = _make_script(n_scenes=3)
        with patch("src.translator.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                _mock_response(_translated_payload(script))
            )
            result = translate_script(script, "Russian")
        for orig, translated in zip(script.scenes, result.scenes):
            assert translated.idx == orig.idx


class TestLogUsage:
    def test_no_op_when_usage_is_none(self):
        # Should return without error
        _log_usage(None)

    def test_calls_record_llm_with_usage(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 20
        usage.prompt_tokens_details = MagicMock(cached_tokens=30)
        with patch("src.translator.get_ledger") as mock_ledger:
            _log_usage(usage, label="test-label")
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_handles_none_details(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 10
        usage.prompt_tokens_details = None
        with patch("src.translator.get_ledger") as mock_ledger:
            _log_usage(usage)
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_handles_zero_prompt_tokens(self):
        usage = MagicMock()
        usage.prompt_tokens = 0
        usage.completion_tokens = 5
        usage.prompt_tokens_details = None
        with patch("src.translator.get_ledger") as mock_ledger:
            _log_usage(usage, label="zero-pct")
        # Should not raise ZeroDivisionError
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_label_optional(self):
        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 10
        usage.prompt_tokens_details = None
        with patch("src.translator.get_ledger") as mock_ledger:
            _log_usage(usage)  # no label
        mock_ledger.return_value.record_llm.assert_called_once()
