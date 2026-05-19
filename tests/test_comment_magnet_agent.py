"""Tests for src/comment_magnet_agent.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.comment_magnet_agent import (
    CommentMagnetResult,
    _build_scenes_block,
    _log_usage,
    apply_comment_magnet,
)
from src.script_generator import Scene, VideoScript


def _make_script(n_scenes: int = 4) -> VideoScript:
    return VideoScript(
        title="AI Revolution",
        description="A look at AI.",
        tags=["AI"],
        hook="Hook text",
        scenes=[
            Scene(
                idx=i,
                heading=f"Scene {i}",
                narration=f"This is narration for scene {i}. " * 5,
                visual_prompt=f"visual {i}",
                duration_sec=30,
            )
            for i in range(n_scenes)
        ],
    )


def _mock_client(question: str = "Would you trust AI to make this decision?") -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps({
        "question": question,
        "reasoning": "Based on scene 2 claim",
    })
    resp.usage = None
    client.chat.completions.create.return_value = resp
    return client


class TestApplyCommentMagnet:
    def test_returns_result_with_question(self):
        script = _make_script()
        with patch("src.comment_magnet_agent.make_openai_client", return_value=_mock_client()):
            result = apply_comment_magnet(script)
        assert isinstance(result, CommentMagnetResult)
        assert result.modified is True
        assert "?" in result.question

    def test_too_few_scenes_returns_unmodified(self):
        script = _make_script(n_scenes=1)
        result = apply_comment_magnet(script)
        assert result.modified is False
        assert result.question == ""
        assert result.inserted_at_scene == -1

    def test_question_inserted_into_second_to_last_scene(self):
        script = _make_script(n_scenes=4)
        question = "Am I wrong that this changes everything?"
        with patch("src.comment_magnet_agent.make_openai_client", return_value=_mock_client(question)):
            result = apply_comment_magnet(script)
        target = result.inserted_at_scene
        assert target == 2  # second to last of 4 scenes
        assert question in result.script.scenes[target].narration

    def test_original_script_not_mutated(self):
        script = _make_script()
        original_narrations = [s.narration for s in script.scenes]
        with patch("src.comment_magnet_agent.make_openai_client", return_value=_mock_client()):
            apply_comment_magnet(script)
        for orig, scene in zip(original_narrations, script.scenes, strict=False):
            assert scene.narration == orig

    def test_returns_unmodified_on_openai_exception(self):
        script = _make_script()
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        with patch("src.comment_magnet_agent.make_openai_client", return_value=client):
            result = apply_comment_magnet(script)
        assert result.modified is False
        assert result.script is script

    def test_returns_unmodified_when_empty_question(self):
        script = _make_script()
        with patch("src.comment_magnet_agent.make_openai_client", return_value=_mock_client("")):
            result = apply_comment_magnet(script)
        assert result.modified is False

    def test_subject_and_format_accepted(self):
        script = _make_script()
        with patch("src.comment_magnet_agent.make_openai_client", return_value=_mock_client()):
            result = apply_comment_magnet(script, subject="OpenAI GPT-5", format_name="deep-dive")
        assert result.modified is True

    def test_scene_count_preserved(self):
        script = _make_script(n_scenes=5)
        with patch("src.comment_magnet_agent.make_openai_client", return_value=_mock_client()):
            result = apply_comment_magnet(script)
        assert len(result.script.scenes) == 5


class TestBuildScenesBlock:
    def test_skips_first_and_last_for_long_script(self):
        script = _make_script(n_scenes=5)
        block = _build_scenes_block(script.scenes)
        assert "Scene 0" not in block
        assert "Scene 4" not in block
        assert "Scene 1" in block or "Scene 2" in block

    def test_uses_all_scenes_when_two_or_fewer(self):
        script = _make_script(n_scenes=2)
        block = _build_scenes_block(script.scenes)
        assert "Scene 0" in block
        assert "Scene 1" in block

    def test_truncates_narration(self):
        script = _make_script(n_scenes=3)
        for s in script.scenes:
            s.narration = "x" * 1000
        block = _build_scenes_block(script.scenes)
        # Each scene narration should be ≤ 300 chars in block
        for line in block.split("\n"):
            assert len(line) <= 320


class TestLogUsage:
    def test_no_op_when_none(self):
        _log_usage(None)

    def test_calls_record_llm(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 20
        usage.prompt_tokens_details = None
        with patch("src.comment_magnet_agent.get_ledger") as mock_ledger:
            _log_usage(usage, "test")
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_handles_cached_tokens(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 20
        details = MagicMock()
        details.cached_tokens = 50
        usage.prompt_tokens_details = details
        with patch("src.comment_magnet_agent.get_ledger") as mock_ledger:
            _log_usage(usage)
        mock_ledger.return_value.record_llm.assert_called_once()
