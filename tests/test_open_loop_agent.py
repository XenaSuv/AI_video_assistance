"""Tests for src/open_loop_agent.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.open_loop_agent import (
    OpenLoop,
    OpenLoopResult,
    _build_scenes_block,
    _log_usage,
    apply_open_loops,
)
from src.script_generator import Scene, VideoScript


def _make_script(n_scenes: int = 4) -> VideoScript:
    return VideoScript(
        title="AI Deep Dive",
        description="AI topics explored.",
        tags=["AI"],
        hook="The hook",
        scenes=[
            Scene(
                idx=i,
                heading=f"Scene {i}",
                narration=f"Narration for scene {i}. " * 8,
                visual_prompt=f"visual {i}",
                duration_sec=30,
            )
            for i in range(n_scenes)
        ],
    )


def _mock_response(payload: dict) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps(payload)
    resp.usage = None
    client.chat.completions.create.return_value = resp
    return client


def _good_payload(n_loops: int = 2) -> dict:
    return {
        "rewritten_intro": "New intro that teases upcoming drama and key reveals.",
        "loops": [
            {
                "teaser": "Why the benchmark numbers are actually misleading",
                "closing": "Here is the payoff for loop one. The results surprised everyone.",
                "target_scene_idx": i + 1,
                "priority": "primary" if i == 0 else "secondary",
            }
            for i in range(n_loops)
        ],
    }


class TestApplyOpenLoops:
    def test_returns_result(self):
        script = _make_script()
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(_good_payload())):
            result = apply_open_loops(script)
        assert isinstance(result, OpenLoopResult)

    def test_modified_true_on_success(self):
        script = _make_script()
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(_good_payload())):
            result = apply_open_loops(script)
        assert result.modified is True

    def test_rewritten_intro_applied(self):
        script = _make_script()
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(_good_payload())):
            result = apply_open_loops(script)
        assert "New intro" in result.script.scenes[0].narration

    def test_original_intro_unchanged(self):
        script = _make_script()
        original_narration = script.scenes[0].narration
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(_good_payload())):
            apply_open_loops(script)
        assert script.scenes[0].narration == original_narration

    def test_loop_closing_prepended_to_target_scene(self):
        script = _make_script()
        payload = _good_payload(n_loops=1)
        payload["loops"][0]["target_scene_idx"] = 2
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(payload)):
            result = apply_open_loops(script)
        assert "payoff for loop one" in result.script.scenes[2].narration

    def test_too_few_scenes_returns_unmodified(self):
        script = _make_script(n_scenes=2)
        result = apply_open_loops(script)
        assert result.modified is False
        assert result.script is script

    def test_api_exception_returns_unmodified(self):
        script = _make_script()
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API error")
        with patch("src.open_loop_agent.make_openai_client", return_value=client):
            result = apply_open_loops(script)
        assert result.modified is False

    def test_empty_rewritten_intro_returns_unmodified(self):
        script = _make_script()
        payload = {"rewritten_intro": "", "loops": [{"teaser": "x", "closing": "y", "target_scene_idx": 1}]}
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(payload)):
            result = apply_open_loops(script)
        assert result.modified is False

    def test_empty_loops_returns_unmodified(self):
        script = _make_script()
        payload = {"rewritten_intro": "New intro.", "loops": []}
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(payload)):
            result = apply_open_loops(script)
        assert result.modified is False

    def test_invalid_loop_items_skipped(self):
        script = _make_script()
        payload = {
            "rewritten_intro": "New intro.",
            "loops": [
                {"teaser": "", "closing": "valid", "target_scene_idx": 1},  # empty teaser
                {"teaser": "valid", "closing": "", "target_scene_idx": 1},  # empty closing
            ],
        }
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(payload)):
            result = apply_open_loops(script)
        assert result.modified is False

    def test_target_scene_idx_clamped_to_valid_range(self):
        script = _make_script(n_scenes=3)
        payload = {
            "rewritten_intro": "Rewritten intro here for testing.",
            "loops": [{"teaser": "teaser text", "closing": "closing text", "target_scene_idx": 99}],
        }
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(payload)):
            result = apply_open_loops(script)
        assert result.modified is True
        assert result.loops[0].target_scene_idx <= 2

    def test_scene_count_preserved(self):
        script = _make_script(n_scenes=5)
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(_good_payload())):
            result = apply_open_loops(script)
        assert len(result.script.scenes) == 5

    def test_returns_loops_list(self):
        script = _make_script()
        with patch("src.open_loop_agent.make_openai_client",
                   return_value=_mock_response(_good_payload(n_loops=2))):
            result = apply_open_loops(script)
        assert isinstance(result.loops, list)
        assert len(result.loops) == 2

    def test_logs_usage_when_present(self):
        script = _make_script()
        client = _mock_response(_good_payload())
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 50
        usage.prompt_tokens_details = None
        client.chat.completions.create.return_value.usage = usage
        with patch("src.open_loop_agent.make_openai_client", return_value=client):
            with patch("src.open_loop_agent.get_ledger") as mock_ledger:
                apply_open_loops(script)
        mock_ledger.return_value.record_llm.assert_called_once()


class TestBuildScenesBlock:
    def test_includes_all_scenes(self):
        script = _make_script(n_scenes=3)
        block = _build_scenes_block(script.scenes)
        assert "Scene 0" in block
        assert "Scene 1" in block
        assert "Scene 2" in block

    def test_truncates_long_narration(self):
        script = _make_script(n_scenes=2)
        for s in script.scenes:
            s.narration = "x" * 1000
        block = _build_scenes_block(script.scenes)
        # Each scene narration is truncated to 400 chars
        for line in block.split("[Scene"):
            assert len(line) < 600


class TestLogUsage:
    def test_no_op_when_none(self):
        _log_usage(None)

    def test_calls_record_llm(self):
        usage = MagicMock()
        usage.prompt_tokens = 80
        usage.completion_tokens = 20
        usage.prompt_tokens_details = None
        with patch("src.open_loop_agent.get_ledger") as mock_ledger:
            _log_usage(usage)
        mock_ledger.return_value.record_llm.assert_called_once()
