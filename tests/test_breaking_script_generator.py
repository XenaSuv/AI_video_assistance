"""Tests for src/breaking_script_generator.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.breaking_script_generator import (
    _log_usage,
    generate_breaking_script,
    generate_breaking_short_script,
)
from src.scraper import NewsItem
from src.script_generator import VideoScript


def _make_item(
    title: str = "GPT-6 Launches",
    source: str = "OpenAI",
    summary: str = "OpenAI releases GPT-6 with breakthrough capabilities.",
    url: str = "https://openai.com/gpt6",
) -> NewsItem:
    return NewsItem(
        source=source,
        title=title,
        url=url,
        summary=summary,
        authors=[],
        published="2026-05-17",
    )


def _mock_scene_payload(n: int = 5) -> dict:
    return {
        "title": "BREAKING: GPT-6 Launches",
        "description": "OpenAI releases GPT-6.",
        "tags": ["AI", "OpenAI"],
        "hook": "Something massive just happened.",
        "scenes": [
            {
                "heading": f"Scene {i}",
                "narration": f"Narration for scene {i}. " * 10,
                "visual_prompt": f"photorealistic prompt {i}",
            }
            for i in range(n)
        ],
    }


def _mock_short_payload() -> dict:
    return {
        "title": "BREAKING: GPT-6",
        "script": "GPT-6 just dropped and it changes everything.",
        "hook": "This is massive.",
        "visual_prompt": "photorealistic editorial AI scene",
        "tags": ["AI"],
    }


def _mock_client(content: str) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage = None
    client.chat.completions.create.return_value = resp
    return client


class TestGenerateBreakingScript:
    def test_returns_video_script(self):
        payload = _mock_scene_payload()
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_script(_make_item())
        assert isinstance(result, VideoScript)

    def test_title_set_from_payload(self):
        payload = _mock_scene_payload()
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_script(_make_item())
        assert result.title == payload["title"]

    def test_hook_set_from_payload(self):
        payload = _mock_scene_payload()
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_script(_make_item())
        assert result.hook == payload["hook"]

    def test_scene_count_matches_payload(self):
        payload = _mock_scene_payload(n=5)
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_script(_make_item())
        assert len(result.scenes) == 5

    def test_scene_idx_sequential(self):
        payload = _mock_scene_payload(n=5)
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_script(_make_item())
        assert [s.idx for s in result.scenes] == [0, 1, 2, 3, 4]

    def test_logs_usage_when_present(self):
        payload = _mock_scene_payload()
        client = _mock_client(json.dumps(payload))
        usage = MagicMock()
        usage.prompt_tokens = 200
        usage.completion_tokens = 50
        usage.prompt_tokens_details = None
        client.chat.completions.create.return_value.usage = usage
        with patch("src.breaking_script_generator.make_openai_client", return_value=client):
            with patch("src.breaking_script_generator.get_ledger") as mock_ledger:
                generate_breaking_script(_make_item())
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_item_without_summary(self):
        payload = _mock_scene_payload()
        item = _make_item(summary=None)
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_script(item)
        assert isinstance(result, VideoScript)

    def test_scene_visual_prompt_set(self):
        payload = _mock_scene_payload(n=2)
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_script(_make_item())
        assert result.scenes[0].visual_prompt == "photorealistic prompt 0"


class TestGenerateBreakingShortScript:
    def test_returns_video_script(self):
        payload = _mock_short_payload()
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_short_script(_make_item())
        assert isinstance(result, VideoScript)

    def test_single_scene(self):
        payload = _mock_short_payload()
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_short_script(_make_item())
        assert len(result.scenes) == 1

    def test_title_from_payload(self):
        payload = _mock_short_payload()
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_short_script(_make_item())
        assert result.title == "BREAKING: GPT-6"

    def test_narration_from_script_field(self):
        payload = _mock_short_payload()
        with patch("src.breaking_script_generator.make_openai_client",
                   return_value=_mock_client(json.dumps(payload))):
            result = generate_breaking_short_script(_make_item())
        assert "GPT-6 just dropped" in result.scenes[0].narration

    def test_logs_usage_when_present(self):
        payload = _mock_short_payload()
        client = _mock_client(json.dumps(payload))
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 30
        usage.prompt_tokens_details = None
        client.chat.completions.create.return_value.usage = usage
        with patch("src.breaking_script_generator.make_openai_client", return_value=client):
            with patch("src.breaking_script_generator.get_ledger") as mock_ledger:
                generate_breaking_short_script(_make_item())
        mock_ledger.return_value.record_llm.assert_called_once()


class TestLogUsage:
    def test_no_op_when_none(self):
        _log_usage(None)

    def test_calls_record_llm(self):
        usage = MagicMock()
        usage.prompt_tokens = 80
        usage.completion_tokens = 15
        usage.prompt_tokens_details = None
        with patch("src.breaking_script_generator.get_ledger") as mock_ledger:
            _log_usage(usage, "breaking")
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_zero_prompt_tokens_no_division_error(self):
        usage = MagicMock()
        usage.prompt_tokens = 0
        usage.completion_tokens = 5
        usage.prompt_tokens_details = None
        with patch("src.breaking_script_generator.get_ledger") as mock_ledger:
            _log_usage(usage)
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_handles_cached_tokens(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 20
        details = MagicMock()
        details.cached_tokens = 40
        usage.prompt_tokens_details = details
        with patch("src.breaking_script_generator.get_ledger") as mock_ledger:
            _log_usage(usage)
        mock_ledger.return_value.record_llm.assert_called_once()
