"""Tests for src/hooks_generator.py — generate_hooks()."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.hooks_generator import generate_hooks


def _mock_client(content: str) -> MagicMock:
    client = MagicMock()
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage = None
    client.chat.completions.create.return_value = resp
    return client


class TestGenerateHooks:
    def test_returns_list_of_strings(self):
        content = json.dumps(["Hook one", "Hook two", "Hook three"])
        with patch("src.hooks_generator.make_openai_client", return_value=_mock_client(content)):
            result = generate_hooks("AI news text")
        assert isinstance(result, list)
        assert all(isinstance(h, str) for h in result)

    def test_returns_n_hooks(self):
        content = json.dumps(["H1", "H2", "H3"])
        with patch("src.hooks_generator.make_openai_client", return_value=_mock_client(content)):
            result = generate_hooks("text", n=3)
        assert len(result) == 3

    def test_strips_code_block_markers(self):
        content = "```json\n[\"Hook A\", \"Hook B\"]\n```"
        with patch("src.hooks_generator.make_openai_client", return_value=_mock_client(content)):
            result = generate_hooks("text", n=2)
        assert len(result) == 2
        assert result[0] == "Hook A"

    def test_fallback_on_invalid_json(self):
        with patch("src.hooks_generator.make_openai_client", return_value=_mock_client("not-json")):
            result = generate_hooks("text", n=3)
        assert len(result) == 3
        assert all("matters" in h.lower() for h in result)

    def test_fallback_when_response_not_list(self):
        content = json.dumps({"hooks": ["A", "B"]})
        with patch("src.hooks_generator.make_openai_client", return_value=_mock_client(content)):
            result = generate_hooks("text", n=2)
        assert len(result) == 2

    def test_returns_empty_list_on_empty_response(self):
        with patch("src.hooks_generator.make_openai_client", return_value=_mock_client("")):
            result = generate_hooks("text", n=2)
        assert isinstance(result, list)

    def test_records_usage_when_present(self):
        content = json.dumps(["H1"])
        client = _mock_client(content)
        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 10
        client.chat.completions.create.return_value.usage = usage
        with patch("src.hooks_generator.make_openai_client", return_value=client):
            with patch("src.hooks_generator.get_ledger") as mock_ledger:
                generate_hooks("text")
        mock_ledger.return_value.record_llm.assert_called_once()

    def test_strips_whitespace_from_hooks(self):
        content = json.dumps(["  Hook with spaces  ", "Another"])
        with patch("src.hooks_generator.make_openai_client", return_value=_mock_client(content)):
            result = generate_hooks("text")
        assert result[0] == "Hook with spaces"

    def test_filters_empty_hooks(self):
        content = json.dumps(["Valid hook", "", "   "])
        with patch("src.hooks_generator.make_openai_client", return_value=_mock_client(content)):
            result = generate_hooks("text")
        assert "" not in result
        assert "   " not in result
