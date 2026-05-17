"""Tests for title_optimizer — generate_titles() with mocked OpenAI."""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import pytest


def _mock_response(content: str) -> MagicMock:
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage = None
    return resp


def _mock_response_str(content: str) -> MagicMock:
    """Same as _mock_response but for pick_best_title which returns a plain string."""
    return _mock_response(content)


# title_optimizer creates a module-level client; patch make_openai_client
# before the module is imported so the module-level call succeeds.
with patch("src.retry_utils.make_openai_client", return_value=MagicMock()):
    from src.title_optimizer import generate_titles


def _item(title: str = "GPT-5 launches", summary: str = "New record benchmarks") -> dict:
    return {"title": title, "summary": summary}


class TestGenerateTitles:
    def test_returns_list_of_strings(self):
        titles = ["This AI is 10x cheaper", "Is GPT-5 real?", "OpenAI's hidden surprise"]
        with patch("src.title_optimizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_response(
                json.dumps(titles)
            )
            result = generate_titles(_item())
        assert isinstance(result, list)
        assert all(isinstance(t, str) for t in result)

    def test_returns_n_titles(self):
        titles = [f"Title {i}" for i in range(5)]
        with patch("src.title_optimizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_response(
                json.dumps(titles)
            )
            result = generate_titles(_item(), n=5)
        assert len(result) == 5

    def test_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps(["AI beats humans at coding"]) + "\n```"
        with patch("src.title_optimizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_response(raw)
            result = generate_titles(_item())
        assert result == ["AI beats humans at coding"]

    def test_fallback_on_invalid_json(self):
        with patch("src.title_optimizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_response("not json")
            result = generate_titles(_item(title="Original Title"))
        # fallback returns original title
        assert result == ["Original Title"]

    def test_raises_on_api_error(self):
        with patch("src.title_optimizer.client") as mock_client:
            mock_client.chat.completions.create.side_effect = Exception("network")
            with pytest.raises(Exception, match="network"):
                generate_titles(_item(title="Fallback Title"))

    def test_non_list_response_falls_back(self):
        with patch("src.title_optimizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_response(
                json.dumps({"title": "wrong shape"})
            )
            result = generate_titles(_item(title="My Title"))
        assert result == ["My Title"]

    def test_empty_strings_filtered_out(self):
        titles = ["", "  ", "Valid title"]
        with patch("src.title_optimizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_response(
                json.dumps(titles)
            )
            result = generate_titles(_item())
        assert result == ["Valid title"]

    def test_records_usage_when_present(self):
        resp = _mock_response(json.dumps(["Title"]))
        resp.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        with patch("src.title_optimizer.client") as mock_client, \
             patch("src.title_optimizer.get_ledger") as mock_ledger:
            mock_client.chat.completions.create.return_value = resp
            generate_titles(_item())
        mock_ledger.return_value.record_llm.assert_called_once()


from src.title_optimizer import pick_best_title  # noqa: E402


class TestPickBestTitle:
    def test_returns_best_title_from_response(self):
        with patch("src.title_optimizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_response_str(
                "This AI beats GPT-4"
            )
            result = pick_best_title(["Title A", "Title B", "This AI beats GPT-4"])
        assert result == "This AI beats GPT-4"

    def test_falls_back_to_first_when_empty_response(self):
        with patch("src.title_optimizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_response_str("")
            result = pick_best_title(["Fallback Title"])
        assert result == "Fallback Title"

    def test_falls_back_to_ai_news_when_empty_list_and_empty_response(self):
        with patch("src.title_optimizer.client") as mock_client:
            mock_client.chat.completions.create.return_value = _mock_response_str("")
            result = pick_best_title([])
        assert result == "AI NEWS"

    def test_records_usage_when_present(self):
        resp = _mock_response_str("Best title")
        resp.usage = MagicMock(prompt_tokens=50, completion_tokens=5)
        with patch("src.title_optimizer.client") as mock_client, \
             patch("src.title_optimizer.get_ledger") as mock_ledger:
            mock_client.chat.completions.create.return_value = resp
            pick_best_title(["Some title"])
        mock_ledger.return_value.record_llm.assert_called_once()
