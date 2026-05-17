"""Tests for viral_selector — keyword scoring, boring detection, and pick logic."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# Stub elevenlabs before project imports
sys.modules.setdefault("elevenlabs", MagicMock())
sys.modules.setdefault("elevenlabs.client", MagicMock())

from src.viral_selector import keyword_score, is_boring, pick_viral_news, gpt_score
from src.scraper import NewsItem


def _item(title: str, summary: str = "", source: str = "test") -> NewsItem:
    return NewsItem(
        source=source,
        title=title,
        url="https://example.com",
        summary=summary,
        authors=[],
        published="2026-01-01",
    )


# ── keyword_score ─────────────────────────────────────────────────────────────

class TestKeywordScore:
    def test_high_impact_keyword_adds_2(self):
        item = _item("OpenAI will replace junior developers")
        score = keyword_score(item)
        # "replace" keyword → +2; "OpenAI" company → +3
        assert score >= 5

    def test_company_match_adds_3(self):
        item = _item("Google releases new model")
        score = keyword_score(item)
        assert score >= 3

    def test_multiple_keywords_accumulate(self):
        item = _item("OpenAI launches breakthrough model faster than GPT-4")
        # "launch", "breakthrough", "faster", "OpenAI" → big score
        score = keyword_score(item)
        assert score >= 8

    def test_boring_academic_gets_zero_or_low(self):
        item = _item("Minor configuration update in test suite")
        score = keyword_score(item)
        assert score < 5

    def test_short_text_bonus(self):
        short = _item("AI kills jobs", "")
        long  = _item("AI kills jobs", "x" * 400)
        # short text (< 300 chars combined) gets +1 bonus
        assert keyword_score(short) > keyword_score(long)

    def test_case_insensitive(self):
        lower = _item("openai launches model")
        upper = _item("OpenAI Launches Model")
        assert keyword_score(lower) == keyword_score(upper)

    def test_summary_also_scored(self):
        item = _item("Interesting update", summary="OpenAI releases breakthrough model")
        assert keyword_score(item) >= 5

    def test_returns_int(self):
        assert isinstance(keyword_score(_item("test")), int)


# ── is_boring ─────────────────────────────────────────────────────────────────

class TestIsBoring:
    def test_paper_in_title_is_boring(self):
        assert is_boring(_item("New paper on language model alignment"))

    def test_study_in_title_is_boring(self):
        assert is_boring(_item("Study finds AI adoption rising"))

    def test_dataset_in_title_is_boring(self):
        assert is_boring(_item("Researchers release new dataset for benchmarking"))

    def test_exciting_title_not_boring(self):
        assert not is_boring(_item("OpenAI launches GPT-5"))

    def test_case_sensitive_boring_check(self):
        # boring check is case-insensitive via .lower()
        assert is_boring(_item("PAPER: new findings in LLM training"))

    def test_empty_title_not_boring(self):
        assert not is_boring(_item(""))


# ── gpt_score (mocked) ────────────────────────────────────────────────────────

class TestGptScore:
    def _make_response(self, content: str):
        resp = MagicMock()
        resp.choices[0].message.content = content
        resp.usage = None
        return resp

    def test_returns_parsed_integer(self):
        with patch("src.viral_selector.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                self._make_response("8")
            )
            score = gpt_score(_item("OpenAI GPT-5 launches"))
        assert score == 8

    def test_clamps_score_between_1_and_10(self):
        with patch("src.viral_selector.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                self._make_response("99")
            )
            score = gpt_score(_item("test"))
        assert score == 10

    def test_returns_0_on_api_failure(self):
        with patch("src.viral_selector.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.side_effect = Exception("network")
            score = gpt_score(_item("test"))
        assert score == 0

    def test_returns_0_on_non_numeric_response(self):
        with patch("src.viral_selector.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                self._make_response("sorry, I cannot rate this")
            )
            score = gpt_score(_item("test"))
        assert score == 0

    def test_parses_score_embedded_in_text(self):
        with patch("src.viral_selector.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = (
                self._make_response("I would rate this 7 out of 10")
            )
            score = gpt_score(_item("test"))
        assert score == 7


# ── pick_viral_news ───────────────────────────────────────────────────────────

class TestPickViralNews:
    def _gpt_returns(self, n: int):
        resp = MagicMock()
        resp.choices[0].message.content = str(n)
        resp.usage = None
        return resp

    def test_returns_top_n_items(self):
        items = [
            _item("OpenAI launches breakthrough model"),
            _item("Google AGI released"),
            _item("Meta AI faster than GPT"),
            _item("New paper on dataset curation"),  # boring — filtered out
        ]
        with patch("src.viral_selector.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = self._gpt_returns(7)
            result = pick_viral_news(items, top_n=3)
        # boring item excluded; at most 3 returned
        assert len(result) <= 3
        # boring item should not appear
        assert not any("paper" in i.title.lower() for i in result)

    def test_empty_list_returns_empty(self):
        assert pick_viral_news([]) == []

    def test_all_boring_returns_empty(self):
        items = [_item("paper on datasets"), _item("study about LLMs")]
        result = pick_viral_news(items)
        assert result == []

    def test_respects_top_n_cap(self):
        items = [_item(f"OpenAI breakthrough model {i}") for i in range(10)]
        with patch("src.viral_selector.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = self._gpt_returns(8)
            result = pick_viral_news(items, top_n=2)
        assert len(result) <= 2


class TestViralSelectorUsageLogging:
    def _gpt_returns(self, score: int):
        resp = MagicMock()
        resp.choices[0].message.content = str(score)
        resp.usage = MagicMock()
        resp.usage.prompt_tokens = 50
        resp.usage.completion_tokens = 10
        return resp

    def test_records_llm_usage_when_present(self):
        from src.viral_selector import pick_viral_news
        items = [_item("OpenAI launches GPT-5 with record benchmarks")]
        with patch("src.viral_selector.make_openai_client") as mock_client:
            mock_client.return_value.chat.completions.create.return_value = self._gpt_returns(8)
            with patch("src.viral_selector.get_ledger") as mock_ledger:
                pick_viral_news(items)
        mock_ledger.return_value.record_llm.assert_called()
