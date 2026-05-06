"""Tests for EditorialBrain — only pure/deterministic methods are covered here.
Methods that call OpenAI (run, _plan_story, _generate_hook_variants, etc.)
are excluded; they belong in integration tests with mocked LLM responses.
"""
import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.editorial_brain import EditorialBrain, FORMAT_RULES
from src.scraper import NewsItem


@pytest.fixture
def brain():
    """EditorialBrain with mocked LLM and FeedbackAnalyzer (no IO, no API calls)."""
    with patch("src.editorial_brain.FeedbackAnalyzer") as mock_fa_cls:
        mock_fa = mock_fa_cls.return_value
        mock_fa.get_angle_performance.return_value = {}
        mock_fa.get_format_performance.return_value = {}
        yield EditorialBrain(llm=MagicMock())


# ── _hype_score() ─────────────────────────────────────────────────────────────

class TestHypeScore:
    def test_high_hype_source(self, brain, fresh_story):
        fresh_story.source = "HackerNews"
        assert brain._hype_score(fresh_story) > 0.5

    def test_low_hype_source(self, brain, fresh_story):
        fresh_story.source = "some_blog"
        score = brain._hype_score(fresh_story)
        assert 0.0 <= score <= 1.0

    def test_buzz_word_increases_score(self, brain, fresh_story):
        fresh_story.source = "some_blog"
        fresh_story.title = "Company launches new product"
        score_with_buzz = brain._hype_score(fresh_story)

        fresh_story.title = "Company releases quarterly report"
        score_without_buzz = brain._hype_score(fresh_story)

        assert score_with_buzz >= score_without_buzz

    def test_score_bounded(self, brain, fresh_story):
        fresh_story.source = "OpenAI"
        fresh_story.title = "OpenAI announces breaking launch revealed viral buzz"
        assert 0.0 <= brain._hype_score(fresh_story) <= 1.0


# ── _novelty_score() ──────────────────────────────────────────────────────────

class TestNoveltyScore:
    def test_empty_history_returns_high_score(self, brain, fresh_story):
        assert brain._novelty_score(fresh_story, []) == 0.8

    def test_identical_title_returns_low_score(self, brain, fresh_story):
        history = [fresh_story.title]
        score = brain._novelty_score(fresh_story, history)
        assert score < 0.3

    def test_unrelated_title_returns_high_score(self, brain):
        story = NewsItem(
            source="arXiv", title="Quantum computing breakthrough", url="u",
            summary="s", authors=[], published=datetime.date.today().isoformat(),
        )
        history = ["GPT-5 launches with record benchmarks"]
        assert brain._novelty_score(story, history) > 0.7

    def test_score_bounded(self, brain, fresh_story):
        history = ["completely unrelated topic about cooking"]
        score = brain._novelty_score(fresh_story, history)
        assert 0.0 <= score <= 1.0


# ── _controversy_score() ──────────────────────────────────────────────────────

class TestControversyScore:
    def test_no_conflict_terms_returns_zero(self, brain, fresh_story):
        fresh_story.title = "New model performs well on benchmarks"
        fresh_story.summary = "Good results overall."
        assert brain._controversy_score(fresh_story) == 0.0

    def test_conflict_terms_raise_score(self, brain, controversy_story):
        assert brain._controversy_score(controversy_story) > 0.0

    def test_multiple_terms_cap_at_one(self, brain):
        story = NewsItem(
            source="X", title="ban lawsuit bias attack crash restriction delay shutdown",
            url="u", summary="policy regulation security ethical privacy misinformation",
            authors=[], published=datetime.date.today().isoformat(),
        )
        assert brain._controversy_score(story) == 1.0


# ── _audience_fit() ───────────────────────────────────────────────────────────

class TestAudienceFit:
    def test_ai_terms_increase_score(self, brain, technical_story):
        score = brain._audience_fit(technical_story, "youtube_long")
        assert score > 0.0

    def test_platform_boost_for_youtube(self, brain, technical_story):
        youtube_score = brain._audience_fit(technical_story, "youtube_long")
        other_score = brain._audience_fit(technical_story, "unknown_platform")
        assert youtube_score >= other_score

    def test_score_bounded(self, brain, fresh_story):
        assert 0.0 <= brain._audience_fit(fresh_story, "shorts") <= 1.0


# ── _recency_score() ──────────────────────────────────────────────────────────

class TestRecencyScore:
    def test_today_scores_near_one(self, brain, fresh_story):
        assert brain._recency_score(fresh_story) > 0.8

    def test_old_story_scores_zero(self, brain, old_story):
        assert brain._recency_score(old_story) == 0.0

    def test_invalid_date_uses_fallback(self, brain, fresh_story):
        fresh_story.published = "not-a-date"
        score = brain._recency_score(fresh_story)
        assert 0.0 <= score <= 1.0


# ── _hook_curiosity / _hook_negativity / _hook_surprise ──────────────────────

class TestHookScorers:
    def test_curiosity_increases_with_question_words(self, brain):
        low = brain._hook_curiosity("AI is getting faster")
        high = brain._hook_curiosity("What if AI could think?")
        assert high > low

    def test_negativity_increases_with_negative_words(self, brain):
        low = brain._hook_negativity("Great results from the new model")
        high = brain._hook_negativity("But this model still fails to avoid the problem")
        assert high > low

    def test_surprise_increases_with_surprise_words(self, brain):
        low = brain._hook_surprise("A new AI model was announced today")
        high = brain._hook_surprise("Nobody expected this secret hidden feature")
        assert high > low

    def test_all_scores_bounded_zero_to_one(self, brain):
        hook = "What nobody secretly expected but could avoid if you think why"
        assert 0.0 <= brain._hook_curiosity(hook) <= 1.0
        assert 0.0 <= brain._hook_negativity(hook) <= 1.0
        assert 0.0 <= brain._hook_surprise(hook) <= 1.0


# ── _normalize_hook() ─────────────────────────────────────────────────────────

class TestNormalizeHook:
    def test_truncates_to_12_words(self, brain):
        long_hook = " ".join(f"word{i}" for i in range(20))
        result = brain._normalize_hook(long_hook)
        assert len(result.split()) == 12

    def test_short_hook_unchanged(self, brain):
        short = "This is a good hook"
        assert brain._normalize_hook(short) == short

    def test_collapses_whitespace(self, brain):
        messy = "  This   has   extra   spaces  "
        result = brain._normalize_hook(messy)
        assert "  " not in result
        assert result == result.strip()


# ── _tokens() and _jaccard() ─────────────────────────────────────────────────

class TestTokensAndJaccard:
    def test_tokens_filters_short_words(self, brain):
        # _tokens uses \w{3,} — filters words shorter than 3 chars
        tokens = brain._tokens("AI is the best model")
        assert "ai" not in tokens   # 2 chars
        assert "is" not in tokens   # 2 chars
        assert "model" in tokens    # 5 chars
        assert "best" in tokens     # 4 chars

    def test_jaccard_identical_sets(self, brain):
        t = brain._tokens("deep learning model")
        assert brain._jaccard(t, t) == 1.0

    def test_jaccard_disjoint_sets(self, brain):
        a = brain._tokens("machine learning algorithm")
        b = brain._tokens("cooking dinner recipes tonight")
        assert brain._jaccard(a, b) == 0.0

    def test_jaccard_partial_overlap(self, brain):
        a = brain._tokens("new language model released")
        b = brain._tokens("new benchmark results released today")
        score = brain._jaccard(a, b)
        assert 0.0 < score < 1.0

    def test_jaccard_empty_sets(self, brain):
        assert brain._jaccard(set(), {"token"}) == 0.0
        assert brain._jaccard({"token"}, set()) == 0.0


# ── _complexity_score() ───────────────────────────────────────────────────────

class TestComplexityScore:
    def test_technical_terms_raise_score(self, brain, technical_story):
        score = brain._complexity_score(technical_story)
        assert score > 0.0

    def test_simple_story_scores_low(self, brain):
        story = NewsItem(
            source="X", title="AI company raises money", url="u",
            summary="A startup got funding.", authors=[],
            published=datetime.date.today().isoformat(),
        )
        assert brain._complexity_score(story) == 0.0

    def test_score_bounded(self, brain, technical_story):
        assert 0.0 <= brain._complexity_score(technical_story) <= 1.0


# ── rank_stories() ────────────────────────────────────────────────────────────

class TestRankStories:
    def test_returns_same_count_as_input(self, brain, fresh_story, old_story):
        ranked = brain.rank_stories([fresh_story, old_story], [], "youtube_long")
        assert len(ranked) == 2

    def test_high_value_story_ranked_first(self, brain):
        high = NewsItem(
            source="OpenAI",
            title="OpenAI announces breakthrough launch revealed",
            url="u1", summary="developer api benchmark model research",
            authors=[], published=datetime.date.today().isoformat(),
        )
        low = NewsItem(
            source="unknown_blog",
            title="Old company update",
            url="u2", summary="Nothing interesting.",
            authors=[],
            published=(datetime.date.today() - datetime.timedelta(days=10)).isoformat(),
        )
        ranked = brain.rank_stories([low, high], [], "youtube_long")
        assert ranked[0].url == "u1"

    def test_empty_story_list_returns_empty(self, brain):
        assert brain.rank_stories([], [], "youtube_long") == []


# ── _decide_format() ─────────────────────────────────────────────────────────

class TestDecideFormat:
    def test_breaking_story_gets_quick_hit(self, brain):
        story = NewsItem(
            source="X", title="BREAKING: GPT-5 is here", url="u",
            summary="s", authors=[], published=datetime.date.today().isoformat(),
        )
        result = brain._decide_format(story, "technical_breakthrough", "fast_news")
        assert result["format"] == "quick_hit"

    def test_complex_story_gets_deep_dive(self, brain, technical_story):
        technical_story.summary = (
            "New benchmark tests architecture parameters fine-tuning dataset latency throughput model size training."
        )
        result = brain._decide_format(technical_story, "technical_breakthrough", "explainer")
        assert result["format"] == "deep_dive"

    def test_result_has_required_keys(self, brain, fresh_story):
        result = brain._decide_format(fresh_story, "industry_impact", "explainer")
        assert "format" in result
        assert "scenes" in result
        assert "pacing" in result


# ── _director_bridge() ────────────────────────────────────────────────────────

class TestDirectorBridge:
    def test_returns_list_of_scene_steps(self, brain):
        format_data = {"format": "quick_hit", "scenes": 5, "pacing": "fast"}
        plan = brain._director_bridge(format_data, "technical_breakthrough", "explainer")
        assert isinstance(plan, list)
        assert len(plan) > 0

    def test_each_step_has_angle(self, brain):
        format_data = {"format": "deep_dive", "scenes": 10, "pacing": "slow"}
        plan = brain._director_bridge(format_data, "industry_impact", "explainer")
        for step in plan:
            assert step["angle"] == "industry_impact"

    def test_storytelling_variation_adds_story_step(self, brain):
        format_data = {"format": "deep_dive", "scenes": 10, "pacing": "slow"}
        plan = brain._director_bridge(format_data, "industry_impact", "storytelling")
        types = [s["type"] for s in plan]
        assert "story" in types

    def test_controversial_hot_take_adds_challenge_step(self, brain):
        format_data = {"format": "hot_take", "scenes": 4, "pacing": "fast"}
        plan = brain._director_bridge(format_data, "overhyped_vs_reality", "controversial")
        types = [s["type"] for s in plan]
        assert "challenge" in types
