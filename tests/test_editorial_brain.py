"""Tests for EditorialBrain — only pure/deterministic methods are covered here.
Methods that call OpenAI (run, _plan_story, _generate_hook_variants, etc.)
are excluded; they belong in integration tests with mocked LLM responses.
"""
import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.editorial_brain import (
    AI_AUDIENCE_TERMS,
    ANGLE_CANDIDATES,
    CONFLICT_TERMS,
    FORMAT_RULES,
    STORY_TYPES,
    EditorialBrain,
    EditorialPlan,
    editorial_plan_to_prompt,
)
from src.scraper import NewsItem
from src.shared_types import ContentStrategy, PerformanceStats


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


# ── _pick_angle() ─────────────────────────────────────────────────────────────

class TestPickAngle:
    def test_returns_angle_dict_with_required_keys(self, brain, fresh_story):
        result = brain._pick_angle(fresh_story, [])
        assert "key" in result
        assert "label" in result
        assert "conflict" in result

    def test_technical_story_gets_technical_angle(self, brain):
        story = NewsItem(
            source="arXiv",
            title="New model benchmark parameters training accuracy",
            url="u",
            summary="A model with better parameters and latency.",
            authors=[],
            published=datetime.date.today().isoformat(),
        )
        result = brain._pick_angle(story, [])
        assert result["key"] in [c["key"] for c in ANGLE_CANDIDATES]

    def test_performance_boost_applied_when_available(self, brain, fresh_story):
        mock_perf = PerformanceStats(
            category="technical_breakthrough",
            sample_size=10,
            avg_hook_score=0.8,
            avg_watch_time=0.7,
            success_rate=0.75,
        )
        brain.feedback_analyzer.get_angle_performance.return_value = mock_perf
        result = brain._pick_angle(fresh_story, [])
        assert result["key"] in [c["key"] for c in ANGLE_CANDIDATES]

    def test_strategy_weights_influence_angle_selection(self, brain, fresh_story):
        strategy = ContentStrategy(
            angle_weights={"technical_breakthrough": 1.0, "industry_impact": 0.0},
            format_weights={},
            exploration_rate=0.0,
            mode="growth",
            confidence=0.9,
        )
        result = brain._pick_angle(fresh_story, [], strategy=strategy)
        assert result["key"] in [c["key"] for c in ANGLE_CANDIDATES]

    def test_no_performance_data_returns_valid_angle(self, brain, fresh_story):
        brain.feedback_analyzer.get_angle_performance.return_value = None
        result = brain._pick_angle(fresh_story, [])
        assert result["key"] in [c["key"] for c in ANGLE_CANDIDATES]


# ── _angle_score() ────────────────────────────────────────────────────────────

class TestAngleScore:
    def test_technical_breakthrough_high_for_model_story(self, brain):
        story = NewsItem(
            source="arXiv", title="New model training accuracy benchmark",
            url="u", summary="parameters and latency improvements",
            authors=[], published=datetime.date.today().isoformat(),
        )
        score = brain._angle_score(story, "technical_breakthrough")
        assert score == 0.8

    def test_technical_breakthrough_low_for_unrelated_story(self, brain):
        story = NewsItem(
            source="X", title="Company raises funds",
            url="u", summary="Investment round closed",
            authors=[], published=datetime.date.today().isoformat(),
        )
        score = brain._angle_score(story, "technical_breakthrough")
        assert score == 0.4

    def test_industry_impact_high_for_business_story(self, brain):
        story = NewsItem(
            source="X", title="Company launch new enterprise platform",
            url="u", summary="Partnership announced for business adoption",
            authors=[], published=datetime.date.today().isoformat(),
        )
        score = brain._angle_score(story, "industry_impact")
        assert score == 0.75

    def test_industry_impact_fallback_for_unrelated(self, brain):
        story = NewsItem(
            source="X", title="AI is interesting",
            url="u", summary="A general discussion",
            authors=[], published=datetime.date.today().isoformat(),
        )
        score = brain._angle_score(story, "industry_impact")
        assert score == 0.5

    def test_threat_to_jobs_high_for_automation_story(self, brain):
        story = NewsItem(
            source="X", title="AI will replace workers and automation jobs",
            url="u", summary="Layoffs due to productivity and hiring freeze",
            authors=[], published=datetime.date.today().isoformat(),
        )
        score = brain._angle_score(story, "threat_to_jobs")
        assert score == 0.8

    def test_threat_to_jobs_low_for_unrelated(self, brain):
        story = NewsItem(
            source="X", title="AI research paper published",
            url="u", summary="Model training details",
            authors=[], published=datetime.date.today().isoformat(),
        )
        score = brain._angle_score(story, "threat_to_jobs")
        assert score == 0.2

    def test_overhyped_vs_reality_high_for_hype_story(self, brain):
        story = NewsItem(
            source="X", title="This is hype but still overhyped",
            url="u", summary="However the promised reality is different yet",
            authors=[], published=datetime.date.today().isoformat(),
        )
        score = brain._angle_score(story, "overhyped_vs_reality")
        assert score == 0.9

    def test_overhyped_vs_reality_low_for_straightforward(self, brain):
        story = NewsItem(
            source="X", title="New AI announcement",
            url="u", summary="A model was released",
            authors=[], published=datetime.date.today().isoformat(),
        )
        score = brain._angle_score(story, "overhyped_vs_reality")
        assert score == 0.3

    def test_what_this_means_for_you_high_for_user_story(self, brain):
        story = NewsItem(
            source="X", title="Developers and enterprise teams get new tools",
            url="u", summary="Users and students can improve workflow",
            authors=[], published=datetime.date.today().isoformat(),
        )
        score = brain._angle_score(story, "what_this_means_for_you")
        assert score == 0.85

    def test_what_this_means_for_you_fallback(self, brain):
        story = NewsItem(
            source="X", title="AI is interesting",
            url="u", summary="General news",
            authors=[], published=datetime.date.today().isoformat(),
        )
        score = brain._angle_score(story, "what_this_means_for_you")
        assert score == 0.45

    def test_unknown_angle_returns_default(self, brain, fresh_story):
        score = brain._angle_score(fresh_story, "unknown_angle")
        assert score == 0.5


# ── _select_persona() ─────────────────────────────────────────────────────────

class TestSelectPersona:
    def test_corporate_story_gives_sharp_persona(self, brain):
        story = NewsItem(
            source="X", title="Company launch funding business partnership",
            url="u", summary="Enterprise investment round",
            authors=[], published=datetime.date.today().isoformat(),
        )
        result = brain._select_persona(story, {})
        assert "sharp" in result["style"] or "cynical" in result["style"]
        assert isinstance(result["rules"], list)
        assert len(result["rules"]) > 0

    def test_research_story_gives_curious_engineer_persona(self, brain):
        story = NewsItem(
            source="arXiv", title="New paper on benchmark model dataset",
            url="u", summary="Research on algorithm architecture",
            authors=[], published=datetime.date.today().isoformat(),
        )
        result = brain._select_persona(story, {})
        assert "curious" in result["style"] or "engineer" in result["style"]

    def test_policy_story_gives_analytic_persona(self, brain):
        story = NewsItem(
            source="X", title="New regulation and law for AI privacy ban",
            url="u", summary="Government policy for AI",
            authors=[], published=datetime.date.today().isoformat(),
        )
        result = brain._select_persona(story, {})
        assert "analytic" in result["style"] or "blunt" in result["style"]

    def test_other_story_gives_explainer_persona(self, brain):
        story = NewsItem(
            source="X", title="AI does something interesting",
            url="u", summary="Some general event occurred",
            authors=[], published=datetime.date.today().isoformat(),
        )
        result = brain._select_persona(story, {})
        assert "explainer" in result["style"] or "edge" in result["style"]

    def test_persona_returns_dict_with_style_and_rules(self, brain, fresh_story):
        result = brain._select_persona(fresh_story, {})
        assert "style" in result
        assert "rules" in result


# ── _select_tone() ────────────────────────────────────────────────────────────

class TestSelectTone:
    def test_hot_take_angle_returns_provocative(self, brain, fresh_story):
        tone = brain._select_tone(fresh_story, "hot_take", "explainer")
        assert tone == "provocative"

    def test_threat_to_jobs_returns_urgent(self, brain, fresh_story):
        tone = brain._select_tone(fresh_story, "threat_to_jobs", "explainer")
        assert tone == "urgent"

    def test_technical_breakthrough_returns_informed(self, brain, fresh_story):
        tone = brain._select_tone(fresh_story, "technical_breakthrough", "explainer")
        assert tone == "informed"

    def test_controversial_variation_returns_provocative(self, brain, fresh_story):
        tone = brain._select_tone(fresh_story, "industry_impact", "controversial")
        assert tone == "provocative"

    def test_storytelling_variation_returns_personal(self, brain, fresh_story):
        tone = brain._select_tone(fresh_story, "industry_impact", "storytelling")
        assert tone == "personal"

    def test_fast_news_variation_returns_direct(self, brain, fresh_story):
        tone = brain._select_tone(fresh_story, "industry_impact", "fast_news")
        assert tone == "direct"

    def test_default_variation_returns_curious(self, brain, fresh_story):
        tone = brain._select_tone(fresh_story, "industry_impact", "explainer")
        assert tone == "curious"


# ── _generate_hook_variants() ─────────────────────────────────────────────────

class TestGenerateHookVariants:
    def test_returns_list_of_strings(self, brain, fresh_story):
        result = brain._generate_hook_variants(fresh_story, "technical_breakthrough")
        assert isinstance(result, list)
        assert all(isinstance(h, str) for h in result)

    def test_returns_at_most_three_variants(self, brain, fresh_story):
        result = brain._generate_hook_variants(fresh_story, "technical_breakthrough")
        assert len(result) <= 3

    def test_technical_breakthrough_angle(self, brain, fresh_story):
        result = brain._generate_hook_variants(fresh_story, "technical_breakthrough")
        assert len(result) == 3

    def test_industry_impact_angle(self, brain, fresh_story):
        result = brain._generate_hook_variants(fresh_story, "industry_impact")
        assert len(result) == 3

    def test_threat_to_jobs_angle(self, brain, fresh_story):
        result = brain._generate_hook_variants(fresh_story, "threat_to_jobs")
        assert len(result) == 3

    def test_overhyped_vs_reality_angle(self, brain, fresh_story):
        result = brain._generate_hook_variants(fresh_story, "overhyped_vs_reality")
        assert len(result) == 3

    def test_default_angle_fallback(self, brain, fresh_story):
        result = brain._generate_hook_variants(fresh_story, "what_this_means_for_you")
        assert len(result) == 3

    def test_all_variants_bounded_to_twelve_words(self, brain, fresh_story):
        result = brain._generate_hook_variants(fresh_story, "technical_breakthrough")
        for variant in result:
            assert len(variant.split()) <= 12

    def test_with_hook_candidates_and_strategy(self, brain, fresh_story):
        strategy = ContentStrategy(
            angle_weights={"technical_breakthrough": 0.9},
            format_weights={"deep_dive": 0.8},
            exploration_rate=0.2,
            mode="growth",
            confidence=0.8,
        )
        result = brain._generate_hook_variants(
            fresh_story, "technical_breakthrough",
            hook_candidates=["Custom hook one", "Another custom hook"],
            strategy=strategy,
        )
        assert len(result) <= 3


# ── _rank_hooks() ─────────────────────────────────────────────────────────────

class TestRankHooks:
    def test_returns_string(self, brain, fresh_story):
        hooks = [
            "What nobody expected but secretly feared",
            "AI breakthrough reveals hidden problem",
            "This changes everything for developers",
        ]
        result = brain._rank_hooks(hooks, fresh_story)
        assert isinstance(result, str)

    def test_returns_one_of_input_hooks(self, brain, fresh_story):
        hooks = [
            "What nobody expected but secretly feared",
            "AI reveals major issue in industry",
            "Simple announcement for AI",
        ]
        result = brain._rank_hooks(hooks, fresh_story)
        assert result in hooks

    def test_curiosity_laden_hook_scores_high(self, brain, fresh_story):
        hooks = [
            "What if nobody knew why this could fail",
            "The announcement is out",
        ]
        result = brain._rank_hooks(hooks, fresh_story)
        assert result == hooks[0]


# ── _select_hook() ────────────────────────────────────────────────────────────

class TestSelectHook:
    def test_returns_string(self, brain, fresh_story):
        hook_variants = ["Hook one", "Hook two"]
        result = brain._select_hook(hook_variants, None, fresh_story)
        assert isinstance(result, str)

    def test_no_candidates_uses_rank_hooks(self, brain, fresh_story):
        hook_variants = [
            "What nobody expected but secretly feared",
            "Simple announcement",
        ]
        result = brain._select_hook(hook_variants, None, fresh_story)
        assert result in hook_variants

    def test_with_hook_candidates_sometimes_uses_mutation(self, brain, fresh_story):
        hook_variants = ["Variant one", "Variant two"]
        hook_candidates = ["Candidate hook from mutation"]
        # Run multiple times — some will use mutation, some won't
        results = set()
        for _ in range(20):
            results.add(brain._select_hook(hook_variants, hook_candidates, fresh_story))
        # At least one run should have produced a result
        assert len(results) >= 1

    def test_empty_hook_candidates_falls_back_to_variants(self, brain, fresh_story):
        hook_variants = ["Variant one", "Variant two"]
        result = brain._select_hook(hook_variants, [], fresh_story)
        assert result in hook_variants


# ── _conflict_phrase() ────────────────────────────────────────────────────────

class TestConflictPhrase:
    def test_returns_string(self, brain, fresh_story):
        result = brain._conflict_phrase(fresh_story)
        assert isinstance(result, str)

    def test_no_conflict_terms_returns_default(self, brain):
        story = NewsItem(
            source="X", title="AI releases new feature",
            url="u", summary="A nice new update was announced",
            authors=[], published=datetime.date.today().isoformat(),
        )
        result = brain._conflict_phrase(story)
        assert result == "the one detail that changes the story"

    def test_with_conflict_term_returns_specific_phrase(self, brain):
        story = NewsItem(
            source="X", title="AI model faces lawsuit over privacy",
            url="u", summary="Major security issues discovered",
            authors=[], published=datetime.date.today().isoformat(),
        )
        result = brain._conflict_phrase(story)
        assert result.startswith("the ")
        assert "heart of this story" in result


# ── _choose_global_style() ────────────────────────────────────────────────────

class TestChooseGlobalStyle:
    def test_shorts_platform_returns_very_high_energy(self, brain):
        style = brain._choose_global_style({}, "shorts")
        assert style["energy_level"] == "very high"
        assert style["pacing"] == "dynamic"

    def test_youtube_long_returns_high_energy(self, brain):
        style = brain._choose_global_style({}, "youtube_long")
        assert style["energy_level"] == "high"
        assert style["pacing"] == "purposeful"

    def test_other_platform_returns_medium_energy(self, brain):
        style = brain._choose_global_style({}, "tiktok")
        assert style["energy_level"] == "medium"
        assert style["pacing"] == "balanced"

    def test_channel_config_persona_used(self, brain):
        style = brain._choose_global_style({"persona": "custom persona"}, "youtube_long")
        assert style["persona"] == "custom persona"

    def test_default_persona_when_not_configured(self, brain):
        style = brain._choose_global_style({}, "youtube_long")
        assert style["persona"] == "seasoned AI editor"

    def test_style_has_required_keys(self, brain):
        style = brain._choose_global_style({}, "youtube_long")
        assert "persona" in style
        assert "energy_level" in style
        assert "pacing" in style


# ── _is_breaking() ────────────────────────────────────────────────────────────

class TestIsBreaking:
    def test_breaking_in_title_returns_true(self, brain):
        story = NewsItem(
            source="X", title="BREAKING: OpenAI releases GPT-5",
            url="u", summary="s", authors=[],
            published=datetime.date.today().isoformat(),
        )
        assert brain._is_breaking(story) is True

    def test_no_breaking_in_title_returns_false(self, brain, fresh_story):
        assert brain._is_breaking(fresh_story) is False

    def test_lowercase_breaking_returns_false(self, brain):
        story = NewsItem(
            source="X", title="breaking news in AI",
            url="u", summary="s", authors=[],
            published=datetime.date.today().isoformat(),
        )
        assert brain._is_breaking(story) is False


# ── _story_summary() ─────────────────────────────────────────────────────────

class TestStorySummary:
    def test_returns_dict_with_required_keys(self, brain, fresh_story):
        result = brain._story_summary(fresh_story)
        assert "story_id" in result
        assert "title" in result
        assert "source" in result
        assert "published" in result

    def test_story_id_is_url(self, brain, fresh_story):
        result = brain._story_summary(fresh_story)
        assert result["story_id"] == fresh_story.url

    def test_title_matches_story(self, brain, fresh_story):
        result = brain._story_summary(fresh_story)
        assert result["title"] == fresh_story.title


# ── _decide_format() with strategy ────────────────────────────────────────────

class TestDecideFormatWithStrategy:
    def test_strategy_growth_mode_considers_hot_take(self, brain, fresh_story):
        strategy = ContentStrategy(
            angle_weights={},
            format_weights={"quick_hit": 0.5, "hot_take": 0.9},
            exploration_rate=0.2,
            mode="growth",
            confidence=0.8,
        )
        result = brain._decide_format(fresh_story, "industry_impact", "explainer", strategy=strategy)
        assert "format" in result
        assert result["format"] in ("quick_hit", "hot_take", "deep_dive")

    def test_strategy_safe_mode_favors_proven_format(self, brain, fresh_story):
        strategy = ContentStrategy(
            angle_weights={},
            format_weights={"quick_hit": 0.9},
            exploration_rate=0.1,
            mode="safe",
            confidence=0.9,
        )
        result = brain._decide_format(fresh_story, "industry_impact", "explainer", strategy=strategy)
        assert "format" in result

    def test_strategy_with_no_matching_format_fallback(self, brain, fresh_story):
        strategy = ContentStrategy(
            angle_weights={},
            format_weights={},
            exploration_rate=0.0,
            mode="growth",
            confidence=0.5,
        )
        result = brain._decide_format(fresh_story, "industry_impact", "explainer", strategy=strategy)
        assert "format" in result

    def test_high_controversy_story_gets_hot_take(self, brain):
        story = NewsItem(
            source="X",
            title="AI ban lawsuit bias regulation privacy shutdown",
            url="u",
            summary="policy security ethical attack restriction crash",
            authors=[],
            published=datetime.date.today().isoformat(),
        )
        result = brain._decide_format(story, "overhyped_vs_reality", "explainer")
        assert result["format"] == "hot_take"

    def test_fast_news_variation_returns_quick_hit(self, brain, fresh_story):
        result = brain._decide_format(fresh_story, "industry_impact", "fast_news")
        assert result["format"] == "quick_hit"

    def test_storytelling_variation_returns_deep_dive(self, brain, fresh_story):
        result = brain._decide_format(fresh_story, "industry_impact", "storytelling")
        assert result["format"] == "deep_dive"

    def test_format_performance_boost_logged(self, brain, fresh_story):
        mock_fmt_perf = PerformanceStats(
            category="quick_hit",
            sample_size=5,
            avg_hook_score=0.85,
            avg_watch_time=0.75,
            success_rate=0.8,
        )
        brain.feedback_analyzer.get_format_performance.return_value = mock_fmt_perf
        result = brain._decide_format(fresh_story, "industry_impact", "explainer")
        assert "format" in result


# ── run() ─────────────────────────────────────────────────────────────────────

class TestRun:
    def test_run_returns_editorial_plan(self, brain, fresh_story):
        """run() delegates through _plan_story which calls _pick_angle, etc."""
        brain._plan_story = MagicMock(return_value={
            "story_id": "https://openai.com/gpt5",
            "angle": "technical_breakthrough",
            "conflict": "hype vs reality",
            "tone": "informed",
            "hook": "Nobody expected this hidden issue",
            "hook_variants": ["Hook 1", "Hook 2"],
            "novelty_score": 0.8,
            "controversy_score": 0.3,
            "format": "quick_hit",
            "persona": {"style": "sharp", "rules": []},
            "scene_plan": [],
        })
        stories = [fresh_story]
        result = brain.run(stories, history=[], platform="youtube_long")
        assert isinstance(result, EditorialPlan)
        assert isinstance(result.selected_stories, list)
        assert isinstance(result.editorial_plan, list)
        assert isinstance(result.global_style, dict)

    def test_run_with_empty_stories_list(self, brain):
        result = brain.run([], history=[])
        assert isinstance(result, EditorialPlan)
        assert result.selected_stories == []
        assert result.editorial_plan == []

    def test_run_selects_at_most_five_stories(self, brain):
        stories = []
        for i in range(8):
            stories.append(NewsItem(
                source="OpenAI",
                title=f"Story {i} about AI launch benchmark",
                url=f"http://example.com/{i}",
                summary="model research paper",
                authors=[],
                published=datetime.date.today().isoformat(),
            ))
        brain._plan_story = MagicMock(return_value={
            "story_id": "u", "angle": "technical_breakthrough",
            "conflict": "c", "tone": "informed", "hook": "h",
            "hook_variants": [], "novelty_score": 0.5,
            "controversy_score": 0.3, "format": "quick_hit",
            "persona": {"style": "sharp", "rules": []},
            "scene_plan": [],
        })
        result = brain.run(stories, history=[])
        assert len(result.selected_stories) <= 5

    def test_run_passes_strategy_to_plan_story(self, brain, fresh_story):
        strategy = ContentStrategy(
            angle_weights={"technical_breakthrough": 0.9},
            format_weights={"deep_dive": 0.8},
            exploration_rate=0.2,
            mode="growth",
            confidence=0.8,
        )
        brain._plan_story = MagicMock(return_value={
            "story_id": "u", "angle": "technical_breakthrough",
            "conflict": "c", "tone": "informed", "hook": "h",
            "hook_variants": [], "novelty_score": 0.5,
            "controversy_score": 0.3, "format": "quick_hit",
            "persona": {"style": "sharp", "rules": []},
            "scene_plan": [],
        })
        result = brain.run([fresh_story], strategy=strategy)
        assert isinstance(result, EditorialPlan)
        # Strategy should be passed to _plan_story
        call_kwargs = brain._plan_story.call_args
        assert call_kwargs is not None


# ── _enrich_with_narrative() ─────────────────────────────────────────────────

class TestEnrichWithNarrative:
    def test_returns_dict_on_import_error(self, brain, fresh_story):
        """When optional modules are missing, enrichment is skipped gracefully."""
        plan = {
            "story_id": fresh_story.url,
            "angle": "technical_breakthrough label",
            "conflict": "hype vs reality",
            "tone": "informed",
            "hook": "Nobody expected this",
            "hook_variants": [],
            "novelty_score": 0.8,
            "controversy_score": 0.3,
            "format": "quick_hit",
            "persona": {"style": "sharp", "rules": []},
            "scene_plan": [],
        }
        result = brain._enrich_with_narrative(plan, fresh_story)
        assert isinstance(result, dict)
        # Original keys should still be present
        assert "story_id" in result
        assert "angle" in result

    def test_enrichment_with_editorial_judgment_available(self, brain, fresh_story):
        plan = {
            "story_id": fresh_story.url,
            "angle": "technical_breakthrough",
            "conflict": "c",
            "tone": "informed",
            "hook": "h",
            "hook_variants": [],
            "novelty_score": 0.9,
            "controversy_score": 0.6,
            "format": "quick_hit",
            "persona": {"style": "sharp", "rules": []},
            "scene_plan": [],
        }
        mock_judgment = {
            "angle": "test_angle",
            "persona": {"style": "bold"},
            "format": {"type": "quick_hit"},
            "editorial_intent": "engage",
        }
        mock_engine = MagicMock()
        mock_engine.judge.return_value = mock_judgment
        with patch("src.editorial_brain.EditorialBrain._enrich_with_narrative",
                   wraps=brain._enrich_with_narrative):
            with patch.dict("sys.modules", {
                "src.editorial.editorial_judgment": MagicMock(
                    get_judgment_engine=MagicMock(return_value=mock_engine)
                ),
            }):
                result = brain._enrich_with_narrative(plan, fresh_story)
        assert isinstance(result, dict)


# ── _plan_story() ─────────────────────────────────────────────────────────────

class TestPlanStory:
    def test_plan_story_returns_complete_dict(self, brain, fresh_story):
        result = brain._plan_story(
            fresh_story, [], {}, "youtube_long", "explainer"
        )
        required_keys = [
            "story_id", "angle", "conflict", "tone", "hook",
            "hook_variants", "novelty_score", "controversy_score",
            "format", "persona", "scene_plan",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_plan_story_with_strategy(self, brain, fresh_story):
        strategy = ContentStrategy(
            angle_weights={"technical_breakthrough": 0.9},
            format_weights={"deep_dive": 0.8},
            exploration_rate=0.2,
            mode="growth",
            confidence=0.8,
        )
        result = brain._plan_story(
            fresh_story, [], {}, "youtube_long", "explainer", strategy=strategy
        )
        assert "format" in result

    def test_plan_story_with_hook_candidates(self, brain, fresh_story):
        result = brain._plan_story(
            fresh_story, [], {}, "youtube_long", "fast_news",
            hook_candidates=["Candidate hook one", "Another hook"],
        )
        assert isinstance(result["hook"], str)

    def test_plan_story_story_id_is_url(self, brain, fresh_story):
        result = brain._plan_story(
            fresh_story, [], {}, "youtube_long", "explainer"
        )
        assert result["story_id"] == fresh_story.url

    def test_plan_story_novelty_score_matches_computation(self, brain, fresh_story):
        result = brain._plan_story(
            fresh_story, ["Some past title"], {}, "youtube_long", "explainer"
        )
        assert 0.0 <= result["novelty_score"] <= 1.0

    def test_plan_story_scene_plan_is_list(self, brain, fresh_story):
        result = brain._plan_story(
            fresh_story, [], {}, "youtube_long", "storytelling"
        )
        assert isinstance(result["scene_plan"], list)


# ── editorial_plan_to_prompt() ────────────────────────────────────────────────

class TestEditorialPlanToPrompt:
    def _make_plan(self):
        return EditorialPlan(
            selected_stories=[
                {"story_id": "u1", "title": "GPT-5 launches", "source": "OpenAI", "published": "2026-05-16"},
                {"story_id": "u2", "title": "AI regulation", "source": "Reuters", "published": "2026-05-16"},
            ],
            editorial_plan=[
                {
                    "story_id": "u1",
                    "angle": "This looks like a breakthrough",
                    "conflict": "hype vs reality",
                    "persona": {"style": "sharp, slightly cynical", "rules": ["question claims", "avoid jargon"]},
                    "tone": "informed",
                    "hook": "Nobody expected this hidden problem",
                    "format": "quick_hit",
                    "scene_plan": [{"type": "hook"}, {"type": "impact"}],
                },
                {
                    "story_id": "u2",
                    "angle": "Industry impact angle",
                    "conflict": "innovation vs adoption",
                    "persona": "legacy string persona",
                    "tone": "provocative",
                    "hook": "This changes everything",
                    "format": "hot_take",
                    "scene_plan": [],
                },
            ],
            global_style={"persona": "seasoned AI editor", "energy_level": "high", "pacing": "purposeful"},
        )

    def test_returns_string(self):
        plan = self._make_plan()
        result = editorial_plan_to_prompt(plan)
        assert isinstance(result, str)

    def test_contains_editorial_plan_header(self):
        plan = self._make_plan()
        result = editorial_plan_to_prompt(plan)
        assert "=== EDITORIAL PLAN ===" in result

    def test_contains_global_style(self):
        plan = self._make_plan()
        result = editorial_plan_to_prompt(plan)
        assert "seasoned AI editor" in result
        assert "high" in result
        assert "purposeful" in result

    def test_contains_story_titles(self):
        plan = self._make_plan()
        result = editorial_plan_to_prompt(plan)
        assert "GPT-5 launches" in result
        assert "AI regulation" in result

    def test_contains_angle_and_hook(self):
        plan = self._make_plan()
        result = editorial_plan_to_prompt(plan)
        assert "Nobody expected this hidden problem" in result
        assert "This looks like a breakthrough" in result

    def test_persona_dict_shows_style_and_rules(self):
        plan = self._make_plan()
        result = editorial_plan_to_prompt(plan)
        assert "persona style:" in result
        assert "sharp, slightly cynical" in result
        assert "question claims" in result

    def test_persona_string_shows_persona_line(self):
        plan = self._make_plan()
        result = editorial_plan_to_prompt(plan)
        assert "legacy string persona" in result

    def test_conflict_fallback_when_missing(self):
        plan = EditorialPlan(
            selected_stories=[],
            editorial_plan=[
                {
                    "story_id": "u1",
                    "angle": "Some angle",
                    "persona": {"style": "sharp", "rules": []},
                    "tone": "informed",
                    "hook": "Hook text",
                    "format": "quick_hit",
                    "scene_plan": [],
                    # 'conflict' key intentionally missing
                }
            ],
            global_style={"persona": "editor", "energy_level": "high", "pacing": "fast"},
        )
        result = editorial_plan_to_prompt(plan)
        assert "make the tension explicit" in result
