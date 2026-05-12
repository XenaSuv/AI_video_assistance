"""Tests for src/editorial/ — the deterministic editorial judgment layer."""
from __future__ import annotations

import json
import pytest

from src.editorial.editorial_identity import (
    EDITORIAL_IDENTITY,
    EditorialIdentity,
)
from src.editorial.story_ranker import (
    StoryRanker,
    _ANTI_HYPE_BONUS,
    _CONTROVERSY_BONUS,
    _PRIORITY_BONUS,
    _SEEN_PENALTY,
)
from src.editorial.angle_selector import (
    ANGLE_INTENT,
    ANGLES,
    AngleSelector,
)
from src.editorial.persona_mapper import (
    ANGLE_PERSONA,
    PERSONAS,
    PersonaMapper,
)
from src.editorial.format_selector import (
    ANGLE_FORMAT,
    FORMATS,
    FormatSelector,
)
from src.editorial.editorial_judgment import (
    EditorialJudgment,
    JudgmentFeedback,
    _RETENTION_WINNER_THRESHOLD,
    get_judgment_engine,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _identity() -> EditorialIdentity:
    return EditorialIdentity.default()


def _engine(tmp_path) -> EditorialJudgment:
    return EditorialJudgment(data_dir=tmp_path)


def _story(**kwargs) -> dict:
    defaults = {
        "importance": 5.0,
        "controversy": 0.0,
        "hype": False,
        "risk": False,
        "tool": False,
        "seen_recently": False,
        "misconception": False,
        "impact": False,
    }
    defaults.update(kwargs)
    return defaults


# ── EditorialIdentity ────────────────────────────────────────────────────────

class TestEditorialIdentity:
    def test_default_voice_is_direct(self):
        assert _identity().voice == "direct"

    def test_default_tone_is_skeptical(self):
        assert _identity().tone == "skeptical"

    def test_default_bias_is_anti_hype(self):
        assert _identity().bias == "anti-hype"

    def test_default_priorities_nonempty(self):
        assert len(_identity().priorities) > 0

    def test_hidden_risks_in_default_priorities(self):
        assert "hidden_risks" in _identity().priorities

    def test_default_avoid_nonempty(self):
        assert len(_identity().avoid) > 0

    def test_signature_patterns_nonempty(self):
        assert len(_identity().signature_patterns) > 0

    def test_to_dict_has_required_keys(self):
        d = _identity().to_dict()
        for key in ("voice", "tone", "bias", "priorities", "avoid", "signature_patterns"):
            assert key in d

    def test_from_dict_roundtrip(self):
        original = _identity()
        restored = EditorialIdentity.from_dict(original.to_dict())
        assert restored.voice == original.voice
        assert restored.priorities == original.priorities

    def test_promote_priority_moves_to_front(self):
        ident = _identity()
        ident.priorities = ["a", "b", "hidden_risks"]
        ident.promote_priority("hidden_risks")
        assert ident.priorities[0] == "hidden_risks"

    def test_promote_priority_deduplicates(self):
        ident = _identity()
        ident.priorities = ["hidden_risks", "misconceptions"]
        ident.promote_priority("hidden_risks")
        assert ident.priorities.count("hidden_risks") == 1

    def test_add_priority_appends_new(self):
        ident = _identity()
        ident.add_priority("brand_new_priority")
        assert "brand_new_priority" in ident.priorities

    def test_add_priority_skips_existing(self):
        ident = _identity()
        before = len(ident.priorities)
        ident.add_priority(ident.priorities[0])
        assert len(ident.priorities) == before

    def test_top_priority_returns_first(self):
        ident = _identity()
        ident.priorities = ["first", "second"]
        assert ident.top_priority() == "first"

    def test_save_and_load_roundtrip(self, tmp_path):
        ident = _identity()
        ident.voice = "measured"
        ident.save(tmp_path)
        loaded = EditorialIdentity.load(tmp_path)
        assert loaded.voice == "measured"

    def test_load_missing_file_returns_default(self, tmp_path):
        loaded = EditorialIdentity.load(tmp_path)
        assert loaded.voice == EDITORIAL_IDENTITY["voice"]

    def test_editorial_identity_constant_has_all_keys(self):
        for key in ("voice", "tone", "bias", "priorities", "avoid", "signature_patterns"):
            assert key in EDITORIAL_IDENTITY


# ── StoryRanker ───────────────────────────────────────────────────────────────

class TestStoryRanker:
    def test_returns_sorted_descending(self):
        ranker = StoryRanker()
        stories = [_story(importance=1), _story(importance=9), _story(importance=5)]
        ranked = ranker.rank(stories, _identity())
        scores = [s for s, _ in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_higher_importance_ranks_first(self):
        ranker = StoryRanker()
        low  = _story(importance=1)
        high = _story(importance=9)
        ranked = ranker.rank([low, high], _identity())
        assert ranked[0][1] is high

    def test_controversy_adds_bonus(self):
        ranker = StoryRanker()
        ident  = _identity()
        plain = _story(importance=5, controversy=0)
        hot   = _story(importance=5, controversy=0.8)
        ranked = ranker.rank([plain, hot], ident)
        assert ranked[0][1] is hot

    def test_controversy_bonus_value(self):
        ranker = StoryRanker()
        ident  = _identity()
        score_no  = ranker._score(_story(importance=5, controversy=0), ident)
        score_yes = ranker._score(_story(importance=5, controversy=0.8), ident)
        assert score_yes - score_no == _CONTROVERSY_BONUS

    def test_anti_hype_bias_boosts_hype_story(self):
        ranker = StoryRanker()
        ident  = _identity()   # bias == "anti-hype"
        plain = _story(importance=5, hype=False)
        hyped = _story(importance=5, hype=True)
        ranked = ranker.rank([plain, hyped], ident)
        assert ranked[0][1] is hyped

    def test_anti_hype_bonus_value(self):
        ranker = StoryRanker()
        ident  = _identity()
        score_no  = ranker._score(_story(importance=5, hype=False), ident)
        score_yes = ranker._score(_story(importance=5, hype=True), ident)
        assert score_yes - score_no == _ANTI_HYPE_BONUS

    def test_seen_recently_penalised(self):
        ranker = StoryRanker()
        fresh = _story(importance=5, seen_recently=False)
        stale = _story(importance=5, seen_recently=True)
        ranked = ranker.rank([fresh, stale], _identity())
        assert ranked[0][1] is fresh

    def test_seen_recently_penalty_value(self):
        ranker = StoryRanker()
        ident  = _identity()
        score_fresh = ranker._score(_story(importance=5, seen_recently=False), ident)
        score_stale = ranker._score(_story(importance=5, seen_recently=True), ident)
        assert score_fresh - score_stale == _SEEN_PENALTY

    def test_priority_match_adds_bonus(self):
        ranker = StoryRanker()
        ident  = _identity()
        base   = ranker._score(_story(importance=5, risk=False), ident)
        risky  = ranker._score(_story(importance=5, risk=True), ident)
        assert risky >= base + _PRIORITY_BONUS

    def test_empty_stories_returns_empty(self):
        ranker = StoryRanker()
        assert ranker.rank([], _identity()) == []

    def test_single_story_returned(self):
        ranker = StoryRanker()
        result = ranker.rank([_story()], _identity())
        assert len(result) == 1


# ── AngleSelector ────────────────────────────────────────────────────────────

class TestAngleSelector:
    def test_hype_story_gets_hype_vs_reality(self):
        sel = AngleSelector()
        assert sel.select(_story(hype=True), _identity()) == "hype_vs_reality"

    def test_risk_story_gets_hidden_risks(self):
        sel = AngleSelector()
        assert sel.select(_story(risk=True), _identity()) == "hidden_risks"

    def test_tool_story_gets_misuse(self):
        sel = AngleSelector()
        assert sel.select(_story(tool=True), _identity()) == "misuse"

    def test_plain_story_gets_impact(self):
        sel = AngleSelector()
        assert sel.select(_story(), _identity()) == "impact"

    def test_hype_takes_priority_over_risk(self):
        sel = AngleSelector()
        both = _story(hype=True, risk=True)
        assert sel.select(both, _identity()) == "hype_vs_reality"

    def test_all_angles_in_constants(self):
        for angle in ANGLES:
            assert angle in ANGLE_INTENT

    def test_all_intent_have_required_keys(self):
        for angle, intent in ANGLE_INTENT.items():
            for key in ("goal", "emotion", "priority"):
                assert key in intent, f"{angle} missing {key!r}"

    def test_intent_returns_copy(self):
        sel = AngleSelector()
        i1 = sel.intent("hype_vs_reality")
        i1["goal"] = "mutated"
        i2 = sel.intent("hype_vs_reality")
        assert i2["goal"] != "mutated"

    def test_hype_vs_reality_emotion_is_doubt(self):
        assert ANGLE_INTENT["hype_vs_reality"]["emotion"] == "doubt"

    def test_hidden_risks_emotion_is_fear(self):
        assert ANGLE_INTENT["hidden_risks"]["emotion"] == "fear"


# ── PersonaMapper ─────────────────────────────────────────────────────────────

class TestPersonaMapper:
    def test_hype_angle_maps_to_skeptical_insider(self):
        mapper = PersonaMapper()
        p = mapper.map("hype_vs_reality")
        assert p["style"] == "skeptical_insider"

    def test_hidden_risks_maps_to_analyst(self):
        mapper = PersonaMapper()
        assert mapper.map("hidden_risks")["style"] == "analyst"

    def test_misuse_maps_to_builder(self):
        mapper = PersonaMapper()
        assert mapper.map("misuse")["style"] == "builder"

    def test_impact_maps_to_explainer(self):
        mapper = PersonaMapper()
        assert mapper.map("impact")["style"] == "explainer"

    def test_unknown_angle_defaults_to_explainer(self):
        mapper = PersonaMapper()
        assert mapper.style("nonexistent_angle") == "explainer"

    def test_persona_has_rules_list(self):
        mapper = PersonaMapper()
        for angle in ANGLE_PERSONA:
            p = mapper.map(angle)
            assert isinstance(p["rules"], list)
            assert len(p["rules"]) > 0

    def test_persona_has_voice_field(self):
        mapper = PersonaMapper()
        for angle in ANGLE_PERSONA:
            assert "voice" in mapper.map(angle)

    def test_map_returns_copy(self):
        mapper = PersonaMapper()
        p1 = mapper.map("impact")
        p1["style"] = "mutated"
        p2 = mapper.map("impact")
        assert p2["style"] != "mutated"

    def test_all_angles_covered(self):
        mapper = PersonaMapper()
        for angle in ANGLE_PERSONA:
            assert mapper.map(angle)["style"] in PERSONAS


# ── FormatSelector ────────────────────────────────────────────────────────────

class TestFormatSelector:
    def test_hype_angle_selects_hot_take(self):
        sel = FormatSelector()
        assert sel.select("hype_vs_reality")["type"] == "hot_take"

    def test_hidden_risks_selects_warning(self):
        sel = FormatSelector()
        assert sel.select("hidden_risks")["type"] == "warning"

    def test_misuse_selects_breakdown(self):
        sel = FormatSelector()
        assert sel.select("misuse")["type"] == "breakdown"

    def test_impact_selects_explainer(self):
        sel = FormatSelector()
        assert sel.select("impact")["type"] == "explainer"

    def test_unknown_angle_defaults_to_explainer(self):
        sel = FormatSelector()
        assert sel.format_type("unknown") == "explainer"

    def test_format_has_structure_list(self):
        sel = FormatSelector()
        for angle in ANGLE_FORMAT:
            fmt = sel.select(angle)
            assert isinstance(fmt["structure"], list)
            assert len(fmt["structure"]) > 0

    def test_format_has_pacing(self):
        sel = FormatSelector()
        for angle in ANGLE_FORMAT:
            assert "pacing" in sel.select(angle)

    def test_select_returns_copy(self):
        sel = FormatSelector()
        f1 = sel.select("impact")
        f1["type"] = "mutated"
        f2 = sel.select("impact")
        assert f2["type"] != "mutated"


# ── EditorialJudgment.judge() ─────────────────────────────────────────────────

class TestJudge:
    def test_returns_dict_with_required_keys(self, tmp_path):
        eng = _engine(tmp_path)
        result = eng.judge(_story())
        for key in ("story", "angle", "persona", "format", "editorial_intent"):
            assert key in result

    def test_angle_is_valid(self, tmp_path):
        eng = _engine(tmp_path)
        result = eng.judge(_story())
        assert result["angle"] in ANGLES

    def test_persona_has_style(self, tmp_path):
        eng = _engine(tmp_path)
        result = eng.judge(_story())
        assert "style" in result["persona"]

    def test_format_has_type(self, tmp_path):
        eng = _engine(tmp_path)
        result = eng.judge(_story())
        assert "type" in result["format"]

    def test_editorial_intent_has_goal_emotion_priority(self, tmp_path):
        eng = _engine(tmp_path)
        result = eng.judge(_story())
        for key in ("goal", "emotion", "priority"):
            assert key in result["editorial_intent"]

    def test_hype_story_gets_doubt_emotion(self, tmp_path):
        eng = _engine(tmp_path)
        result = eng.judge(_story(hype=True))
        assert result["editorial_intent"]["emotion"] == "doubt"

    def test_risk_story_gets_fear_emotion(self, tmp_path):
        eng = _engine(tmp_path)
        result = eng.judge(_story(risk=True))
        assert result["editorial_intent"]["emotion"] == "fear"

    def test_original_story_not_mutated(self, tmp_path):
        eng = _engine(tmp_path)
        s = _story(importance=5.0)
        eng.judge(s)
        assert s["importance"] == 5.0


# ── EditorialJudgment.process() ───────────────────────────────────────────────

class TestProcess:
    def test_empty_stories_returns_empty_dict(self, tmp_path):
        eng = _engine(tmp_path)
        assert eng.process([]) == {}

    def test_returns_judgment_for_top_story(self, tmp_path):
        eng = _engine(tmp_path)
        stories = [_story(importance=1), _story(importance=9)]
        result  = eng.process(stories)
        assert result["story"]["importance"] == 9

    def test_result_has_all_keys(self, tmp_path):
        eng = _engine(tmp_path)
        result = eng.process([_story(importance=5)])
        for key in ("story", "angle", "persona", "format", "editorial_intent"):
            assert key in result

    def test_single_story_processed(self, tmp_path):
        eng = _engine(tmp_path)
        result = eng.process([_story()])
        assert result["angle"] in ANGLES


# ── EditorialJudgment.update_from_feedback() ─────────────────────────────────

class TestUpdateFromFeedback:
    def test_high_retention_promotes_priority(self, tmp_path):
        eng = _engine(tmp_path)
        eng.identity.priorities = ["real_world_impact", "misconceptions", "hidden_risks"]
        eng.update_from_feedback({"angle": "hidden_risks", "retention": 0.75}, persist=False)
        assert eng.identity.priorities[0] == "hidden_risks"

    def test_low_retention_does_not_change_priorities(self, tmp_path):
        eng = _engine(tmp_path)
        before = list(eng.identity.priorities)
        eng.update_from_feedback({"angle": "impact", "retention": 0.30}, persist=False)
        assert eng.identity.priorities == before

    def test_at_threshold_promotes(self, tmp_path):
        eng = _engine(tmp_path)
        eng.identity.priorities = ["real_world_impact", "hidden_risks"]
        eng.update_from_feedback(
            {"angle": "hidden_risks", "retention": _RETENTION_WINNER_THRESHOLD + 0.01},
            persist=False,
        )
        assert eng.identity.priorities[0] == "hidden_risks"

    def test_feedback_written_to_file(self, tmp_path):
        eng = _engine(tmp_path)
        eng.update_from_feedback({"angle": "impact", "retention": 0.80}, persist=False)
        assert (tmp_path / "editorial_feedback.jsonl").exists()

    def test_empty_angle_is_safe(self, tmp_path):
        eng = _engine(tmp_path)
        before = list(eng.identity.priorities)
        eng.update_from_feedback({"angle": "", "retention": 0.99}, persist=False)
        assert eng.identity.priorities == before


# ── load_feedback / winning_angles ────────────────────────────────────────────

class TestFeedbackPersistence:
    def test_load_empty_store(self, tmp_path):
        eng = _engine(tmp_path)
        assert eng.load_feedback() == []

    def test_load_returns_records(self, tmp_path):
        eng = _engine(tmp_path)
        eng.update_from_feedback({"angle": "impact", "retention": 0.8}, persist=False)
        records = eng.load_feedback()
        assert len(records) == 1
        assert records[0].angle == "impact"

    def test_load_respects_n(self, tmp_path):
        eng = _engine(tmp_path)
        for _ in range(10):
            eng.update_from_feedback({"angle": "impact", "retention": 0.8}, persist=False)
        assert len(eng.load_feedback(n=3)) == 3

    def test_winning_angles_above_threshold(self, tmp_path):
        eng = _engine(tmp_path)
        for _ in range(3):
            eng.update_from_feedback({"angle": "hype_vs_reality", "retention": 0.8}, persist=False)
        assert "hype_vs_reality" in eng.winning_angles()

    def test_losing_angles_not_in_winners(self, tmp_path):
        eng = _engine(tmp_path)
        for _ in range(3):
            eng.update_from_feedback({"angle": "impact", "retention": 0.2}, persist=False)
        assert "impact" not in eng.winning_angles()

    def test_feedback_roundtrip(self, tmp_path):
        eng = _engine(tmp_path)
        eng.update_from_feedback({"angle": "misuse", "retention": 0.65, "video_id": "v1"}, persist=False)
        rec = eng.load_feedback()[0]
        assert rec.angle == "misuse"
        assert rec.retention == 0.65
        assert rec.video_id == "v1"


# ── singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_returns_engine(self):
        assert isinstance(get_judgment_engine(), EditorialJudgment)

    def test_same_instance(self):
        assert get_judgment_engine() is get_judgment_engine()
