"""Tests for audience_profile.py and audience_judgment.py."""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch


# ── AudienceProfile ───────────────────────────────────────────────────────────

def _make_profile(tmp_path: Path):
    from src.editorial.audience_profile import AudienceProfile
    return AudienceProfile(data_dir=tmp_path)


def test_profile_starts_blank(tmp_path):
    p = _make_profile(tmp_path)
    assert p.get_angle_score("hype_vs_reality") == 0.0
    assert p.get_persona_affinity("skeptical_insider") == 0.0


def test_reinforce_on_good_retention(tmp_path):
    p = _make_profile(tmp_path)
    p.update_from_feedback("hype_vs_reality", 0.70, format_type="hot_take", persona="skeptical_insider")
    assert p.get_angle_score("hype_vs_reality") > 0.0
    assert p.get_format_score("hot_take") > 0.0
    assert p.get_persona_affinity("skeptical_insider") > 0.0


def test_decay_on_bad_retention(tmp_path):
    p = _make_profile(tmp_path)
    # seed some initial score
    p._p["preferred_angles"]["impact"] = 0.30
    p.update_from_feedback("impact", 0.20)
    assert p.get_angle_score("impact") < 0.30


def test_disliked_pattern_added_on_bad_retention(tmp_path):
    p = _make_profile(tmp_path)
    p.update_from_feedback("impact", 0.20, hook_pattern="generic_question")
    assert p.is_disliked("generic_question")


def test_no_reinforcement_in_neutral_zone(tmp_path):
    p = _make_profile(tmp_path)
    p.update_from_feedback("impact", 0.50)
    # 0.50 is between _RETENTION_BAD and _RETENTION_GOOD — no change
    assert p.get_angle_score("impact") == 0.0


def test_scores_clamped_max(tmp_path):
    from src.editorial.audience_profile import _MAX_SCORE
    p = _make_profile(tmp_path)
    p._p["preferred_angles"]["hype_vs_reality"] = _MAX_SCORE
    p.update_from_feedback("hype_vs_reality", 0.90)
    assert p.get_angle_score("hype_vs_reality") == _MAX_SCORE


def test_scores_clamped_min(tmp_path):
    p = _make_profile(tmp_path)
    p._p["preferred_angles"]["impact"] = 0.0
    p.update_from_feedback("impact", 0.10)
    assert p.get_angle_score("impact") >= 0.0


def test_emotion_reinforced(tmp_path):
    p = _make_profile(tmp_path)
    p.update_from_feedback("hype_vs_reality", 0.80, emotions=["doubt", "surprise"])
    assert p.get_emotion_score("doubt") > 0.0
    assert p.get_emotion_score("surprise") > 0.0


def test_retention_pattern_running_avg(tmp_path):
    p = _make_profile(tmp_path)
    p.update_from_feedback("impact", 0.60)
    p.update_from_feedback("impact", 0.80)
    avg = p.get_angle_retention("impact")
    assert abs(avg - 0.70) < 1e-6


def test_persists_and_reloads(tmp_path):
    p1 = _make_profile(tmp_path)
    p1.update_from_feedback("hype_vs_reality", 0.75, persona="skeptical_insider")
    p2 = _make_profile(tmp_path)
    assert p2.get_angle_score("hype_vs_reality") > 0.0
    assert p2.get_persona_affinity("skeptical_insider") > 0.0


def test_top_angles_sorted(tmp_path):
    p = _make_profile(tmp_path)
    p._p["preferred_angles"] = {"hype_vs_reality": 0.80, "impact": 0.50, "hidden_risks": 0.65}
    top = p.top_angles(n=2)
    assert top[0][0] == "hype_vs_reality"
    assert top[1][0] == "hidden_risks"


def test_blank_profile_backfills_missing_keys(tmp_path):
    import json
    # write an old profile missing the emotion_preferences key
    path = tmp_path / "audience_profile.json"
    path.write_text(json.dumps({"preferred_angles": {"impact": 0.5}}))
    p = _make_profile(tmp_path)
    assert "emotion_preferences" in p._p
    assert "persona_affinity" in p._p


# ── AudienceAwareJudgment ────────────────────────────────────────────────────

def _make_judgment(tmp_path: Path, *, seed_profile: bool = False):
    from src.editorial.audience_judgment import AudienceAwareJudgment
    from src.editorial.audience_profile import AudienceProfile
    from src.editorial.editorial_memory import EditorialMemory
    from src.editorial.editorial_identity import EditorialIdentity

    profile = AudienceProfile(data_dir=tmp_path)
    memory  = EditorialMemory(data_dir=tmp_path)
    identity = EditorialIdentity()

    if seed_profile:
        profile._p["preferred_angles"]["hype_vs_reality"] = 0.90
        profile._p["persona_affinity"]["skeptical_insider"] = 0.70
        profile._p["retention_patterns"]["hype_vs_reality"] = {"avg": 0.72, "count": 5}

    return AudienceAwareJudgment(identity=identity, audience=profile, memory=memory)


def test_judge_returns_full_schema(tmp_path):
    engine = _make_judgment(tmp_path)
    result = engine.judge({"title": "GPT-5 launch"})
    assert "angle" in result
    assert "persona" in result
    assert "format" in result
    assert "editorial_intent" in result
    assert "audience_scores" in result


def test_angle_scores_all_candidates(tmp_path):
    engine = _make_judgment(tmp_path)
    scores = engine.score_angles({"hype": True})
    from src.editorial.angle_selector import ANGLES
    for angle in ANGLES:
        assert angle in scores


def test_hype_signal_boosts_hype_vs_reality(tmp_path):
    engine = _make_judgment(tmp_path)
    scores = engine.score_angles({"hype": True})
    # hype signal gives a half-identity-weight bonus to hype_vs_reality
    assert scores["hype_vs_reality"] > scores["impact"]


def test_audience_preference_shifts_selection(tmp_path):
    engine = _make_judgment(tmp_path, seed_profile=True)
    # With 0.90 audience preference + 0.72 retention for hype_vs_reality,
    # it should win even for a neutral story
    angle = engine.select_angle({"title": "neutral story"})
    assert angle == "hype_vs_reality"


def test_persona_fallback_on_low_affinity(tmp_path):
    engine = _make_judgment(tmp_path)
    # Force low affinity for skeptical_insider
    engine.audience._p["persona_affinity"]["skeptical_insider"] = 0.10
    persona = engine.select_persona("hype_vs_reality")
    assert persona == "explainer"


def test_persona_canonical_when_no_data(tmp_path):
    engine = _make_judgment(tmp_path)
    # No affinity data → canonical persona returned
    persona = engine.select_persona("hype_vs_reality")
    assert persona == "skeptical_insider"


def test_persona_canonical_when_affinity_above_threshold(tmp_path):
    engine = _make_judgment(tmp_path)
    engine.audience._p["persona_affinity"]["skeptical_insider"] = 0.60
    persona = engine.select_persona("hype_vs_reality")
    assert persona == "skeptical_insider"


def test_filter_hooks_removes_disliked(tmp_path):
    engine = _make_judgment(tmp_path)
    engine.audience._p["disliked_patterns"]["generic_question"] = 0.15
    hooks = [
        {"pattern": "generic_question", "text": "Do you know about AI?"},
        {"pattern": "bold_claim",        "text": "AI just changed everything"},
    ]
    result = engine.filter_hooks(hooks)
    assert len(result) == 1
    assert result[0]["pattern"] == "bold_claim"


def test_filter_hooks_passes_all_clean(tmp_path):
    engine = _make_judgment(tmp_path)
    hooks = [{"pattern": "bold_claim"}, {"pattern": "question"}]
    assert engine.filter_hooks(hooks) == hooks


def test_callback_injected_from_memory(tmp_path):
    from src.editorial.editorial_memory import EditorialMemory
    mem = EditorialMemory(data_dir=tmp_path)
    mem.add_entry({"topic": "OpenAI agents", "angle": "hype_vs_reality",
                   "prediction": "agents will disappoint"})
    from src.editorial.audience_judgment import AudienceAwareJudgment
    from src.editorial.audience_profile import AudienceProfile
    engine = AudienceAwareJudgment(audience=AudienceProfile(data_dir=tmp_path), memory=mem)
    result = engine.judge({"title": "OpenAI agents update"})
    assert "callback" in result
    assert "agents will disappoint" in result["callback"]


# ── EditorialJudgment delegates to AudienceAwareJudgment ──────────────────────

def test_editorial_judgment_uses_audience_path_when_profile_has_data(tmp_path):
    from src.editorial.editorial_judgment import EditorialJudgment
    from src.editorial.audience_profile import AudienceProfile

    # Seed the profile file so judge() activates the audience path
    profile = AudienceProfile(data_dir=tmp_path)
    profile._p["preferred_angles"]["hidden_risks"] = 0.80
    profile._p["retention_patterns"]["hidden_risks"] = {"avg": 0.75, "count": 3}
    profile.save()

    engine = EditorialJudgment(data_dir=tmp_path)
    result = engine.judge({"title": "AI risks paper"})
    # audience_scores only present on the audience-aware path
    assert "audience_scores" in result


def test_editorial_judgment_fallback_when_no_profile(tmp_path):
    from src.editorial.editorial_judgment import EditorialJudgment
    engine = EditorialJudgment(data_dir=tmp_path)
    result = engine.judge({"title": "brand new topic"})
    assert "angle" in result
    # fallback path does not include audience_scores
    assert "audience_scores" not in result


def test_update_from_feedback_updates_audience_profile(tmp_path):
    from src.editorial.editorial_judgment import EditorialJudgment
    from src.editorial.audience_profile import AudienceProfile

    engine = EditorialJudgment(data_dir=tmp_path)
    engine.update_from_feedback({
        "angle": "hype_vs_reality",
        "retention": 0.75,
        "persona": "skeptical_insider",
        "format": "hot_take",
    })

    profile = AudienceProfile(data_dir=tmp_path)
    assert profile.get_angle_score("hype_vs_reality") > 0.0
    assert profile.get_persona_affinity("skeptical_insider") > 0.0
