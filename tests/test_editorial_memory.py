"""Tests for src/editorial/editorial_memory.py."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_memory(tmp_path: Path):
    from src.editorial.editorial_memory import EditorialMemory
    return EditorialMemory(data_dir=tmp_path)


def _entry(topic="OpenAI agents", angle="hype_vs_reality", prediction="agents are overhyped",
           retention=0.67, persona="skeptical_insider", fmt="hot_take", video_id="vid001"):
    return {
        "topic": topic,
        "angle": angle,
        "persona": persona,
        "format": fmt,
        "prediction": prediction,
        "retention": retention,
        "comments_sentiment": "positive",
        "video_id": video_id,
    }


# ── basic CRUD ────────────────────────────────────────────────────────────────

def test_add_and_load(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry())
    mem2 = _make_memory(tmp_path)
    assert len(mem2._entries) == 1
    assert mem2._entries[0]["topic"] == "OpenAI agents"


def test_persists_date(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry())
    assert mem._entries[0]["date"] != ""


def test_max_entries_pruned(tmp_path):
    from src.editorial.editorial_memory import _MAX_ENTRIES
    mem = _make_memory(tmp_path)
    for i in range(_MAX_ENTRIES + 10):
        mem._entries.append({"topic": f"t{i}", "angle": "impact"})
    mem.add_entry(_entry())
    assert len(mem._entries) == _MAX_ENTRIES


# ── relevant context ──────────────────────────────────────────────────────────

def test_get_relevant_context_matches(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry("OpenAI agents launch"))
    mem.add_entry(_entry("Google DeepMind paper"))
    mem.add_entry(_entry("OpenAI agents controversy"))

    ctx = mem.get_relevant_context("openai agents")
    assert len(ctx) == 2
    assert all("openai" in c["topic"].lower() for c in ctx)


def test_get_relevant_context_empty_for_unknown_topic(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry("OpenAI agents"))
    assert mem.get_relevant_context("quantum computing") == []


def test_context_window_limit(tmp_path):
    from src.editorial.editorial_memory import _CONTEXT_WINDOW
    mem = _make_memory(tmp_path)
    for i in range(_CONTEXT_WINDOW + 3):
        mem.add_entry(_entry(f"OpenAI agents v{i}"))
    ctx = mem.get_relevant_context("openai agents")
    assert len(ctx) == _CONTEXT_WINDOW


# ── has_prediction / callback ─────────────────────────────────────────────────

def test_has_prediction_true(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry(prediction="agents will fail"))
    assert mem.has_prediction("openai agents")


def test_has_prediction_false_no_pred(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry(prediction=""))
    assert not mem.has_prediction("openai agents")


def test_has_prediction_false_wrong_topic(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry())
    assert not mem.has_prediction("totally different topic")


def test_get_callback_line_returns_string(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry(prediction="agents are overhyped"))
    line = mem.get_callback_line("openai agents")
    assert line is not None
    assert "agents are overhyped" in line


def test_get_callback_line_none_without_prediction(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry(prediction=""))
    assert mem.get_callback_line("openai agents") is None


# ── topic_is_recent ───────────────────────────────────────────────────────────

def test_topic_is_recent_true(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry("OpenAI launches GPT-5"))
    assert mem.topic_is_recent("openai launches")


def test_topic_is_recent_false_when_old(tmp_path):
    from src.editorial.editorial_memory import _RECENT_WINDOW
    mem = _make_memory(tmp_path)
    # push the matching entry beyond the recent window
    for i in range(_RECENT_WINDOW):
        mem.add_entry(_entry(f"unrelated topic {i}", prediction=""))
    mem._entries.insert(0, _entry("OpenAI GPT-5"))
    assert not mem.topic_is_recent("openai gpt-5")


# ── confirm_prediction ────────────────────────────────────────────────────────

def test_confirm_prediction(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry(prediction="agents will fail"))
    mem.confirm_prediction("openai agents")
    assert mem._entries[-1]["prediction_status"] == "confirmed"


def test_wrong_prediction(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry(prediction="agents will succeed"))
    mem.confirm_prediction("openai agents", wrong=True)
    assert mem._entries[-1]["prediction_status"] == "wrong"


def test_confirm_no_match_does_nothing(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry())
    mem.confirm_prediction("quantum computing")  # should not raise


# ── analytics ─────────────────────────────────────────────────────────────────

def test_top_angles(tmp_path):
    mem = _make_memory(tmp_path)
    for _ in range(3):
        mem.add_entry(_entry(angle="hype_vs_reality"))
    for _ in range(2):
        mem.add_entry(_entry(angle="impact"))
    mem.add_entry(_entry(angle="hidden_risks"))

    top = mem.top_angles(n=2)
    assert top[0] == ("hype_vs_reality", 3)
    assert top[1] == ("impact", 2)


def test_avg_retention_by_angle(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry(angle="hype_vs_reality", retention=0.70))
    mem.add_entry(_entry(angle="hype_vs_reality", retention=0.60))
    mem.add_entry(_entry(angle="impact", retention=0.50))

    avgs = mem.avg_retention_by_angle()
    assert abs(avgs["hype_vs_reality"] - 0.65) < 1e-9
    assert abs(avgs["impact"] - 0.50) < 1e-9


def test_recent_predictions(tmp_path):
    mem = _make_memory(tmp_path)
    mem.add_entry(_entry(prediction="pred 1"))
    mem.add_entry(_entry(prediction=""))       # no prediction — excluded
    mem.add_entry(_entry(prediction="pred 2"))

    preds = mem.recent_predictions(n=10)
    assert len(preds) == 2
    assert preds[-1]["prediction"] == "pred 2"


# ── story ranker integration ──────────────────────────────────────────────────

def test_ranker_penalises_recent_topic(tmp_path):
    from src.editorial.editorial_memory import EditorialMemory
    from src.editorial.story_ranker import StoryRanker
    from src.editorial.editorial_identity import EditorialIdentity

    mem = EditorialMemory(data_dir=tmp_path)
    mem.add_entry({"topic": "OpenAI GPT-5", "angle": "impact"})

    identity = EditorialIdentity()
    ranker = StoryRanker()

    story_repeated = {"title": "OpenAI GPT-5 release", "importance": 5.0}
    story_fresh    = {"title": "Google Gemini paper",  "importance": 5.0}

    with patch("src.editorial.editorial_memory.get_editorial_memory", return_value=mem):
        ranked = ranker.rank([story_repeated, story_fresh], identity)

    # fresh story should score higher than repeated one
    assert ranked[0][1]["title"] == "Google Gemini paper"


# ── editorial judgment integration ───────────────────────────────────────────

def test_judgment_injects_callback(tmp_path):
    from src.editorial.editorial_memory import EditorialMemory
    from src.editorial.editorial_judgment import EditorialJudgment

    mem = EditorialMemory(data_dir=tmp_path)
    mem.add_entry({
        "topic": "OpenAI agents",
        "angle": "hype_vs_reality",
        "prediction": "agents are overhyped",
    })

    engine = EditorialJudgment(data_dir=tmp_path)
    with patch("src.editorial.editorial_memory.get_editorial_memory", return_value=mem):
        judgment = engine.judge({"title": "OpenAI agents new release", "hype": True})

    assert "callback" in judgment
    assert "agents are overhyped" in judgment["callback"]


def test_judgment_marks_recent_topic(tmp_path):
    from src.editorial.editorial_memory import EditorialMemory
    from src.editorial.editorial_judgment import EditorialJudgment

    mem = EditorialMemory(data_dir=tmp_path)
    mem.add_entry({"topic": "OpenAI GPT-5", "angle": "impact"})

    engine = EditorialJudgment(data_dir=tmp_path)
    with patch("src.editorial.editorial_memory.get_editorial_memory", return_value=mem):
        judgment = engine.judge({"title": "OpenAI GPT-5 follow-up"})

    assert judgment.get("topic_is_recent") is True


def test_judgment_no_callback_for_new_topic(tmp_path):
    from src.editorial.editorial_judgment import EditorialJudgment
    from src.editorial.editorial_memory import EditorialMemory

    mem = EditorialMemory(data_dir=tmp_path)
    engine = EditorialJudgment(data_dir=tmp_path)
    with patch("src.editorial.editorial_memory.get_editorial_memory", return_value=mem):
        judgment = engine.judge({"title": "Brand new quantum AI paper"})

    assert "callback" not in judgment
    assert not judgment.get("topic_is_recent")
