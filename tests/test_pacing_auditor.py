"""Tests for pacing_auditor — rule-based scene scoring and flat zone detection."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.script_generator import Scene, VideoScript
from src.pacing_auditor import (
    _score_scene,
    _find_flat_zones,
    SceneScore,
    FlatZone,
    _FLAT_THRESHOLD,
    _FLAT_RUN_MIN,
    _MAX_INTERRUPTS,
)


def _scene(idx: int, narration: str, heading: str = "") -> Scene:
    return Scene(
        idx=idx,
        heading=heading or f"Scene {idx}",
        narration=narration,
        visual_prompt="",
        duration_sec=30,
    )


# ── _score_scene (rule-based, zero mocks) ────────────────────────────────────

class TestScoreScene:
    def test_question_adds_signal(self):
        sc = _scene(0, "Is this the future of AI?")
        result = _score_scene(sc)
        assert "question" in result.signals

    def test_number_with_percent_adds_signal(self):
        sc = _scene(0, "Performance improved by 35%.")
        result = _score_scene(sc)
        assert "number" in result.signals

    def test_dollar_amount_adds_signal(self):
        sc = _scene(0, "This costs $20 per month.")
        result = _score_scene(sc)
        assert "number" in result.signals

    def test_multiplier_adds_signal(self):
        sc = _scene(0, "That is 10x faster than before.")
        result = _score_scene(sc)
        assert "number" in result.signals

    def test_conflict_word_adds_signal(self):
        sc = _scene(0, "But there is a problem with this approach.")
        result = _score_scene(sc)
        assert "conflict" in result.signals

    def test_however_is_conflict_word(self):
        sc = _scene(0, "However the results were surprising.")
        result = _score_scene(sc)
        assert "conflict" in result.signals

    def test_short_sentence_adds_signal(self):
        sc = _scene(0, "This is great. And it matters.")
        result = _score_scene(sc)
        assert "short_sentence" in result.signals

    def test_flat_scene_scores_zero(self):
        sc = _scene(0, "The model was trained on a large dataset for several weeks using distributed computing.")
        result = _score_scene(sc)
        assert result.score == 0

    def test_all_four_signals_possible(self):
        sc = _scene(0, "But can it handle 100% of cases? Yes. Actually no.")
        result = _score_scene(sc)
        assert result.score == 4

    def test_score_is_signal_count(self):
        sc = _scene(0, "Is this 50% better? But wait. That is significant!")
        result = _score_scene(sc)
        assert result.score == len(result.signals)

    def test_returns_scene_score_with_correct_idx(self):
        sc = _scene(5, "Plain narration here.")
        result = _score_scene(sc)
        assert result.idx == 5


# ── _find_flat_zones ──────────────────────────────────────────────────────────

def _flat_score(idx: int) -> SceneScore:
    return SceneScore(idx=idx, score=0, signals=[])


def _engaging_score(idx: int) -> SceneScore:
    return SceneScore(idx=idx, score=3, signals=["question", "number", "conflict"])


class TestFindFlatZones:
    def test_no_zones_when_all_engaging(self):
        scores = [_engaging_score(i) for i in range(5)]
        assert _find_flat_zones(scores) == []

    def test_detects_run_of_flat_scenes(self):
        # intro + 3 flat interior scenes + outro
        scores = [
            _engaging_score(0),   # intro — skipped
            _flat_score(1),
            _flat_score(2),
            _flat_score(3),
            _engaging_score(4),   # final — skipped
        ]
        zones = _find_flat_zones(scores)
        assert len(zones) == 1
        assert zones[0].scene_indices == [1, 2, 3]
        assert zones[0].interrupt_at == 3

    def test_single_flat_scene_not_a_zone(self):
        scores = [
            _engaging_score(0),
            _flat_score(1),
            _engaging_score(2),
            _engaging_score(3),
        ]
        assert _find_flat_zones(scores) == []

    def test_skips_intro_and_final_scene(self):
        # Only 2 scenes: both are intro+final → interior is empty
        scores = [_flat_score(0), _flat_score(1)]
        assert _find_flat_zones(scores) == []

    def test_caps_at_max_interrupts(self):
        # 3 separate flat runs in interior
        scores = [
            _engaging_score(0),      # intro
            _flat_score(1), _flat_score(2),
            _engaging_score(3),
            _flat_score(4), _flat_score(5),
            _engaging_score(6),
            _flat_score(7), _flat_score(8),
            _engaging_score(9),
            _flat_score(10), _flat_score(11),
            _engaging_score(12),     # final
        ]
        zones = _find_flat_zones(scores)
        assert len(zones) <= _MAX_INTERRUPTS

    def test_two_separate_flat_runs_detected(self):
        scores = [
            _engaging_score(0),
            _flat_score(1), _flat_score(2),
            _engaging_score(3),
            _flat_score(4), _flat_score(5),
            _engaging_score(6),
        ]
        zones = _find_flat_zones(scores)
        assert len(zones) == 2

    def test_interrupt_at_is_last_flat_scene_index(self):
        scores = [
            _engaging_score(0),
            _flat_score(1), _flat_score(2), _flat_score(3),
            _engaging_score(4),
        ]
        zones = _find_flat_zones(scores)
        assert zones[0].interrupt_at == 3
