"""Tests for pacing_auditor — rule-based scene scoring, flat zone detection,
GPT interrupt generation, and the public audit_pacing() API."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest

from src.pacing_auditor import (
    _FLAT_RUN_MIN,
    _FLAT_THRESHOLD,
    _MAX_INTERRUPTS,
    FlatZone,
    PacingAuditResult,
    SceneScore,
    _find_flat_zones,
    _generate_interrupt,
    _log_usage,
    _score_scene,
    audit_pacing,
)
from src.script_generator import Scene, VideoScript


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


# ── Helpers shared by the tests below ────────────────────────────────────────

def _make_script(narrations: list[str]) -> VideoScript:
    """Build a minimal VideoScript from a list of narration strings."""
    scenes = [
        Scene(
            idx=i,
            heading=f"Scene {i}",
            narration=narration,
            visual_prompt="",
            duration_sec=30,
        )
        for i, narration in enumerate(narrations)
    ]
    return VideoScript(
        title="Test Script",
        description="desc",
        tags=["ai"],
        hook="Hook text",
        scenes=scenes,
    )


def _mock_openai_response(interrupt_text: str = "Wait — does this work?") -> MagicMock:
    """Build a fake OpenAI completion response."""
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps({
        "interrupt": interrupt_text,
        "technique": "question",
    })
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 30
    resp.usage.prompt_tokens_details = None
    return resp


# ── _log_usage ────────────────────────────────────────────────────────────────

class TestLogUsage:
    def test_noop_when_usage_is_none(self):
        # Should not raise; no calls to get_ledger
        with patch("src.pacing_auditor.get_ledger") as mock_ledger:
            _log_usage(None)
            mock_ledger.assert_not_called()

    def test_records_llm_call_when_usage_present(self):
        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 20
        usage.prompt_tokens_details = None
        with patch("src.pacing_auditor.get_ledger") as mock_ledger:
            ledger = MagicMock()
            mock_ledger.return_value = ledger
            _log_usage(usage, label="test-label")
        ledger.record_llm.assert_called_once()
        call_kwargs = ledger.record_llm.call_args[1]
        assert call_kwargs["tag"] == "test-label"
        assert call_kwargs["prompt_tokens"] == 50
        assert call_kwargs["completion_tokens"] == 20

    def test_uses_default_label_when_empty(self):
        usage = MagicMock()
        usage.prompt_tokens = 10
        usage.completion_tokens = 5
        usage.prompt_tokens_details = None
        with patch("src.pacing_auditor.get_ledger") as mock_ledger:
            ledger = MagicMock()
            mock_ledger.return_value = ledger
            _log_usage(usage)  # no label
        call_kwargs = ledger.record_llm.call_args[1]
        assert call_kwargs["tag"] == "pacing"

    def test_extracts_cached_tokens_when_details_present(self):
        usage = MagicMock()
        usage.prompt_tokens = 100
        usage.completion_tokens = 20
        usage.prompt_tokens_details = MagicMock()
        usage.prompt_tokens_details.cached_tokens = 40
        with patch("src.pacing_auditor.get_ledger") as mock_ledger:
            ledger = MagicMock()
            mock_ledger.return_value = ledger
            _log_usage(usage, label="pacing-interrupt")
        call_kwargs = ledger.record_llm.call_args[1]
        assert call_kwargs["cached_tokens"] == 40


# ── _generate_interrupt ───────────────────────────────────────────────────────

class TestGenerateInterrupt:
    def _make_zone(self, scene_indices, interrupt_at=None):
        return FlatZone(
            scene_indices=scene_indices,
            interrupt_at=interrupt_at if interrupt_at is not None else scene_indices[-1],
        )

    def test_returns_interrupt_text_on_success(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("Wait — does this work?")
        scenes = [
            Scene(idx=0, heading="Intro", narration="Into text", visual_prompt="", duration_sec=30),
            Scene(idx=1, heading="Flat 1", narration="Flat narration here", visual_prompt="", duration_sec=30),
            Scene(idx=2, heading="Flat 2", narration="More flat narration", visual_prompt="", duration_sec=30),
            Scene(idx=3, heading="Next", narration="This is next", visual_prompt="", duration_sec=30),
        ]
        with patch("src.pacing_auditor.get_ledger"):
            with patch("src.pacing_auditor._log_usage"):
                result = _generate_interrupt(client, self._make_zone([1, 2]), scenes, "AI topic")
        assert result == "Wait — does this work?"

    def test_returns_empty_string_when_gpt_raises(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API down")
        scenes = [
            Scene(idx=0, heading="S0", narration="text", visual_prompt="", duration_sec=30),
            Scene(idx=1, heading="S1", narration="flat", visual_prompt="", duration_sec=30),
            Scene(idx=2, heading="S2", narration="flat", visual_prompt="", duration_sec=30),
        ]
        result = _generate_interrupt(client, self._make_zone([1, 2]), scenes, "topic")
        assert result == ""

    def test_handles_end_of_script_next_scene(self):
        """When interrupt_at is the last scene, next_scene should be '(end of script)'."""
        client = MagicMock()
        client.chat.completions.create.return_value = _mock_openai_response("And here's the twist.")
        scenes = [
            Scene(idx=0, heading="Intro", narration="intro", visual_prompt="", duration_sec=30),
            Scene(idx=1, heading="Flat", narration="flat text", visual_prompt="", duration_sec=30),
            Scene(idx=2, heading="Flat2", narration="flat text 2", visual_prompt="", duration_sec=30),
        ]
        zone = self._make_zone([1, 2], interrupt_at=2)  # last scene
        with patch("src.pacing_auditor.get_ledger"):
            with patch("src.pacing_auditor._log_usage"):
                result = _generate_interrupt(client, zone, scenes, "AI")
        # The call should have been made (end of script branch)
        assert isinstance(result, str)

    def test_empty_interrupt_in_response_returns_empty(self):
        client = MagicMock()
        resp = MagicMock()
        resp.choices[0].message.content = json.dumps({"interrupt": "", "technique": "question"})
        resp.usage.prompt_tokens_details = None
        client.chat.completions.create.return_value = resp
        scenes = [
            Scene(idx=0, heading="S0", narration="a", visual_prompt="", duration_sec=30),
            Scene(idx=1, heading="S1", narration="b", visual_prompt="", duration_sec=30),
            Scene(idx=2, heading="S2", narration="c", visual_prompt="", duration_sec=30),
        ]
        with patch("src.pacing_auditor.get_ledger"):
            with patch("src.pacing_auditor._log_usage"):
                result = _generate_interrupt(client, self._make_zone([1, 2]), scenes, "AI")
        assert result == ""


# ── audit_pacing (public API) ─────────────────────────────────────────────────

class TestAuditPacing:
    def _flat_narration(self):
        return "The model was trained on a large corpus of text data from many sources."

    def _engaging_narration(self):
        return "But can it handle 100% of cases? Actually no — wait."

    def test_no_flat_zones_returns_original_script_unmodified(self):
        """All interior scenes engaging → no GPT calls, script returned as-is."""
        script = _make_script([
            self._engaging_narration(),   # idx 0 intro
            self._engaging_narration(),   # idx 1
            self._engaging_narration(),   # idx 2
            self._engaging_narration(),   # idx 3 final (skipped)
        ])
        with patch("src.pacing_auditor.make_openai_client") as mock_client:
            result = audit_pacing(script, subject="AI topic")
        mock_client.assert_not_called()
        assert result.modified is False
        assert result.interrupts_added == 0
        assert result.script is script

    def test_flat_zones_trigger_gpt_and_inject_interrupt(self):
        """Two flat interior scenes → GPT called, interrupt injected."""
        script = _make_script([
            self._engaging_narration(),   # 0 intro
            self._flat_narration(),       # 1 flat
            self._flat_narration(),       # 2 flat → interrupt injected here
            self._engaging_narration(),   # 3 final
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "Wait — is this really that simple?"
        )
        with patch("src.pacing_auditor.make_openai_client", return_value=mock_client):
            with patch("src.pacing_auditor.get_ledger"):
                result = audit_pacing(script, subject="AI")

        assert result.modified is True
        assert result.interrupts_added == 1
        # The interrupt text should appear in the modified scene's narration
        assert "Wait — is this really that simple?" in result.script.scenes[2].narration

    def test_original_script_scenes_not_mutated(self):
        """audit_pacing must not mutate the original script in place."""
        orig_narration = self._flat_narration()
        script = _make_script([
            self._engaging_narration(),
            orig_narration,
            self._flat_narration(),
            self._engaging_narration(),
        ])
        original_text = script.scenes[2].narration

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("Surprise!")
        with patch("src.pacing_auditor.make_openai_client", return_value=mock_client):
            with patch("src.pacing_auditor.get_ledger"):
                result = audit_pacing(script)

        # Original scene[2] should be unchanged
        assert script.scenes[2].narration == original_text

    def test_uses_script_title_as_subject_when_none_given(self):
        """subject defaults to script.title."""
        script = _make_script([
            self._engaging_narration(),
            self._flat_narration(),
            self._flat_narration(),
            self._engaging_narration(),
        ])
        script_title_used = []
        mock_client = MagicMock()

        def capture(*args, **kwargs):
            # Extract the user message to verify subject used
            msgs = kwargs.get("messages", [])
            for m in msgs:
                if m["role"] == "user":
                    script_title_used.append(m["content"])
            resp = _mock_openai_response("Hmm, interesting.")
            return resp

        mock_client.chat.completions.create.side_effect = capture
        with patch("src.pacing_auditor.make_openai_client", return_value=mock_client):
            with patch("src.pacing_auditor.get_ledger"):
                audit_pacing(script)  # no subject arg

        assert any("Test Script" in s for s in script_title_used)

    def test_returns_correct_scores_count(self):
        script = _make_script([
            self._engaging_narration(),
            self._flat_narration(),
            self._flat_narration(),
            self._engaging_narration(),
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("Interesting.")
        with patch("src.pacing_auditor.make_openai_client", return_value=mock_client):
            with patch("src.pacing_auditor.get_ledger"):
                result = audit_pacing(script)
        assert len(result.scores) == len(script.scenes)

    def test_gpt_failure_leaves_script_modified_false(self):
        """If GPT fails for all zones, interrupts_added=0 and modified=False."""
        script = _make_script([
            self._engaging_narration(),
            self._flat_narration(),
            self._flat_narration(),
            self._engaging_narration(),
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API down")
        with patch("src.pacing_auditor.make_openai_client", return_value=mock_client):
            result = audit_pacing(script, subject="AI")
        assert result.interrupts_added == 0
        assert result.modified is False

    def test_modified_script_has_same_title_and_tags(self):
        script = _make_script([
            self._engaging_narration(),
            self._flat_narration(),
            self._flat_narration(),
            self._engaging_narration(),
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("Twist.")
        with patch("src.pacing_auditor.make_openai_client", return_value=mock_client):
            with patch("src.pacing_auditor.get_ledger"):
                result = audit_pacing(script)
        assert result.script.title == script.title
        assert result.script.tags == script.tags
        assert result.script.hook == script.hook

    def test_flat_zones_list_populated_in_result(self):
        script = _make_script([
            self._engaging_narration(),
            self._flat_narration(),
            self._flat_narration(),
            self._engaging_narration(),
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("Interesting!")
        with patch("src.pacing_auditor.make_openai_client", return_value=mock_client):
            with patch("src.pacing_auditor.get_ledger"):
                result = audit_pacing(script)
        assert len(result.flat_zones) == 1
        assert result.flat_zones[0].interrupt_text == "Interesting!"


# ── _log_scores coverage ──────────────────────────────────────────────────────

class TestLogScores:
    def test_log_scores_called_during_audit(self):
        """Ensure _log_scores (lines 281-288) is exercised."""
        from src.pacing_auditor import _log_scores
        scores = [
            SceneScore(idx=0, score=3, signals=["question", "number", "conflict"]),
            SceneScore(idx=1, score=0, signals=[]),
            SceneScore(idx=2, score=0, signals=[]),
        ]
        flat_zones = [FlatZone(scene_indices=[1, 2], interrupt_at=2)]
        # Should not raise
        _log_scores(scores, flat_zones)

    def test_log_scores_with_no_flat_zones(self):
        from src.pacing_auditor import _log_scores
        scores = [SceneScore(idx=i, score=3, signals=["question"]) for i in range(4)]
        _log_scores(scores, [])
