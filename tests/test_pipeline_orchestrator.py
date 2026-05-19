"""Tests for PipelineOrchestrator and pipeline helpers.

Helpers (pipeline_helpers.py) are stateless and fully unit-testable.
PipelineOrchestrator is an integration boundary (20+ deps); these are
smoke tests that mock every external call and verify control flow only.
conftest.py stubs numpy and Google SDK modules.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

# Stub additional deps the orchestrator imports at module level
for _m in (
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
    "elevenlabs",
    "elevenlabs.client",
):
    sys.modules.setdefault(_m, MagicMock())

from src.decision_engine_v3 import UnifiedStrategy
from src.pipeline_helpers import (
    _build_scene_map,
    _build_v3_context,
    _classify_hook_type,
    _get_intro_duration,
    _get_shared_outro,
    _load_audio_durations,
    _load_cached_script,
    _load_retention_state,
    _needs_video_rebuild,
    _save_retention_state,
    _setup_logging,
    _unified_strategy_to_content_strategy,
)
from src.script_generator import Scene, VideoScript

# ── _classify_hook_type ───────────────────────────────────────────────────────

class TestClassifyHookType:
    def test_conflict_on_problem_keyword(self):
        assert _classify_hook_type("There is a real problem with this model") == "conflict"

    def test_conflict_on_fail_keyword(self):
        assert _classify_hook_type("Why GPT-4 failed on this benchmark") == "conflict"

    def test_conflict_on_vs_keyword(self):
        assert _classify_hook_type("GPT-4 vs Claude: who wins?") == "conflict"

    def test_conflict_on_threat_keyword(self):
        assert _classify_hook_type("AI is a threat to jobs") == "conflict"

    def test_curiosity_on_nobody_keyword(self):
        assert _classify_hook_type("Nobody is talking about this") == "curiosity"

    def test_curiosity_on_secret_keyword(self):
        assert _classify_hook_type("The hidden secret of neural scaling") == "curiosity"

    def test_curiosity_on_question_mark(self):
        assert _classify_hook_type("Is this the end of GPT?") == "curiosity"

    def test_curiosity_on_wait_keyword(self):
        assert _classify_hook_type("Wait — this changes everything") == "curiosity"

    def test_simple_on_plain_statement(self):
        assert _classify_hook_type("OpenAI released GPT-5 today") == "simple"

    def test_simple_on_empty_string(self):
        assert _classify_hook_type("") == "simple"

    def test_conflict_takes_priority_over_curiosity(self):
        # "problem" is a conflict word; "secret" is curiosity — conflict wins
        hook = "The secret problem with AI scaling"
        assert _classify_hook_type(hook) == "conflict"

    def test_case_insensitive(self):
        assert _classify_hook_type("THE PROBLEM IS REAL") == "conflict"
        assert _classify_hook_type("NOBODY KNOWS THIS") == "curiosity"


# ── _build_scene_map ──────────────────────────────────────────────────────────

def _scene_dict(idx: int, dur: int = 60, intent: str = "explain") -> dict:
    return {"idx": idx, "duration_sec": dur, "scene_intent": intent, "scene_type": "image"}


class TestBuildSceneMap:
    def test_empty_scenes_returns_empty(self):
        assert _build_scene_map([], curve_len=10) == {}

    def test_zero_curve_len_returns_empty(self):
        assert _build_scene_map([_scene_dict(0)], curve_len=0) == {}

    def test_every_bucket_mapped_for_single_scene(self):
        result = _build_scene_map([_scene_dict(0)], curve_len=5)
        assert set(result.keys()) == {0, 1, 2, 3, 4}

    def test_bucket_scene_idx_values_are_valid(self):
        scenes = [_scene_dict(i, dur=30) for i in range(3)]
        result = _build_scene_map(scenes, curve_len=9)
        scene_indices = {v["scene_idx"] for v in result.values()}
        assert scene_indices <= {0, 1, 2}

    def test_intent_preserved(self):
        scenes = [_scene_dict(0, intent="shock")]
        result = _build_scene_map(scenes, curve_len=3)
        assert all(v["intent"] == "shock" for v in result.values())

    def test_longer_scene_gets_more_buckets(self):
        scenes = [_scene_dict(0, dur=90), _scene_dict(1, dur=30)]
        result = _build_scene_map(scenes, curve_len=12)
        buckets_0 = sum(1 for v in result.values() if v["scene_idx"] == 0)
        buckets_1 = sum(1 for v in result.values() if v["scene_idx"] == 1)
        assert buckets_0 > buckets_1

    def test_defaults_when_duration_missing(self):
        scene = {"scene_intent": "explain", "scene_type": "image"}  # no duration_sec
        result = _build_scene_map([scene], curve_len=4)
        assert len(result) > 0


# ── _unified_strategy_to_content_strategy ─────────────────────────────────────

def _strategy(mode: str = "stable", hook_aggressiveness: float = 0.5) -> UnifiedStrategy:
    return UnifiedStrategy(mode=mode, hook_aggressiveness=hook_aggressiveness)


class TestUnifiedStrategyToContentStrategy:
    def test_retention_fix_maps_to_safe(self):
        cs = _unified_strategy_to_content_strategy(_strategy("retention_fix"))
        assert cs.mode == "safe"

    def test_stable_maps_to_balanced(self):
        cs = _unified_strategy_to_content_strategy(_strategy("stable"))
        assert cs.mode == "balanced"

    def test_growth_maps_to_growth(self):
        cs = _unified_strategy_to_content_strategy(_strategy("growth"))
        assert cs.mode == "growth"

    def test_packaging_focus_maps_to_growth(self):
        cs = _unified_strategy_to_content_strategy(_strategy("packaging_focus"))
        assert cs.mode == "growth"

    def test_unknown_mode_defaults_to_balanced(self):
        cs = _unified_strategy_to_content_strategy(_strategy("unknown_mode"))
        assert cs.mode == "balanced"

    def test_exploration_rate_decreases_with_aggressiveness(self):
        low  = _unified_strategy_to_content_strategy(_strategy(hook_aggressiveness=0.0))
        high = _unified_strategy_to_content_strategy(_strategy(hook_aggressiveness=1.0))
        assert low.exploration_rate > high.exploration_rate

    def test_exploration_rate_clamped_between_0_1_and_0_4(self):
        for agr in (0.0, 0.5, 1.0):
            cs = _unified_strategy_to_content_strategy(_strategy(hook_aggressiveness=agr))
            assert 0.1 <= cs.exploration_rate <= 0.4

    def test_retention_fix_has_high_confidence(self):
        cs = _unified_strategy_to_content_strategy(_strategy("retention_fix"))
        assert cs.confidence >= 0.7

    def test_angle_and_format_weights_empty(self):
        cs = _unified_strategy_to_content_strategy(_strategy())
        assert cs.angle_weights == {}
        assert cs.format_weights == {}


# ── _load_retention_state / _save_retention_state ─────────────────────────────

class TestRetentionState:
    def test_returns_empty_dict_when_file_absent(self, tmp_path):
        result = _load_retention_state(tmp_path)
        assert result == {}

    def test_loads_saved_state(self, tmp_path):
        (tmp_path / "retention_state.json").write_text(
            json.dumps({"corrections": [], "adjustments": {"foo": 1}})
        )
        result = _load_retention_state(tmp_path)
        assert result["adjustments"] == {"foo": 1}

    def test_returns_empty_on_corrupt_json(self, tmp_path):
        (tmp_path / "retention_state.json").write_text("not-json{")
        result = _load_retention_state(tmp_path)
        assert result == {}

    def test_save_and_reload_roundtrip(self, tmp_path):
        corrections = [
            MagicMock(to_dict=lambda: {"scene_idx": 0, "drop": {"delta": 0.2}})
        ]
        _save_retention_state(tmp_path, corrections, {"mood": "flat"})
        state = _load_retention_state(tmp_path)
        assert state["adjustments"] == {"mood": "flat"}
        assert len(state["corrections"]) == 1

    def test_save_creates_data_dir(self, tmp_path):
        subdir = tmp_path / "nested" / "data"
        _save_retention_state(subdir, [], {})
        assert (subdir / "retention_state.json").exists()


# ── _load_cached_script ───────────────────────────────────────────────────────

class TestLoadCachedScript:
    def _write_script_json(self, path: Path, n_scenes: int = 2) -> None:
        data = {
            "title": "Test",
            "description": "desc",
            "tags": ["ai"],
            "hook": "hook text",
            "hook_variants": [],
            "scenes": [
                {
                    "heading": f"Scene {i}",
                    "narration": "text",
                    "visual_prompt": "prompt",
                    "duration_sec": 30,
                }
                for i in range(n_scenes)
            ],
        }
        path.write_text(json.dumps(data))

    def test_returns_none_when_file_absent(self, tmp_path):
        assert _load_cached_script(tmp_path / "missing.json") is None

    def test_returns_video_script(self, tmp_path):
        p = tmp_path / "script.json"
        self._write_script_json(p)
        result = _load_cached_script(p)
        assert isinstance(result, VideoScript)

    def test_scene_count_correct(self, tmp_path):
        p = tmp_path / "script.json"
        self._write_script_json(p, n_scenes=3)
        result = _load_cached_script(p)
        assert len(result.scenes) == 3

    def test_scene_idx_assigned_sequentially(self, tmp_path):
        p = tmp_path / "script.json"
        self._write_script_json(p, n_scenes=3)
        result = _load_cached_script(p)
        assert [s.idx for s in result.scenes] == [0, 1, 2]

    def test_title_preserved(self, tmp_path):
        p = tmp_path / "script.json"
        self._write_script_json(p)
        result = _load_cached_script(p)
        assert result.title == "Test"


# ── _build_v3_context ─────────────────────────────────────────────────────────

class TestBuildV3Context:
    def test_returns_dict_with_required_keys(self):
        perf_store = MagicMock()
        perf_store.get_channel_metrics.return_value = {"views": 1000}
        v3_engine = MagicMock()
        v3_engine.load_history.return_value = []

        result = _build_v3_context(
            perf_store, v3_engine,
            thompson_preferred_type="conflict",
            predicted_risks=[{"scene_idx": 1, "probability": 0.3}],
        )

        assert "metrics" in result
        assert "bandit" in result
        assert "prediction" in result
        assert "history" in result

    def test_thompson_type_in_packaging(self):
        perf_store = MagicMock()
        perf_store.get_channel_metrics.return_value = {}
        v3_engine = MagicMock()
        v3_engine.load_history.return_value = []

        result = _build_v3_context(perf_store, v3_engine, "curiosity", [])
        assert result["bandit"]["packaging"]["preferred_variant_type"] == "curiosity"

    def test_none_thompson_type_becomes_empty_string(self):
        perf_store = MagicMock()
        perf_store.get_channel_metrics.return_value = {}
        v3_engine = MagicMock()
        v3_engine.load_history.return_value = []

        result = _build_v3_context(perf_store, v3_engine, None, [])
        assert result["bandit"]["packaging"]["preferred_variant_type"] == ""

    def test_predicted_risks_forwarded(self):
        perf_store = MagicMock()
        perf_store.get_channel_metrics.return_value = {}
        v3_engine = MagicMock()
        v3_engine.load_history.return_value = []
        risks = [{"scene_idx": 2, "probability": 0.7}]

        result = _build_v3_context(perf_store, v3_engine, None, risks)
        assert result["prediction"]["risks"] == risks

    def test_channel_metrics_from_perf_store(self):
        perf_store = MagicMock()
        perf_store.get_channel_metrics.return_value = {"ctr": 0.05}
        v3_engine = MagicMock()
        v3_engine.load_history.return_value = []

        result = _build_v3_context(perf_store, v3_engine, None, [])
        assert result["metrics"] == {"ctr": 0.05}


# ── _get_intro_duration ───────────────────────────────────────────────────────

class TestGetIntroDuration:
    def test_returns_zero_when_none(self):
        assert _get_intro_duration(None) == 0.0

    def test_returns_zero_when_path_not_exists(self, tmp_path):
        assert _get_intro_duration(tmp_path / "missing.mp4") == 0.0

    def test_calls_ffmpeg_duration_when_exists(self, tmp_path):
        intro = tmp_path / "intro.mp4"
        intro.write_bytes(b"fake")
        with patch("src.pipeline_helpers.ffmpeg_utils.duration", return_value=5.0) as mock_dur:
            result = _get_intro_duration(intro)
        assert result == 5.0
        mock_dur.assert_called_once_with(intro)


# ── _get_shared_outro ─────────────────────────────────────────────────────────

class TestGetSharedOutro:
    def test_returns_none_when_outro_absent(self, tmp_path):
        with patch("src.pipeline_helpers.settings") as mock_settings:
            mock_settings.source_dir = tmp_path
            result = _get_shared_outro()
        assert result is None

    def test_returns_path_when_outro_exists(self, tmp_path):
        outro = tmp_path / "ai-news-outro.mp4"
        outro.write_bytes(b"fake")
        with patch("src.pipeline_helpers.settings") as mock_settings:
            mock_settings.source_dir = tmp_path
            result = _get_shared_outro()
        assert result == outro


# ── _load_audio_durations ─────────────────────────────────────────────────────

class TestLoadAudioDurations:
    def _make_script(self, n_scenes: int = 2) -> VideoScript:
        return VideoScript(
            title="Test",
            description="desc",
            tags=[],
            hook="hook",
            scenes=[
                Scene(idx=i, heading=f"S{i}", narration="text", visual_prompt="img", duration_sec=30)
                for i in range(n_scenes)
            ],
        )

    def test_updates_duration_sec_from_audio(self, tmp_path):
        script = self._make_script(2)
        with patch("src.pipeline_helpers.ffmpeg_utils.duration", return_value=10.5):
            _load_audio_durations(script, tmp_path)
        # duration_sec = int(10.5) + 1 = 11
        assert script.scenes[0].duration_sec == 11
        assert script.scenes[1].duration_sec == 11

    def test_uses_correct_filename_pattern(self, tmp_path):
        script = self._make_script(1)
        with patch("src.pipeline_helpers.ffmpeg_utils.duration", return_value=5.0) as mock_dur:
            _load_audio_durations(script, tmp_path)
        called_path = mock_dur.call_args[0][0]
        assert "scene_00.mp3" in str(called_path)


# ── _needs_video_rebuild ──────────────────────────────────────────────────────

class TestNeedsVideoRebuild:
    def test_returns_true_when_video_missing(self, tmp_path):
        video = tmp_path / "final.mp4"
        assert _needs_video_rebuild(video) is True

    def test_returns_false_when_video_has_audio(self, tmp_path):
        video = tmp_path / "final.mp4"
        video.write_bytes(b"fake")
        (tmp_path / "assembled").mkdir()
        with patch("src.pipeline_helpers.ffmpeg_utils.has_audio_stream", return_value=True):
            result = _needs_video_rebuild(video)
        assert result is False

    def test_returns_true_when_legacy_end_card_png(self, tmp_path):
        video = tmp_path / "final.mp4"
        video.write_bytes(b"fake")
        assembled = tmp_path / "assembled"
        assembled.mkdir()
        (assembled / "end_card.png").write_bytes(b"fake")
        with patch("src.pipeline_helpers.ffmpeg_utils.has_audio_stream", return_value=True):
            result = _needs_video_rebuild(video)
        assert result is True
        assert not video.exists()

    def test_returns_true_when_legacy_end_card_mp4(self, tmp_path):
        video = tmp_path / "final.mp4"
        video.write_bytes(b"fake")
        assembled = tmp_path / "assembled"
        assembled.mkdir()
        (assembled / "end_card.mp4").write_bytes(b"fake")
        with patch("src.pipeline_helpers.ffmpeg_utils.has_audio_stream", return_value=True):
            result = _needs_video_rebuild(video)
        assert result is True

    def test_returns_true_when_legacy_title_card(self, tmp_path):
        video = tmp_path / "final.mp4"
        video.write_bytes(b"fake")
        assembled = tmp_path / "assembled"
        assembled.mkdir()
        (assembled / "title_01.mp4").write_bytes(b"fake")
        with patch("src.pipeline_helpers.ffmpeg_utils.has_audio_stream", return_value=True):
            result = _needs_video_rebuild(video)
        assert result is True

    def test_returns_true_when_probe_fails(self, tmp_path):
        video = tmp_path / "final.mp4"
        video.write_bytes(b"fake")
        (tmp_path / "assembled").mkdir()
        with patch("src.pipeline_helpers.ffmpeg_utils.has_audio_stream", side_effect=Exception("probe error")):
            result = _needs_video_rebuild(video)
        assert result is True

    def test_returns_true_when_video_has_no_audio(self, tmp_path):
        video = tmp_path / "final.mp4"
        video.write_bytes(b"fake")
        (tmp_path / "assembled").mkdir()
        with patch("src.pipeline_helpers.ffmpeg_utils.has_audio_stream", return_value=False):
            result = _needs_video_rebuild(video)
        assert result is True

    def test_intro_outro_audio_missing_triggers_rebuild(self, tmp_path):
        video = tmp_path / "final.mp4"
        video.write_bytes(b"fake")
        assembled = tmp_path / "assembled"
        assembled.mkdir()
        (assembled / "intro_resized.mp4").write_bytes(b"fake")

        def _has_audio(path):
            if "intro_resized" in str(path):
                return False
            return True

        with patch("src.pipeline_helpers.ffmpeg_utils.has_audio_stream", side_effect=_has_audio):
            result = _needs_video_rebuild(video)
        assert result is True


# ── _setup_logging ────────────────────────────────────────────────────────────

class TestSetupLogging:
    def test_calls_logger_add_three_times(self, tmp_path):
        with patch("src.pipeline_helpers.logger") as mock_logger:
            _setup_logging(tmp_path)
        assert mock_logger.remove.called
        # stderr sink + text file sink + JSON sink
        assert mock_logger.add.call_count == 3


# ── PipelineOrchestrator smoke tests ─────────────────────────────────────────
# Each test mocks external I/O at the src.pipeline_orchestrator namespace.
# We test control-flow only: which methods get called, what ends up in
# self._summary, and how errors propagate.

from src.media_builder import _MediaBuilder
from src.pipeline_orchestrator import PipelineOrchestrator
from src.publish_step import _PublishStep
from src.quality_gate import QualityGateError


def _fake_script() -> VideoScript:
    scenes = [
        Scene(idx=i, heading=f"Heading {i}", narration="narration text",
              visual_prompt="robot", duration_sec=10.0)
        for i in range(3)
    ]
    return VideoScript(
        title="AI Test Episode",
        hook="This is the hook.",
        scenes=scenes,
        description="Episode description.",
        tags=["ai", "news"],
    )


@pytest.fixture()
def orch(tmp_path) -> PipelineOrchestrator:
    """PipelineOrchestrator instance with all internal state pre-populated."""
    o = PipelineOrchestrator()
    o._run_dir = tmp_path
    o._dry_run = False
    o._skip_upload = False
    o._summary = {"date": "2026-05-17", "run_dir": str(tmp_path)}
    o._ledger = MagicMock()
    o._ledger.save.return_value = {"total_usd": 0.42}
    o._cp = MagicMock()
    o._cp.is_done.return_value = False
    o._cp.metadata.return_value = {}
    o._live = MagicMock()
    o._observer = MagicMock()
    o._observer.finish.return_value = {"steps": [], "total_sec": 10.0}
    o._seen = MagicMock()
    o._seen.stats.return_value = {"in_ttl_window": 5, "total_seen": 20}
    o._seen.recent_titles.return_value = []
    o._seen.filter_new.side_effect = lambda x: x
    o._script = _fake_script()
    o._news = []
    o._history = []
    o._feedback_history = []
    o._subtitle_path = None
    o._en_intro = None
    o._en_outro = None
    o._long_video = tmp_path / "final_video.mp4"
    o._long_video.write_bytes(b"fake")
    o._short_video = None
    o._thumbnail = tmp_path / "thumbnail_default.jpg"
    o._thumbnail.write_bytes(b"fake")
    o._auto_strategy = None
    o._auto_actions = []
    o._auto_insights = []
    o._thompson_preferred_type = None
    o._v3_strategy = None
    # _builder and _publisher are created during run(); pre-populate for step tests
    o._builder = MagicMock()
    o._builder.subtitle_path = None
    o._builder.en_intro = None
    o._builder.en_outro = None
    o._builder.long_video = tmp_path / "final_video.mp4"
    o._builder.short_video = None
    o._builder.thumbnail = tmp_path / "thumbnail_default.jpg"
    o._publisher = MagicMock()
    return o


# ── Helper factories for new classes ─────────────────────────────────────────

def _make_builder(orch, tmp_path) -> _MediaBuilder:
    return _MediaBuilder(
        script=orch._script,
        run_dir=tmp_path,
        cp=orch._cp,
        observer=orch._observer,
        live=orch._live,
        seen=orch._seen,
        news=orch._news,
        summary=orch._summary,
        thompson_preferred_type=None,
    )


def _make_publisher(orch, tmp_path) -> _PublishStep:
    return _PublishStep(
        script=orch._script,
        run_dir=tmp_path,
        cp=orch._cp,
        observer=orch._observer,
        live=orch._live,
        seen=orch._seen,
        news=orch._news,
        summary=orch._summary,
        skip_upload=orch._skip_upload,
        auto_strategy=orch._auto_strategy,
        auto_actions=orch._auto_actions,
        long_video=orch._long_video,
        short_video=orch._short_video,
        thumbnail=orch._thumbnail,
        subtitle_path=orch._subtitle_path,
        feedback_history=orch._feedback_history,
    )


# ── run() control flow ────────────────────────────────────────────────────────

_INFRA = [
    "src.pipeline_orchestrator._setup_logging_helper",
    "src.pipeline_orchestrator.reset_ledger",
    "src.pipeline_orchestrator.PipelineCheckpoint",
    "src.pipeline_orchestrator.LiveState",
    "src.pipeline_orchestrator.PipelineObserver",
    "src.pipeline_orchestrator.SeenStories",
]


def _patch_infra(tmp_path):
    """Return a dict of configured MagicMocks for run() infrastructure."""
    fake_ledger = MagicMock()
    fake_ledger.save.return_value = {"total_usd": 0.0}
    mock_cp = MagicMock()
    mock_cp.is_done.return_value = False
    mock_cp.metadata.return_value = {}
    mock_obs = MagicMock()
    mock_obs.finish.return_value = {"steps": [], "total_sec": 5.0}
    mock_seen = MagicMock()
    mock_seen.stats.return_value = {"in_ttl_window": 0, "total_seen": 0}
    mock_seen.recent_titles.return_value = []
    mock_seen.filter_new.side_effect = lambda x: x
    return {
        "ledger": fake_ledger,
        "cp": mock_cp,
        "obs": mock_obs,
        "seen": mock_seen,
        "live": MagicMock(),
    }


class TestPipelineOrchestratorRun:
    """Smoke tests for PipelineOrchestrator.run()."""

    def test_dry_run_returns_summary_with_date(self, tmp_path):
        mocks = _patch_infra(tmp_path)
        news_item = MagicMock()
        news_item.title = "Story A"
        with (
            patch("src.pipeline_orchestrator._setup_logging_helper"),
            patch("src.pipeline_orchestrator.reset_ledger", return_value=mocks["ledger"]),
            patch("src.pipeline_orchestrator.PipelineCheckpoint", return_value=mocks["cp"]),
            patch("src.pipeline_orchestrator.LiveState", return_value=mocks["live"]),
            patch("src.pipeline_orchestrator.PipelineObserver", return_value=mocks["obs"]),
            patch("src.pipeline_orchestrator.SeenStories", return_value=mocks["seen"]),
            patch("src.pipeline_orchestrator.scrape_all", return_value=[news_item]),
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=[news_item]),
        ):
            result = PipelineOrchestrator().run(dry_run=True)
        assert "date" in result

    def test_dry_run_stops_after_scrape(self, tmp_path):
        mocks = _patch_infra(tmp_path)
        with (
            patch("src.pipeline_orchestrator._setup_logging_helper"),
            patch("src.pipeline_orchestrator.reset_ledger", return_value=mocks["ledger"]),
            patch("src.pipeline_orchestrator.PipelineCheckpoint", return_value=mocks["cp"]),
            patch("src.pipeline_orchestrator.LiveState", return_value=mocks["live"]),
            patch("src.pipeline_orchestrator.PipelineObserver", return_value=mocks["obs"]),
            patch("src.pipeline_orchestrator.SeenStories", return_value=mocks["seen"]),
            patch("src.pipeline_orchestrator.scrape_all", return_value=[]) as mock_scrape,
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=[]),
            patch("src.pipeline_orchestrator._ScriptStep") as mock_step,
        ):
            PipelineOrchestrator().run(dry_run=True)
        mock_scrape.assert_called_once()
        mock_step.assert_not_called()

    def test_exception_in_step_populates_summary_error(self, tmp_path):
        mocks = _patch_infra(tmp_path)
        with (
            patch("src.pipeline_orchestrator._setup_logging_helper"),
            patch("src.pipeline_orchestrator.reset_ledger", return_value=mocks["ledger"]),
            patch("src.pipeline_orchestrator.PipelineCheckpoint", return_value=mocks["cp"]),
            patch("src.pipeline_orchestrator.LiveState", return_value=mocks["live"]),
            patch("src.pipeline_orchestrator.PipelineObserver", return_value=mocks["obs"]),
            patch("src.pipeline_orchestrator.SeenStories", return_value=mocks["seen"]),
            patch("src.pipeline_orchestrator.scrape_all", side_effect=RuntimeError("net fail")),
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=[]),
            patch("src.pipeline_orchestrator.notify_failure"),
        ):
            o = PipelineOrchestrator()
            with pytest.raises(RuntimeError, match="net fail"):
                o.run()
        assert o._summary.get("status") == "failed"
        assert "net fail" in o._summary.get("error", "")

    def test_quality_gate_error_propagates_and_sets_status(self, tmp_path):
        mocks = _patch_infra(tmp_path)
        fake_news = MagicMock()
        fake_news.title = "X"
        with (
            patch("src.pipeline_orchestrator._setup_logging_helper"),
            patch("src.pipeline_orchestrator.reset_ledger", return_value=mocks["ledger"]),
            patch("src.pipeline_orchestrator.PipelineCheckpoint", return_value=mocks["cp"]),
            patch("src.pipeline_orchestrator.LiveState", return_value=mocks["live"]),
            patch("src.pipeline_orchestrator.PipelineObserver", return_value=mocks["obs"]),
            patch("src.pipeline_orchestrator.SeenStories", return_value=mocks["seen"]),
            patch("src.pipeline_orchestrator.scrape_all", return_value=[fake_news]),
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=[fake_news]),
            patch("src.pipeline_orchestrator.notify_failure"),
            patch("src.pipeline_orchestrator._MediaBuilder"),
            patch("src.pipeline_orchestrator._PublishStep"),
        ):
            o = PipelineOrchestrator()
            # Patch step methods to skip to quality gate quickly
            o._step_script = MagicMock()
            o._step_voice = MagicMock()
            o._step_subtitles = MagicMock()
            o._step_video = MagicMock()
            o._step_thumbnail = MagicMock()
            o._step_quality_gate = MagicMock(
                side_effect=QualityGateError("hook too short")
            )
            with pytest.raises(QualityGateError):
                o.run()
        assert o._summary.get("status") == "quality_gate_failed"

    def test_successful_run_calls_finalize(self, tmp_path):
        mocks = _patch_infra(tmp_path)
        fake_news = MagicMock()
        fake_news.title = "X"
        with (
            patch("src.pipeline_orchestrator._setup_logging_helper"),
            patch("src.pipeline_orchestrator.reset_ledger", return_value=mocks["ledger"]),
            patch("src.pipeline_orchestrator.PipelineCheckpoint", return_value=mocks["cp"]),
            patch("src.pipeline_orchestrator.LiveState", return_value=mocks["live"]),
            patch("src.pipeline_orchestrator.PipelineObserver", return_value=mocks["obs"]),
            patch("src.pipeline_orchestrator.SeenStories", return_value=mocks["seen"]),
            patch("src.pipeline_orchestrator.scrape_all", return_value=[fake_news]),
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=[fake_news]),
            patch("src.pipeline_orchestrator._MediaBuilder"),
            patch("src.pipeline_orchestrator._PublishStep"),
        ):
            o = PipelineOrchestrator()
            o._step_script = MagicMock()
            o._step_voice = MagicMock()
            o._step_subtitles = MagicMock()
            o._step_video = MagicMock()
            o._step_thumbnail = MagicMock()
            o._step_quality_gate = MagicMock()
            o._step_upload_en = MagicMock()
            o._step_shorts_and_ru = MagicMock()
            o._finalize = MagicMock()
            o.run()
        o._finalize.assert_called_once()

    def test_skip_upload_flag_stored_on_instance(self, tmp_path):
        mocks = _patch_infra(tmp_path)
        with (
            patch("src.pipeline_orchestrator._setup_logging_helper"),
            patch("src.pipeline_orchestrator.reset_ledger", return_value=mocks["ledger"]),
            patch("src.pipeline_orchestrator.PipelineCheckpoint", return_value=mocks["cp"]),
            patch("src.pipeline_orchestrator.LiveState", return_value=mocks["live"]),
            patch("src.pipeline_orchestrator.PipelineObserver", return_value=mocks["obs"]),
            patch("src.pipeline_orchestrator.SeenStories", return_value=mocks["seen"]),
            patch("src.pipeline_orchestrator.scrape_all", return_value=[]),
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=[]),
        ):
            o = PipelineOrchestrator()
            o.run(dry_run=True, skip_upload=True)
        assert o._skip_upload is True


# ── _step_scrape ──────────────────────────────────────────────────────────────

class TestStepScrape:

    def test_fresh_scrape_stores_news(self, orch, tmp_path):
        items = [MagicMock(title="A"), MagicMock(title="B")]
        with (
            patch("src.pipeline_orchestrator.scrape_all", return_value=items),
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=items),
        ):
            orch._step_scrape()
        assert orch._news == items

    def test_viral_fallback_when_pick_returns_empty(self, orch):
        items = [MagicMock(title="A")]
        with (
            patch("src.pipeline_orchestrator.scrape_all", return_value=items),
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=[]),
        ):
            orch._step_scrape()
        # Falls back to original scrape result after dedup filter
        assert orch._news == items

    def test_viral_news_replaces_original_when_found(self, orch):
        original = [MagicMock(title="A"), MagicMock(title="B")]
        viral = [MagicMock(title="V")]
        with (
            patch("src.pipeline_orchestrator.scrape_all", return_value=original),
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=viral),
        ):
            orch._step_scrape()
        assert orch._news == viral

    def test_checkpoint_hit_still_calls_scrape_all(self, orch):
        orch._cp.is_done.return_value = True
        orch._cp.metadata.return_value = {"scraped_items": 3}
        with (
            patch("src.pipeline_orchestrator.scrape_all", return_value=[]) as mock_scrape,
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=[]),
        ):
            orch._step_scrape()
        mock_scrape.assert_called_once()

    def test_summary_gets_scraped_items_count(self, orch):
        items = [MagicMock(title=f"S{i}") for i in range(4)]
        with (
            patch("src.pipeline_orchestrator.scrape_all", return_value=items),
            patch("src.pipeline_orchestrator.pick_viral_news", return_value=items),
        ):
            orch._step_scrape()
        assert orch._summary["scraped_items"] == 4


# ── _step_voice ───────────────────────────────────────────────────────────────

class TestStepVoice:

    def test_checkpoint_hit_loads_audio_durations_only(self, orch, tmp_path):
        orch._cp.is_done.return_value = True
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        builder = _make_builder(orch, tmp_path)
        with (
            patch("src.media_builder._load_audio_durations") as mock_load,
            patch("src.media_builder.asyncio") as mock_asyncio,
        ):
            builder.build_voice()
        mock_load.assert_called_once()
        mock_asyncio.run.assert_not_called()

    def test_cached_audio_dir_skips_synthesis(self, orch, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        for i in range(3):
            (audio_dir / f"scene_{i:02d}.mp3").write_bytes(b"audio")
        builder = _make_builder(orch, tmp_path)
        with (
            patch("src.media_builder._load_audio_durations") as mock_load,
            patch("src.media_builder.asyncio") as mock_asyncio,
        ):
            builder.build_voice()
        mock_asyncio.run.assert_not_called()
        mock_load.assert_called_once()

    def test_fresh_voice_runs_async_synthesis(self, orch, tmp_path):
        builder = _make_builder(orch, tmp_path)
        with patch("src.media_builder.asyncio") as mock_asyncio:
            builder.build_voice()
        mock_asyncio.run.assert_called_once()

    def test_summary_gets_total_duration_sec(self, orch, tmp_path):
        builder = _make_builder(orch, tmp_path)
        with patch("src.media_builder.asyncio"):
            builder.build_voice()
        assert "total_duration_sec" in orch._summary
        assert orch._summary["total_duration_sec"] == 30.0  # 3 scenes × 10s


# ── _step_subtitles ───────────────────────────────────────────────────────────

class TestStepSubtitles:

    def test_subtitle_failure_is_nonfatal(self, orch, tmp_path):
        builder = _make_builder(orch, tmp_path)
        with patch(
            "src.media_builder.generate_subtitles",
            side_effect=RuntimeError("whisper unavailable"),
        ):
            builder.build_subtitles()  # must not raise
        assert builder.subtitle_path is None

    def test_checkpoint_hit_skips_generate_subtitles(self, orch, tmp_path):
        orch._cp.is_done.return_value = True
        builder = _make_builder(orch, tmp_path)
        with patch("src.media_builder.generate_subtitles") as mock_gen:
            builder.build_subtitles()
        mock_gen.assert_not_called()

    def test_subtitle_path_set_on_success(self, orch, tmp_path):
        builder = _make_builder(orch, tmp_path)
        fake_srt = tmp_path / "subtitles.srt"
        with (
            patch("src.media_builder.generate_subtitles", return_value=fake_srt),
            patch("src.media_builder._get_intro_duration", return_value=0.0),
        ):
            builder.build_subtitles()
        assert builder.subtitle_path == fake_srt


# ── _step_video ───────────────────────────────────────────────────────────────

class TestStepVideo:

    def test_fresh_build_calls_build_video(self, orch, tmp_path):
        builder = _make_builder(orch, tmp_path)
        with (
            patch("src.media_builder.build_video") as mock_build,
            patch("src.media_builder._needs_video_rebuild", return_value=False),
            patch.dict("sys.modules", {"src.broll_fetcher": MagicMock(
                get_pexels_credits=MagicMock(return_value=[])
            )}),
        ):
            builder.build_video()
        mock_build.assert_called_once()

    def test_checkpoint_hit_reuses_video(self, orch, tmp_path):
        orch._cp.is_done.return_value = True
        builder = _make_builder(orch, tmp_path)
        with (
            patch("src.media_builder.build_video") as mock_build,
            patch("src.media_builder._needs_video_rebuild", return_value=False),
            patch.dict("sys.modules", {"src.broll_fetcher": MagicMock(
                get_pexels_credits=MagicMock(return_value=[])
            )}),
        ):
            builder.build_video()
        mock_build.assert_not_called()

    def test_rebuild_forced_when_video_needs_rebuild(self, orch, tmp_path):
        orch._cp.is_done.return_value = True
        builder = _make_builder(orch, tmp_path)
        with (
            patch("src.media_builder.build_video") as mock_build,
            patch("src.media_builder._needs_video_rebuild", return_value=True),
            patch.dict("sys.modules", {"src.broll_fetcher": MagicMock(
                get_pexels_credits=MagicMock(return_value=[])
            )}),
        ):
            builder.build_video()
        mock_build.assert_called_once()


# ── _step_quality_gate ────────────────────────────────────────────────────────

class TestStepQualityGate:

    def test_quality_score_stored_in_summary(self, orch, tmp_path):
        publisher = _make_publisher(orch, tmp_path)
        fake_report = MagicMock(score=0.87, warnings=[])
        with patch("src.publish_step.run_gate", return_value=fake_report):
            publisher.run_quality_gate()
        assert orch._summary["quality_score"] == 0.87

    def test_quality_warnings_stored_in_summary(self, orch, tmp_path):
        publisher = _make_publisher(orch, tmp_path)
        fake_report = MagicMock(score=0.6, warnings=["hook too short"])
        with patch("src.publish_step.run_gate", return_value=fake_report):
            publisher.run_quality_gate()
        assert orch._summary["quality_warnings"] == ["hook too short"]

    def test_quality_gate_error_propagates(self, orch, tmp_path):
        publisher = _make_publisher(orch, tmp_path)
        with patch(
            "src.publish_step.run_gate",
            side_effect=QualityGateError("missing audio"),
        ):
            with pytest.raises(QualityGateError, match="missing audio"):
                publisher.run_quality_gate()


# ── _step_upload_en ───────────────────────────────────────────────────────────

class TestStepUploadEn:

    def test_skip_upload_sets_built_not_uploaded(self, orch, tmp_path):
        orch._skip_upload = True
        publisher = _make_publisher(orch, tmp_path)
        with patch("src.publish_step.publish_episode") as mock_pub:
            publisher.upload_en()
        assert orch._summary["status"] == "built_not_uploaded"
        mock_pub.assert_not_called()

    def test_checkpoint_done_skipped_restores_status(self, orch, tmp_path):
        orch._cp.is_done.return_value = True
        orch._cp.metadata.return_value = {"skipped": True}
        publisher = _make_publisher(orch, tmp_path)
        with patch("src.publish_step.publish_episode") as mock_pub:
            publisher.upload_en()
        assert orch._summary["status"] == "built_not_uploaded"
        mock_pub.assert_not_called()

    def test_publish_episode_called_when_not_skipping(self, orch, tmp_path):
        publisher = _make_publisher(orch, tmp_path)
        with (
            patch("src.publish_step.publish_episode",
                  return_value={"video_id": "vid123"}) as mock_pub,
            patch("src.publish_step.get_recommendations", return_value={}),
            patch("src.publish_step.record_usage"),
            patch("src.publish_step.record_thumbnail_usage"),
            patch("src.publish_step.save_result"),
        ):
            publisher.upload_en()
        mock_pub.assert_called_once()
        assert orch._summary["video_id"] == "vid123"
        assert orch._summary["status"] == "published"

    def test_checkpoint_done_published_restores_status(self, orch, tmp_path):
        orch._cp.is_done.return_value = True
        orch._cp.metadata.return_value = {"skipped": False, "video_id": "vid999"}
        publisher = _make_publisher(orch, tmp_path)
        with patch("src.publish_step.publish_episode") as mock_pub:
            publisher.upload_en()
        assert orch._summary["status"] == "published"
        mock_pub.assert_not_called()


# ── _finalize ─────────────────────────────────────────────────────────────────

class TestFinalize:

    def _run_finalize(self, orch, *, slow=None, budget_exceeded=False):
        fake_tracker = MagicMock()
        fake_tracker.slow_steps.return_value = slow or []
        fake_tracker.step_summary.return_value = []

        fake_status = MagicMock()
        fake_status.any_exceeded = budget_exceeded

        with (
            patch("src.pipeline_orchestrator.LatencyTracker", return_value=fake_tracker),
            patch("src.pipeline_orchestrator.BudgetGuard") as mock_bg_cls,
            patch("src.pipeline_orchestrator.notify_slow_steps") as mock_slow,
            patch("src.pipeline_orchestrator.notify_budget_summary"),
            patch("src.pipeline_orchestrator.notify_budget_alert") as mock_alert,
            patch("src.pipeline_orchestrator.notify_success") as mock_success,
        ):
            mock_bg_cls.return_value.check.return_value = fake_status
            orch._finalize()
        return mock_slow, mock_alert, mock_success, fake_tracker

    def test_cost_usd_added_to_summary(self, orch):
        self._run_finalize(orch)
        assert orch._summary["cost_usd"] == 0.42

    def test_no_slow_steps_skips_notification(self, orch):
        mock_slow, _, _, _ = self._run_finalize(orch, slow=[])
        mock_slow.assert_not_called()

    def test_slow_steps_triggers_notification(self, orch):
        slow_entry = {"name": "video", "duration_sec": 1200, "threshold_sec": 900, "p95_sec": None}
        mock_slow, _, _, _ = self._run_finalize(orch, slow=[slow_entry])
        mock_slow.assert_called_once()

    def test_budget_alert_sent_when_exceeded(self, orch):
        _, mock_alert, _, _ = self._run_finalize(orch, budget_exceeded=True)
        mock_alert.assert_called_once()

    def test_no_budget_alert_when_not_exceeded(self, orch):
        _, mock_alert, _, _ = self._run_finalize(orch, budget_exceeded=False)
        mock_alert.assert_not_called()

    def test_notify_success_always_called(self, orch):
        _, _, mock_success, _ = self._run_finalize(orch)
        mock_success.assert_called_once()


# ── _step_shorts_and_ru ───────────────────────────────────────────────────────

class TestStepShortsAndRu:
    """_step_shorts_and_ru runs shorts and RU upload concurrently via asyncio."""

    def test_both_steps_called(self, orch):
        orch._step_shorts_experiments = MagicMock()
        orch._step_upload_ru = MagicMock()
        orch._step_shorts_and_ru()
        orch._step_shorts_experiments.assert_called_once()
        orch._step_upload_ru.assert_called_once()

    def test_shorts_exception_propagates(self, orch):
        orch._step_shorts_experiments = MagicMock(side_effect=RuntimeError("shorts failed"))
        orch._step_upload_ru = MagicMock()
        with pytest.raises(RuntimeError, match="shorts failed"):
            orch._step_shorts_and_ru()

    def test_ru_exception_propagates(self, orch):
        orch._step_shorts_experiments = MagicMock()
        orch._step_upload_ru = MagicMock(side_effect=RuntimeError("ru failed"))
        with pytest.raises(RuntimeError, match="ru failed"):
            orch._step_shorts_and_ru()

    def test_both_run_when_neither_raises(self, orch):
        calls: list[str] = []
        orch._step_shorts_experiments = MagicMock(side_effect=lambda: calls.append("shorts"))
        orch._step_upload_ru = MagicMock(side_effect=lambda: calls.append("ru"))
        orch._step_shorts_and_ru()
        assert sorted(calls) == ["ru", "shorts"]
