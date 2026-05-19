"""Tests for SystemOrchestrator.

Covers:
- CycleResult.to_dict()
- run_cycle() — happy path, each pipeline step called correctly
- run_cycle() — fail-safe: exceptions become CycleResult(success=False)
- _run_cycle_unsafe() — strategy mode, scene policy id, packaging hook propagated
- drop-predictor integration — high_risk count, pre-emptive corrections applied
- drop hints applied to strategy (pace=fast, force_avatar_intro)
- render + publish callables invoked and results threaded through
- _persist_cycle() + _load_cycle() — JSONL roundtrip
- learn_from_feedback() — happy path, all components called
- learn_from_feedback() — missing cycle record is a no-op (warning only)
- learn_from_feedback() — no retention curve skips retention engine
- learn_from_feedback() — scene policy "none" skips bandit updates
- learn_from_feedback() — performance store receives correct record
- create_orchestrator() factory — returns a SystemOrchestrator with null renderer/publisher
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from src.system_orchestrator import CycleResult, SystemOrchestrator, create_orchestrator

# ── Helpers / stubs ────────────────────────────────────────────────────────────

def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


@dataclass
class _Scene:
    idx:          int  = 0
    scene_type:   str  = "image"
    scene_intent: str  = "explain"
    duration_sec: int  = 30


@dataclass
class _Strategy:
    mode:         str              = "aggressive"
    pace:         str              = "normal"
    scene_policy: Any              = None
    scene_mix:    dict[str, float] = field(default_factory=dict)


@dataclass
class _Packaging:
    hook:      str              = "Did you know AI is changing everything?"
    title:     str              = "AI Update"
    thumbnail: dict[str, str]  = field(default_factory=dict)


@dataclass
class _ScenePolicyArm:
    id:        str              = "policy_A"
    label:     str              = "balanced"
    scene_mix: dict[str, float] = field(
        default_factory=lambda: {"avatar": 0.3, "image": 0.7}
    )


@dataclass
class _DropPrediction:
    scene_idx:   int   = 0
    probability: float = 0.2


def _run_cycle(orch: SystemOrchestrator, editorial_input: dict,
               corrections=None, hints=None) -> CycleResult:
    """Run one cycle with the drop_predictor module-level functions patched."""
    with patch("src.drop_predictor.apply_preemptive_corrections",
               return_value=corrections or []), \
         patch("src.drop_predictor.suggest_strategy_adjustments",
               return_value=hints or {}):
        return orch.run_cycle(editorial_input)


def _make_orchestrator(
    data_dir:              Path | None = None,
    strategy_mode:         str         = "aggressive",
    scene_policy_id:       str         = "policy_A",
    hook:                  str         = "Test hook",
    high_risk:             int         = 0,
    corrections_applied:   int         = 0,
    renderer_returns:      Any         = Path("/tmp/video.mp4"),
    publisher_returns:     Any         = "vid_abc123",
    drop_hints:            dict        | None = None,
    predictions:           list        | None = None,
) -> tuple[SystemOrchestrator, dict[str, MagicMock]]:
    """Build an orchestrator with all dependencies mocked."""
    d = data_dir or _tmp()

    strategy   = _Strategy(mode=strategy_mode)
    packaging  = _Packaging(hook=hook)
    policy_arm = _ScenePolicyArm(id=scene_policy_id)

    decision_engine = MagicMock()
    decision_engine.decide.return_value = strategy

    packaging_engine = MagicMock()
    packaging_engine.generate.return_value = packaging

    scene_bandit = MagicMock()
    scene_bandit.select_policy.return_value = policy_arm

    scene_engine = MagicMock()

    if predictions is None:
        predictions = [_DropPrediction(probability=0.8) for _ in range(high_risk)] + \
                      [_DropPrediction(probability=0.2) for _ in range(3 - high_risk)]
    drop_predictor = MagicMock()
    drop_predictor.predict_scenes.return_value = predictions

    retention_engine = MagicMock()
    retention_engine.analyze.return_value = []
    retention_engine.suggest_strategy_adjustments.return_value = {}

    video_renderer  = MagicMock(return_value=renderer_returns)
    video_publisher = MagicMock(return_value=publisher_returns)

    performance_store = MagicMock()

    mocks = {
        "decision_engine":   decision_engine,
        "packaging_engine":  packaging_engine,
        "scene_bandit":      scene_bandit,
        "scene_engine":      scene_engine,
        "drop_predictor":    drop_predictor,
        "retention_engine":  retention_engine,
        "video_renderer":    video_renderer,
        "video_publisher":   video_publisher,
        "performance_store": performance_store,
    }

    orch = SystemOrchestrator(
        decision_engine   = decision_engine,
        packaging_engine  = packaging_engine,
        scene_bandit      = scene_bandit,
        scene_engine      = scene_engine,
        drop_predictor    = drop_predictor,
        retention_engine  = retention_engine,
        video_renderer    = video_renderer,
        video_publisher   = video_publisher,
        performance_store = performance_store,
        data_dir          = d,
    )
    return orch, mocks


def _editorial(scenes=None, metrics=None, platform="youtube", video_id=None):
    return {
        "scenes":         scenes or [_Scene(idx=0)],
        "editorial_plan": {"angle": "technical"},
        "platform":       platform,
        "video_id":       video_id,
        "metrics":        metrics or {},
    }


# ── CycleResult ────────────────────────────────────────────────────────────────

class TestCycleResult:
    def test_to_dict_success(self):
        r = CycleResult(
            video_id            = "vid_x",
            strategy_mode       = "aggressive",
            scene_policy_id     = "policy_B",
            packaging_hook      = "Hook text",
            high_risk_scenes    = 2,
            corrections_applied = 1,
        )
        d = r.to_dict()
        assert d["video_id"]            == "vid_x"
        assert d["strategy_mode"]       == "aggressive"
        assert d["scene_policy_id"]     == "policy_B"
        assert d["packaging_hook"]      == "Hook text"
        assert d["high_risk_scenes"]    == 2
        assert d["corrections_applied"] == 1
        assert d["success"]             is True
        assert d["error"]               is None

    def test_to_dict_failure(self):
        r = CycleResult(
            video_id            = None,
            strategy_mode       = "unknown",
            scene_policy_id     = "unknown",
            packaging_hook      = "",
            high_risk_scenes    = 0,
            corrections_applied = 0,
            success             = False,
            error               = "boom",
        )
        d = r.to_dict()
        assert d["success"] is False
        assert d["error"]   == "boom"


# ── run_cycle happy path ───────────────────────────────────────────────────────

class TestRunCycleHappyPath:
    def _run(self, **kw):
        orch, mocks = _make_orchestrator(**kw)
        result = _run_cycle(orch, _editorial())
        return result, orch, mocks

    def test_returns_cycle_result_on_success(self):
        result, _, _ = self._run()
        assert isinstance(result, CycleResult)
        assert result.success is True
        assert result.error   is None

    def test_strategy_mode_propagated(self):
        result, _, _ = self._run(strategy_mode="safe")
        assert result.strategy_mode == "safe"

    def test_scene_policy_id_propagated(self):
        result, _, _ = self._run(scene_policy_id="policy_C")
        assert result.scene_policy_id == "policy_C"

    def test_packaging_hook_propagated(self):
        result, _, _ = self._run(hook="My hook")
        assert result.packaging_hook == "My hook"

    def test_video_id_from_publisher(self):
        result, _, _ = self._run(publisher_returns="vid_999")
        assert result.video_id == "vid_999"

    def test_video_id_none_when_renderer_returns_none(self):
        result, _, mocks = self._run(renderer_returns=None)
        mocks["video_publisher"].assert_not_called()
        assert result.video_id is None

    def test_high_risk_scenes_counted(self):
        preds = [
            _DropPrediction(probability=0.9),
            _DropPrediction(probability=0.7),
            _DropPrediction(probability=0.3),
        ]
        orch, mocks = _make_orchestrator(predictions=preds)
        result = _run_cycle(orch, _editorial())
        assert result.high_risk_scenes == 2

    def test_decision_engine_receives_metrics(self):
        orch, mocks = _make_orchestrator()
        metrics = {"ctr": 0.07, "retention_30s": 0.55}
        _run_cycle(orch, _editorial(metrics=metrics))
        mocks["decision_engine"].decide.assert_called_once()
        _, kwargs = mocks["decision_engine"].decide.call_args
        assert kwargs.get("scene_bandit") is mocks["scene_bandit"]

    def test_packaging_engine_called(self):
        orch, mocks = _make_orchestrator()
        _run_cycle(orch, _editorial())
        mocks["packaging_engine"].generate.assert_called_once()

    def test_scene_engine_assign_called(self):
        orch, mocks = _make_orchestrator()
        _run_cycle(orch, _editorial())
        mocks["scene_engine"].assign_from_config.assert_called_once()

    def test_renderer_called_with_correct_signature(self):
        orch, mocks = _make_orchestrator()
        inp = _editorial()
        _run_cycle(orch, inp)
        call_kwargs = mocks["video_renderer"].call_args.kwargs
        assert "scenes"          in call_kwargs
        assert "packaging"       in call_kwargs
        assert "strategy"        in call_kwargs
        assert "editorial_input" in call_kwargs

    def test_publisher_called_with_video_path(self):
        vpath = Path("/tmp/out.mp4")
        orch, mocks = _make_orchestrator(renderer_returns=vpath)
        _run_cycle(orch, _editorial())
        mocks["video_publisher"].assert_called_once()
        call_kwargs = mocks["video_publisher"].call_args.kwargs
        assert call_kwargs["video_path"] == vpath


# ── run_cycle fail-safe ────────────────────────────────────────────────────────

class TestRunCycleFailSafe:
    def test_exception_returns_failure_result(self):
        orch, mocks = _make_orchestrator()
        mocks["decision_engine"].decide.side_effect = RuntimeError("network down")
        result = orch.run_cycle(_editorial())
        assert result.success is False
        assert "network down" in result.error

    def test_failure_result_has_unknown_mode(self):
        orch, mocks = _make_orchestrator()
        mocks["packaging_engine"].generate.side_effect = ValueError("bad input")
        result = orch.run_cycle(_editorial())
        assert result.strategy_mode   == "unknown"
        assert result.scene_policy_id == "unknown"
        assert result.video_id        is None

    def test_failure_zeroes_counts(self):
        orch, mocks = _make_orchestrator()
        mocks["scene_engine"].assign_from_config.side_effect = Exception("crash")
        result = orch.run_cycle(_editorial())
        assert result.high_risk_scenes    == 0
        assert result.corrections_applied == 0


# ── Drop-hint propagation ──────────────────────────────────────────────────────

class TestDropHints:
    def test_pace_hint_applied_to_strategy(self):
        orch, mocks = _make_orchestrator()
        strategy = mocks["decision_engine"].decide.return_value

        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments",
                   return_value={"pace": "fast"}):
            orch.run_cycle(_editorial())

        assert strategy.pace == "fast"

    def test_force_avatar_intro_sets_first_scene_type(self):
        scene = _Scene(idx=0, scene_type="image")
        orch, mocks = _make_orchestrator()

        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments",
                   return_value={"force_avatar_intro": True}):
            orch.run_cycle(_editorial(scenes=[scene]))

        assert scene.scene_type == "avatar"

    def test_no_force_avatar_when_no_scenes(self):
        orch, mocks = _make_orchestrator()
        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments",
                   return_value={"force_avatar_intro": True}):
            result = orch.run_cycle(_editorial(scenes=[]))
        assert result.success is True

    def test_corrections_applied_count(self):
        corrections = [object(), object(), object()]
        orch, mocks = _make_orchestrator()
        with patch("src.drop_predictor.apply_preemptive_corrections",
                   return_value=corrections), \
             patch("src.drop_predictor.suggest_strategy_adjustments",
                   return_value={}):
            result = orch.run_cycle(_editorial())
        assert result.corrections_applied == 3


# ── Scene bandit "none" policy ─────────────────────────────────────────────────

class TestSceneBanditNonePolicy:
    def test_none_policy_gives_scene_policy_id_none(self):
        orch, mocks = _make_orchestrator()
        mocks["scene_bandit"].select_policy.return_value = None
        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments", return_value={}):
            result = orch.run_cycle(_editorial())
        assert result.scene_policy_id == "none"


# ── Cycle persistence ─────────────────────────────────────────────────────────

class TestCyclePersistence:
    def test_persist_and_load_cycle(self):
        d = _tmp()
        orch, _ = _make_orchestrator(data_dir=d, publisher_returns="vid_123")
        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments", return_value={}):
            orch.run_cycle(_editorial())
        loaded = orch._load_cycle("vid_123")
        assert loaded is not None
        assert loaded["video_id"] == "vid_123"

    def test_load_missing_cycle_returns_none(self):
        d = _tmp()
        orch, _ = _make_orchestrator(data_dir=d)
        result = orch._load_cycle("does_not_exist")
        assert result is None

    def test_multiple_cycles_only_loads_matching(self):
        d = _tmp()
        orch, _ = _make_orchestrator(data_dir=d)
        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments", return_value={}):
            orch._persist_cycle({"video_id": "vid_001", "strategy_mode": "safe"})
            orch._persist_cycle({"video_id": "vid_002", "strategy_mode": "aggressive"})

        assert orch._load_cycle("vid_001")["strategy_mode"] == "safe"
        assert orch._load_cycle("vid_002")["strategy_mode"] == "aggressive"
        assert orch._load_cycle("vid_999") is None

    def test_cycle_log_written_to_data_dir(self):
        d = _tmp()
        orch, _ = _make_orchestrator(data_dir=d, publisher_returns="vid_log")
        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments", return_value={}):
            orch.run_cycle(_editorial())
        log_path = d / "cycle_log.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["video_id"] == "vid_log"


# ── learn_from_feedback ────────────────────────────────────────────────────────

class TestLearnFromFeedback:
    def _setup(self, policy_id="policy_A", video_id="vid_x"):
        d = _tmp()
        orch, mocks = _make_orchestrator(
            data_dir=d, scene_policy_id=policy_id, publisher_returns=video_id
        )
        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments", return_value={}):
            orch.run_cycle(_editorial())
        return orch, mocks, video_id

    def test_bandit_updated_with_retention_metrics(self):
        orch, mocks, vid = self._setup(policy_id="policy_A")
        performance = {
            "ctr": 0.06, "retention_30s": 0.55, "avg_watch_pct": 0.42,
            "retention_curve": [], "scene_map": {},
        }
        orch.learn_from_feedback(vid, performance)
        mocks["scene_bandit"].update.assert_called_once_with(
            "policy_A",
            retention_30s=0.55,
            avg_watch_pct=0.42,
        )

    def test_penalize_called_with_ctr_and_retention(self):
        orch, mocks, vid = self._setup(policy_id="policy_B")
        performance = {
            "ctr": 0.09, "retention_30s": 0.20, "avg_watch_pct": 0.30,
            "retention_curve": [], "scene_map": {},
        }
        orch.learn_from_feedback(vid, performance)
        mocks["scene_bandit"].penalize_for_low_retention.assert_called_once_with(
            "policy_B",
            packaging_ctr=0.09,
            retention=0.20,
        )

    def test_bandit_not_updated_when_policy_none(self):
        d = _tmp()
        orch, mocks = _make_orchestrator(data_dir=d, publisher_returns="vid_none")
        mocks["scene_bandit"].select_policy.return_value = None
        with patch("src.drop_predictor.apply_preemptive_corrections", return_value=[]), \
             patch("src.drop_predictor.suggest_strategy_adjustments", return_value={}):
            orch.run_cycle(_editorial())
        performance = {
            "ctr": 0.06, "retention_30s": 0.50, "avg_watch_pct": 0.40,
            "retention_curve": [], "scene_map": {},
        }
        orch.learn_from_feedback("vid_none", performance)
        mocks["scene_bandit"].update.assert_not_called()
        mocks["scene_bandit"].penalize_for_low_retention.assert_not_called()

    def test_retention_engine_analyze_called_with_curve(self):
        orch, mocks, vid = self._setup()
        curve     = [1.0, 0.9, 0.7, 0.5]
        scene_map = {1: {"scene_idx": 0}}
        performance = {
            "ctr": 0.05, "retention_30s": 0.50, "avg_watch_pct": 0.40,
            "retention_curve": curve, "scene_map": scene_map,
        }
        orch.learn_from_feedback(vid, performance)
        mocks["retention_engine"].analyze.assert_called_once()
        call_args = mocks["retention_engine"].analyze.call_args
        passed_data = call_args[0][0]
        assert passed_data["curve"] == curve

    def test_retention_engine_skipped_without_curve(self):
        orch, mocks, vid = self._setup()
        performance = {
            "ctr": 0.05, "retention_30s": 0.50, "avg_watch_pct": 0.40,
            "retention_curve": [], "scene_map": {},
        }
        orch.learn_from_feedback(vid, performance)
        mocks["retention_engine"].analyze.assert_not_called()

    def test_performance_store_receives_correct_record(self):
        orch, mocks, vid = self._setup(policy_id="policy_C", video_id="vid_perf")
        performance = {
            "ctr": 0.07, "retention_30s": 0.60, "avg_watch_pct": 0.45,
            "retention_curve": [], "scene_map": {},
        }
        orch.learn_from_feedback("vid_perf", performance)
        mocks["performance_store"].save.assert_called_once()
        saved = mocks["performance_store"].save.call_args[0][0]
        assert saved["video_id"]        == "vid_perf"
        assert saved["ctr"]             == 0.07
        assert saved["retention_30s"]   == 0.60
        assert saved["avg_watch_pct"]   == 0.45
        assert saved["scene_policy_id"] == "policy_C"

    def test_missing_cycle_is_no_op(self):
        d = _tmp()
        orch, mocks = _make_orchestrator(data_dir=d)
        performance = {"ctr": 0.05, "retention_30s": 0.5, "avg_watch_pct": 0.4,
                       "retention_curve": [], "scene_map": {}}
        orch.learn_from_feedback("does_not_exist", performance)
        mocks["scene_bandit"].update.assert_not_called()
        mocks["performance_store"].save.assert_not_called()

    def test_default_performance_values_when_missing(self):
        orch, mocks, vid = self._setup()
        orch.learn_from_feedback(vid, {})
        mocks["scene_bandit"].update.assert_called_once_with(
            "policy_A",
            retention_30s=0.0,
            avg_watch_pct=0.0,
        )


# ── create_orchestrator factory ────────────────────────────────────────────────

class TestCreateOrchestrator:
    def test_returns_system_orchestrator(self):
        d = _tmp()
        orch = create_orchestrator(data_dir=d)
        assert isinstance(orch, SystemOrchestrator)

    def test_null_renderer_returns_none(self):
        d = _tmp()
        orch = create_orchestrator(data_dir=d)
        result = orch.video_renderer(scenes=[], packaging=None, strategy=None,
                                     editorial_input={})
        assert result is None

    def test_null_publisher_returns_none(self):
        d = _tmp()
        orch = create_orchestrator(data_dir=d)
        result = orch.video_publisher(video_path=None, packaging=None)
        assert result is None

    def test_performance_store_save_creates_file(self):
        d = _tmp()
        orch = create_orchestrator(data_dir=d)
        orch.performance_store.save({"video_id": "v1", "ctr": 0.05})
        perf_path = d / "orchestrator_performance.json"
        assert perf_path.exists()
        records = json.loads(perf_path.read_text())
        assert records[0]["video_id"] == "v1"

    def test_performance_store_appends_on_multiple_saves(self):
        d = _tmp()
        orch = create_orchestrator(data_dir=d)
        orch.performance_store.save({"video_id": "v1"})
        orch.performance_store.save({"video_id": "v2"})
        records = json.loads((d / "orchestrator_performance.json").read_text())
        assert len(records) == 2
        assert {r["video_id"] for r in records} == {"v1", "v2"}
