"""System Orchestrator — single control point for the full video pipeline.

Wires all v2 components into one repeatable cycle:
    analyze → decide → package → assign_scenes → predict → render → publish → learn

Usage
-----
    orchestrator = create_orchestrator(data_dir=run_dir)
    result = orchestrator.run_cycle(editorial_input)

    # After YouTube Analytics arrive (≥24h later):
    orchestrator.learn_from_feedback(result.video_id, performance_dict)

editorial_input dict shape
--------------------------
    {
        "scenes":         list[Scene],
        "editorial_plan": EditorialPlan | dict,   # from EditorialBrain
        "platform":       "youtube",
        "video_id":       str | None,             # None before publish
    }

performance dict shape (learn_from_feedback)
--------------------------------------------
    {
        "ctr":            float,   # click-through rate
        "retention_30s":  float,   # 30-second retention
        "avg_watch_pct":  float,   # average watch percentage
        "retention_curve": list[float],
        "scene_map":      dict[int, dict],   # timestamp_idx → scene info
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ── Cycle result ──────────────────────────────────────────────────────────────

@dataclass
class CycleResult:
    """Everything the orchestrator produced in one run."""
    video_id:            str | None
    strategy_mode:       str
    scene_policy_id:     str
    packaging_hook:      str
    high_risk_scenes:    int
    corrections_applied: int
    success:             bool = True
    error:               str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id":            self.video_id,
            "strategy_mode":       self.strategy_mode,
            "scene_policy_id":     self.scene_policy_id,
            "packaging_hook":      self.packaging_hook,
            "high_risk_scenes":    self.high_risk_scenes,
            "corrections_applied": self.corrections_applied,
            "success":             self.success,
            "error":               self.error,
        }


# ── Orchestrator ──────────────────────────────────────────────────────────────

class SystemOrchestrator:
    """Coordinates all v2 pipeline components in a single repeatable cycle.

    Each component is injected so the orchestrator is fully testable without
    real API keys, files, or network access.

    Component contracts
    -------------------
    decision_engine  : .decide(metrics, scene_bandit=...) → StrategyConfig
    packaging_engine : .generate(editorial_input, strategy) → PackagingResult
    scene_bandit     : .select_policy() → ScenePolicyArm | None
                       .update(policy_id, retention_30s, avg_watch_pct)
                       .penalize_for_low_retention(policy_id, ctr, retention)
    scene_engine     : .assign_from_config(scenes, strategy, editorial_plan=...)
    drop_predictor   : .predict_scenes(scenes) → list[DropPrediction]
    retention_engine : .analyze(retention_data, scene_map) → list[Correction]
                       .apply_corrections(scenes, corrections)
                       .suggest_strategy_adjustments(corrections, scene_map, curve)
                       .record_outcome(fix, scene_type, intent, delta)
    video_renderer   : callable(scenes, packaging, strategy, editorial_input) → Path | None
    video_publisher  : callable(video_path, packaging) → str | None
    performance_store: .save(record: dict)
    """

    def __init__(
        self,
        decision_engine,
        packaging_engine,
        scene_bandit,
        scene_engine,
        drop_predictor,
        retention_engine,
        video_renderer,
        video_publisher,
        performance_store,
        data_dir: Path | None = None,
    ) -> None:
        self.decision_engine   = decision_engine
        self.packaging_engine  = packaging_engine
        self.scene_bandit      = scene_bandit
        self.scene_engine      = scene_engine
        self.drop_predictor    = drop_predictor
        self.retention_engine  = retention_engine
        self.video_renderer    = video_renderer
        self.video_publisher   = video_publisher
        self.performance_store = performance_store
        self._data_dir         = data_dir or settings.data_dir
        self._cycle_log_path   = self._data_dir / "cycle_log.jsonl"

    # ── Main cycle ─────────────────────────────────────────────────────────────

    def run_cycle(self, editorial_input: dict[str, Any]) -> CycleResult:
        """Run one full video production cycle.

        Catches all exceptions: on failure returns a CycleResult with
        success=False so the scheduler can retry or fall back.
        """
        try:
            return self._run_cycle_unsafe(editorial_input)
        except Exception as exc:
            logger.error(f"SystemOrchestrator: cycle failed — {exc}")
            return CycleResult(
                video_id            = None,
                strategy_mode       = "unknown",
                scene_policy_id     = "unknown",
                packaging_hook      = "",
                high_risk_scenes    = 0,
                corrections_applied = 0,
                success             = False,
                error               = str(exc),
            )

    def _run_cycle_unsafe(self, editorial_input: dict[str, Any]) -> CycleResult:
        scenes         = editorial_input.get("scenes", [])
        editorial_plan = editorial_input.get("editorial_plan")

        # ── 1. Strategy decision (scene_bandit wired in) ───────────────────
        metrics  = editorial_input.get("metrics", {})
        strategy = self.decision_engine.decide(metrics, scene_bandit=self.scene_bandit)
        logger.info(f"SystemOrchestrator: strategy mode={strategy.mode}")

        # ── 2. Scene policy arm used this cycle ────────────────────────────
        scene_policy     = self.scene_bandit.select_policy()
        scene_policy_id  = scene_policy.id if scene_policy else "none"

        # ── 3. Packaging (hook / title / thumbnail) ────────────────────────
        packaging = self.packaging_engine.generate(editorial_input, strategy)
        logger.info(f"SystemOrchestrator: hook={packaging.hook!r}")

        # ── 4. Scene type assignment ───────────────────────────────────────
        self.scene_engine.assign_from_config(
            scenes, strategy, editorial_plan=editorial_plan
        )

        # ── 5. Drop risk prediction + pre-emptive corrections ──────────────
        from src.drop_predictor import (
            apply_preemptive_corrections,
            suggest_strategy_adjustments as drop_adjustments,
        )
        predictions  = self.drop_predictor.predict_scenes(scenes)
        high_risk    = sum(1 for p in predictions if p.probability >= 0.6)
        applied      = apply_preemptive_corrections(scenes, predictions)
        drop_hints   = drop_adjustments(predictions)

        if drop_hints.get("pace") == "fast":
            strategy.pace = "fast"
        if drop_hints.get("force_avatar_intro") and scenes:
            scenes[0].scene_type = "avatar"

        # ── 6. Render ──────────────────────────────────────────────────────
        video_path = self.video_renderer(
            scenes         = scenes,
            packaging      = packaging,
            strategy       = strategy,
            editorial_input = editorial_input,
        )

        # ── 7. Publish ─────────────────────────────────────────────────────
        video_id = self.video_publisher(
            video_path = video_path,
            packaging  = packaging,
        ) if video_path is not None else None

        # ── 8. Persist cycle record for deferred learning ──────────────────
        cycle_record = {
            "video_id":       video_id,
            "strategy_mode":  strategy.mode,
            "scene_policy_id": scene_policy_id,
            "hook":           packaging.hook,
            "high_risk":      high_risk,
            "corrections":    len(applied),
        }
        self._persist_cycle(cycle_record)

        result = CycleResult(
            video_id            = video_id,
            strategy_mode       = strategy.mode,
            scene_policy_id     = scene_policy_id,
            packaging_hook      = packaging.hook,
            high_risk_scenes    = high_risk,
            corrections_applied = len(applied),
        )
        self._log_cycle(result)
        return result

    # ── Deferred learning (call after analytics arrive) ───────────────────

    def learn_from_feedback(
        self,
        video_id:    str,
        performance: dict[str, Any],
    ) -> None:
        """Update all bandits and predictors once YouTube Analytics are available.

        Call this ≥24h after publish when analytics are stable.
        """
        ctr           = performance.get("ctr",           0.0)
        retention_30s = performance.get("retention_30s", 0.0)
        avg_watch_pct = performance.get("avg_watch_pct", 0.0)
        curve         = performance.get("retention_curve", [])
        scene_map     = performance.get("scene_map",     {})

        # Retrieve what was used in this cycle
        cycle = self._load_cycle(video_id)
        if cycle is None:
            logger.warning(f"SystemOrchestrator: no cycle record for {video_id!r}")
            return

        scene_policy_id = cycle.get("scene_policy_id", "")

        # ── Update Scene Bandit ────────────────────────────────────────────
        if scene_policy_id and scene_policy_id != "none":
            self.scene_bandit.update(
                scene_policy_id,
                retention_30s = retention_30s,
                avg_watch_pct = avg_watch_pct,
            )
            self.scene_bandit.penalize_for_low_retention(
                scene_policy_id,
                packaging_ctr = ctr,
                retention     = retention_30s,
            )

        # ── Retention curve → scene-level corrections for next video ───────
        if curve:
            retention_data = {"curve": curve, "scene_map": scene_map}
            corrections    = self.retention_engine.analyze(retention_data, scene_map)
            adjustments    = self.retention_engine.suggest_strategy_adjustments(
                corrections, scene_map, curve
            )
            logger.info(
                f"SystemOrchestrator: retention analysis → "
                f"{len(corrections)} correction(s), adjustments={adjustments}"
            )

        # ── Update predictor training dataset ─────────────────────────────
        scenes_snapshot = cycle.get("scenes_snapshot", [])
        if scenes_snapshot and curve:
            self.drop_predictor.dataset.build_from_video(
                scenes_snapshot,
                {"curve": curve},
                scene_map,
            )

        # ── Persist performance record ─────────────────────────────────────
        self.performance_store.save({
            "video_id":       video_id,
            "ctr":            ctr,
            "retention_30s":  retention_30s,
            "avg_watch_pct":  avg_watch_pct,
            "strategy_mode":  cycle.get("strategy_mode"),
            "scene_policy_id": scene_policy_id,
        })

        logger.info(
            f"SystemOrchestrator: learned from {video_id!r} — "
            f"ctr={ctr:.3f} ret={retention_30s:.2f} watch={avg_watch_pct:.2f}"
        )

    # ── Logging ────────────────────────────────────────────────────────────────

    def _log_cycle(self, result: CycleResult) -> None:
        logger.info(
            f"SystemOrchestrator cycle complete | "
            f"video_id={result.video_id!r} "
            f"mode={result.strategy_mode} "
            f"policy={result.scene_policy_id} "
            f"hook={result.packaging_hook!r} "
            f"high_risk={result.high_risk_scenes} "
            f"corrections={result.corrections_applied} "
            f"success={result.success}"
        )

    # ── Cycle persistence ──────────────────────────────────────────────────────

    def _persist_cycle(self, record: dict[str, Any]) -> None:
        try:
            self._cycle_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._cycle_log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"SystemOrchestrator: could not persist cycle ({exc})")

    def _load_cycle(self, video_id: str) -> dict[str, Any] | None:
        try:
            if not self._cycle_log_path.exists():
                return None
            with open(self._cycle_log_path) as f:
                for line in f:
                    record = json.loads(line.strip())
                    if record.get("video_id") == video_id:
                        return record
        except Exception as exc:
            logger.warning(f"SystemOrchestrator: could not load cycle ({exc})")
        return None


# ── Factory ───────────────────────────────────────────────────────────────────

def create_orchestrator(data_dir: Path | None = None) -> SystemOrchestrator:
    """Wire up all v2 components with production defaults.

    All components read/write under *data_dir* so every video run is isolated.
    """
    from src.decision_engine_v2     import DecisionEngineV2
    from src.packaging_engine       import PackagingEngine
    from src.scene_bandit           import SceneBandit
    from src.scene_variety_engine   import SceneVarietyEngineV2
    from src.scene_strategy_engine  import SceneStrategyEngine
    from src.drop_predictor         import DropPredictor
    from src.retention_correction_engine import RetentionCorrectionEngine

    d = data_dir or settings.data_dir

    strat_engine  = SceneStrategyEngine()
    scene_engine  = SceneVarietyEngineV2(strat_engine)

    def _null_renderer(scenes, packaging, strategy, editorial_input):
        """Override in production with build_video()."""
        logger.warning("SystemOrchestrator: no video_renderer configured")
        return None

    def _null_publisher(video_path, packaging):
        logger.warning("SystemOrchestrator: no video_publisher configured")
        return None

    class _SimpleStore:
        def __init__(self, path: Path) -> None:
            self._path = path

        def save(self, record: dict) -> None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            records = []
            if self._path.exists():
                try:
                    records = json.loads(self._path.read_text())
                except Exception:
                    pass
            records.append(record)
            self._path.write_text(json.dumps(records, indent=2))

    return SystemOrchestrator(
        decision_engine   = DecisionEngineV2(data_dir=d),
        packaging_engine  = PackagingEngine(),
        scene_bandit      = SceneBandit(data_dir=d),
        scene_engine      = scene_engine,
        drop_predictor    = DropPredictor(data_dir=d),
        retention_engine  = RetentionCorrectionEngine(data_dir=d),
        video_renderer    = _null_renderer,
        video_publisher   = _null_publisher,
        performance_store = _SimpleStore(d / "orchestrator_performance.json"),
        data_dir          = d,
    )
