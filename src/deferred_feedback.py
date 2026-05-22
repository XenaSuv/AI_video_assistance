"""Deferred feedback collection — updates Thompson, Performance, Sequence, and Cross-learning.

Called once at the start of each daily pipeline run, before script generation.
All work is best-effort: individual failures are logged as warnings and never propagate.
Returns the most common preferred hook type across recent videos, or None.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings
from src.cross_learning_engine import CrossLearningEngine
from src.decision_engine_v3 import PerformanceStore
from src.pipeline_helpers import _build_scene_map, _save_retention_state
from src.retention_correction_engine import RetentionCorrectionEngine
from src.sequence_learning_engine import SequenceLearningEngine
from src.shorts_experiment_engine import ShortsExperimentEngine
from src.thompson_bandit import ThompsonBandit
from src.youtube_analytics import get_retention_curve, get_video_metrics

LONG_WINDOW_DAYS: int = 30   # calendar days to look back for feedback


def _parse_dir_date(name: str) -> datetime | None:
    """Parse YYYY-MM-DD from a run-directory name, return None on failure."""
    try:
        return datetime.strptime(name[:10], "%Y-%m-%d")
    except Exception:
        return None


class _DeferredFeedbackCollector:
    """Scan the last 30 days of run dirs, fetch YouTube metrics, and update all
    learning stores.  Returns the most common preferred hook type, or None.
    """

    def __init__(self, current_run_dir: Path) -> None:
        self._current_run_dir = current_run_dir

    def collect(self) -> str | None:
        """Run deferred feedback collection.  Returns preferred hook type or None."""
        preferred_types: list[str] = []

        cutoff = datetime.now() - timedelta(days=LONG_WINDOW_DAYS)
        run_dirs = sorted(
            [
                p.parent
                for p in settings.output_dir.glob("*/thompson_state.json")
                if (_parse_dir_date(p.parent.name) or datetime.min) >= cutoff
            ],
            key=lambda p: p.name,
            reverse=True,
        )

        for run_dir in run_dirs:
            if run_dir == self._current_run_dir:
                continue
            try:
                preferred = self._process_run_dir(run_dir)
                if preferred:
                    preferred_types.append(preferred)
            except Exception as exc:
                logger.warning(
                    f"ThompsonBandit: deferred update failed for {run_dir.name}: {exc}"
                )

        self._collect_shorts_analytics()

        if not preferred_types:
            return None
        return max(set(preferred_types), key=preferred_types.count)

    # ── private helpers ───────────────────────────────────────────────────────

    def _process_run_dir(self, run_dir: Path) -> str | None:
        cp_path = run_dir / "checkpoint.json"
        if not cp_path.exists():
            return None
        cp_data = json.loads(cp_path.read_text())
        video_id = (
            cp_data.get("steps", {}).get("upload_en", {}).get("video_id")
        )
        if not video_id:
            return None

        bandit = ThompsonBandit(data_dir=run_dir)
        if not bandit.get_state().arms:
            return None

        yt_metrics = get_video_metrics(video_id)
        if not yt_metrics:
            return None

        impressions = int(yt_metrics.get("views", 0))
        retention = float(yt_metrics.get("avg_view_percentage", 0.0)) / 100.0
        clicks = impressions

        for arm in bandit.get_state().arms:
            new_imps = max(0, impressions - arm.impressions)
            new_clicks = max(0, clicks - arm.clicks)
            if new_imps > 0:
                bandit.update(
                    arm.variant_id,
                    impressions=new_imps,
                    clicks=new_clicks,
                    retention_30s=retention,
                )
                logger.info(
                    f"ThompsonBandit: updated arm {arm.variant_id!r} "
                    f"for {video_id} (+{new_imps} impressions, "
                    f"retention={retention:.2f})"
                )
                break

        self._update_perf_store(cp_data, video_id, retention)
        self._store_sequence(run_dir, retention)
        self._learn_cross(run_dir, cp_data, retention)
        self._analyze_retention_curve(run_dir, video_id)

        adjustments = bandit.suggest_strategy_adjustments()
        return adjustments.get("preferred_variant_type")

    def _update_perf_store(self, cp_data: dict[str, Any], video_id: str, retention: float) -> None:
        prior_mode = (
            cp_data.get("steps", {}).get("script", {}).get("v3_mode", "stable")
        )
        PerformanceStore(data_dir=settings.data_dir).save(
            video_id=video_id,
            metrics={
                "avg_watch_pct": retention,
                "hook_retention": retention,
                "ctr": 0.0,
                "drop_points": [],
            },
            strategy_mode=prior_mode,
        )
        logger.info(
            f"PerformanceStore: recorded metrics for {video_id} "
            f"(mode={prior_mode!r}, retention={retention:.2f})"
        )

    def _store_sequence(self, run_dir: Path, retention: float) -> None:
        script_cache = run_dir / "script.json"
        if not script_cache.exists():
            return
        try:
            script_data = json.loads(script_cache.read_text())
            scene_dicts = [
                {"intent": s.get("scene_intent", s.get("intent", "explain"))}
                for s in script_data.get("scenes", [])
            ]
            SequenceLearningEngine(data_dir=settings.data_dir).store_sequence({
                "scenes": scene_dicts,
                "performance": {
                    "avg_watch_pct": retention,
                    "ctr": 0.0,
                    "drops": [],
                },
            })
        except Exception as exc:
            logger.warning(
                f"SequenceLearningEngine: store_sequence failed "
                f"for {run_dir.name}: {exc}"
            )

    def _learn_cross(self, run_dir: Path, cp_data: dict[str, Any], retention: float) -> None:
        cross_combo = (
            cp_data.get("steps", {}).get("script", {}).get("cross_combo")
        )
        if not cross_combo:
            return
        try:
            CrossLearningEngine(data_dir=settings.data_dir).learn({
                "strategy": {
                    "pace":    cross_combo.get("pace",    "normal"),
                    "persona": cross_combo.get("persona", "balanced"),
                },
                "packaging": {
                    "hook_type": cross_combo.get("hook_type", "curiosity"),
                },
                "editorial": {
                    "topic": cross_combo.get("topic", "AI news"),
                },
                "scenes": [{"type": cross_combo.get("scene_type", "image")}],
                "performance": {
                    "avg_watch_pct": retention,
                    "ctr": 0.0,
                },
            })
            logger.info(
                f"CrossLearningEngine: learned combo "
                f"{cross_combo} → retention={retention:.2f}"
            )
        except Exception as exc:
            logger.warning(
                f"CrossLearningEngine: learn() failed "
                f"for {run_dir.name}: {exc}"
            )

    def _analyze_retention_curve(self, run_dir: Path, video_id: str) -> None:
        script_cache = run_dir / "script.json"
        try:
            _rce_curve = get_retention_curve(video_id)
            if _rce_curve and script_cache.exists():
                _rce_script_data = json.loads(script_cache.read_text())
                _rce_scene_map = _build_scene_map(
                    _rce_script_data.get("scenes", []), len(_rce_curve)
                )
                _rce = RetentionCorrectionEngine(data_dir=settings.data_dir)
                _rce_found = _rce.analyze({"curve": _rce_curve}, _rce_scene_map)
                _rce_adj = _rce.suggest_strategy_adjustments(
                    _rce_found, _rce_scene_map, _rce_curve
                )
                _save_retention_state(settings.data_dir, _rce_found, _rce_adj)
                logger.info(
                    f"RetentionCorrectionEngine: {len(_rce_found)} drop(s) "
                    f"for {video_id}, state persisted for next run"
                )
        except Exception as exc:
            logger.warning(
                f"RetentionCorrectionEngine: analysis failed for "
                f"{run_dir.name}: {exc}"
            )

    def _collect_shorts_analytics(self) -> None:
        try:
            _shorts_engine = ShortsExperimentEngine(data_dir=settings.data_dir)
            _new_results = _shorts_engine.collect_analytics(min_age_hours=4.0)
            if _new_results:
                _analysis = _shorts_engine.analyze()
                logger.info(
                    f"ShortsExperimentEngine: collected {_new_results} result(s), "
                    f"winners={_analysis['winners']}, "
                    f"best_hook={_analysis['best_hook_type']!r}"
                )
        except Exception as exc:
            logger.warning(
                f"ShortsExperimentEngine: deferred feedback failed: {exc}"
            )
