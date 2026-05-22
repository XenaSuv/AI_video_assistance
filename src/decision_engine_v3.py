"""Decision Engine v3 — unified signals → strategy pipeline.

Unified signals → strategy pipeline.  A single decide(context) call
integrates every signal layer:

    performance metrics  → mode selection
    bandit recommendations → scene policy + packaging style
    drop-predictor risks → pre-emptive actions
    decision history     → stability / mode-lock

Execution pipeline (in SystemOrchestrator)
------------------------------------------
    strategy  = engine_v3.decide(context)
    packaging = packaging_engine.generate(editorial, strategy)
    scenes    = scene_engine.assign(scenes, strategy)

    if "shorten_scenes"   in strategy.actions: shorten_explain_scenes(scenes)
    if "increase_avatar"  in strategy.actions: boost_avatar_frequency(scenes)
    if "add_hooks"        in strategy.actions: inject_micro_hooks(scenes)
    if "preemptive_fix"   in strategy.actions: apply_preemptive_corrections(scenes, preds)

Context shape (input)
---------------------
{
    "metrics": {
        "avg_watch_pct":  float,   # 0–1
        "ctr":            float,   # 0–1
        "drops":          list[int],       # scene indices with retention drops
        "retention_30s":  float,           # optional
    },
    "bandit": {
        "scene":     dict,   # SceneBandit.suggest_strategy_adjustments()
        "packaging": dict,   # ThompsonBandit.suggest_strategy_adjustments()
    },
    "prediction": {
        "risks": list[dict],   # [{scene_idx, probability}, ...]
    },
    "history": list[dict],     # past UnifiedStrategy.to_dict() entries (newest last)
}

UnifiedStrategy fields
----------------------
    mode                "growth" | "packaging_focus" | "retention_fix" | "stable"
    pace                "fast" | "normal" | "slow"
    hook_aggressiveness float 0–1
    scene_policy        dict[str, float]  — scene-type weights summing ≈ 1.0
    packaging_style     "conflict" | "curiosity" | "simple"
    persona             "energetic" | "curious" | "clear_explainer" | "balanced"
    actions             list[str]
    mode_locked         bool
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

# ── Rolling-window constants ──────────────────────────────────────────────────

WINDOW_SHORT: int = 7   # days — short-term trend detection
WINDOW_LONG:  int = 30  # days — strategic pattern calibration

# ── PerformanceStore ──────────────────────────────────────────────────────────

_DEFAULT_METRICS: dict[str, Any] = {
    "avg_watch_pct":  0.50,
    "hook_retention": 0.50,
    "drop_points":    [],
    "ctr":            0.04,
    "recent_trend":   "stable",
}


class PerformanceStore:
    """Append-only JSON log of channel metrics + strategy used per video.

    Used by DecisionEngineV3 to build channel-level context for decide().
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or settings.data_dir
        self._path = self._data_dir / "performance_store.json"

    def save(
        self,
        video_id: str,
        metrics: dict[str, Any],
        strategy_mode: str,
    ) -> None:
        entries = self._load_raw()
        entries.append({
            "video_id":      video_id,
            "timestamp":     datetime.now(timezone.utc).isoformat(),
            "metrics":       metrics,
            "strategy_mode": strategy_mode,
        })
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(entries, indent=2))
        logger.debug(f"PerformanceStore: saved entry for {video_id!r}")

    def load_recent(self, n: int = 10) -> list[dict[str, Any]]:
        return self._load_raw()[-n:]

    def load_within(self, days: int) -> list[dict[str, Any]]:
        """Return entries whose timestamp falls within the last *days* days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result: list[dict[str, Any]] = []
        for entry in self._load_raw():
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts >= cutoff:
                    result.append(entry)
            except Exception:
                pass
        return result

    def get_channel_metrics(self, window_days: int = WINDOW_LONG) -> dict[str, Any]:
        recent = self.load_within(window_days)
        if len(recent) < 3:
            # Fall back to last 10 entries when time window has too few data points
            recent = self.load_recent(10)
        if not recent:
            return dict(_DEFAULT_METRICS)
        watch_pcts = [e["metrics"].get("avg_watch_pct",  0.5)  for e in recent]
        hook_rets  = [e["metrics"].get("hook_retention",  0.5)  for e in recent]
        ctrs       = [e["metrics"].get("ctr",             0.04) for e in recent]
        trend      = self._compute_trend(watch_pcts)
        all_drops: list[int] = []
        for e in recent:
            all_drops.extend(e["metrics"].get("drop_points", []))
        common_drops = [p for p, _ in Counter(all_drops).most_common(3)]
        return {
            "avg_watch_pct":  round(sum(watch_pcts) / len(watch_pcts), 4),
            "hook_retention": round(sum(hook_rets)  / len(hook_rets),  4),
            "drop_points":    common_drops,
            "ctr":            round(sum(ctrs)        / len(ctrs),       4),
            "recent_trend":   trend,
        }

    def _compute_trend(self, values: list[float]) -> str:
        if len(values) < 2:
            return "stable"
        mid        = len(values) // 2
        recent_avg = sum(values[mid:]) / len(values[mid:])
        older_avg  = sum(values[:mid]) / len(values[:mid])
        if recent_avg < older_avg - 0.05:
            return "down"
        if recent_avg > older_avg + 0.05:
            return "up"
        return "stable"

    def _load_raw(self) -> list[dict[str, Any]]:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                if isinstance(data, list):
                    return data
        except Exception as exc:
            logger.warning(f"PerformanceStore: could not read store ({exc})")
        return []


# ── Constants ─────────────────────────────────────────────────────────────────

_VALID_MODES     = {"growth", "packaging_focus", "retention_fix", "stable"}
_VALID_PACES     = {"fast", "normal", "slow"}
_VALID_PACKAGING = {"conflict", "curiosity", "simple"}
_VALID_PERSONAS  = {"energetic", "curious", "clear_explainer", "balanced"}

HIGH_RISK_THRESHOLD = 0.6   # matches drop_predictor.HIGH_RISK_THRESHOLD

DEFAULT_SCENE_MIX: dict[str, float] = {
    "avatar":       0.30,
    "text_overlay": 0.30,
    "image":        0.20,
    "diagram":      0.20,
}
SAFE_SCENE_MIX: dict[str, float] = {
    "avatar":       0.50,
    "image":        0.30,
    "text_overlay": 0.10,
    "diagram":      0.10,
}


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class UnifiedStrategy:
    """Fully resolved strategy produced by DecisionEngineV3.decide()."""

    mode:                str              = "stable"
    pace:                str              = "normal"
    hook_aggressiveness: float            = 0.5
    scene_policy:        dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SCENE_MIX))
    packaging_style:     str              = "curiosity"
    persona:             str              = "balanced"
    actions:             list[str]        = field(default_factory=list)
    mode_locked:         bool             = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode":                self.mode,
            "pace":                self.pace,
            "hook_aggressiveness": self.hook_aggressiveness,
            "scene_policy":        dict(self.scene_policy),
            "packaging_style":     self.packaging_style,
            "persona":             self.persona,
            "actions":             list(self.actions),
            "mode_locked":         self.mode_locked,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "UnifiedStrategy":
        return cls(
            mode                = d.get("mode",                "stable"),
            pace                = d.get("pace",                "normal"),
            hook_aggressiveness = float(d.get("hook_aggressiveness", 0.5)),
            scene_policy        = d.get("scene_policy",        dict(DEFAULT_SCENE_MIX)),
            packaging_style     = d.get("packaging_style",     "curiosity"),
            persona             = d.get("persona",             "balanced"),
            actions             = list(d.get("actions",        [])),
            mode_locked         = bool(d.get("mode_locked",    False)),
        )


# ── History persistence ───────────────────────────────────────────────────────

class DecisionHistoryStore:
    """JSONL-backed decision history for stability analysis."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._path = (data_dir or settings.data_dir) / "decision_history_v3.jsonl"

    def append(self, strategy: UnifiedStrategy) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            record = strategy.to_dict()
            record["_ts"] = datetime.now(timezone.utc).isoformat()
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"DecisionHistoryStore: could not append ({exc})")

    def load_recent(self, n: int = 10) -> list[dict[str, Any]]:
        try:
            if not self._path.exists():
                return []
            lines = self._path.read_text().strip().splitlines()
            records = [json.loads(ln) for ln in lines if ln.strip()]
            return records[-n:]
        except Exception as exc:
            logger.warning(f"DecisionHistoryStore: could not load history ({exc})")
            return []


# ── DecisionEngineV3 ──────────────────────────────────────────────────────────

class DecisionEngineV3:
    """Unified decision engine: signals → UnifiedStrategy.

    Typical usage::

        engine = DecisionEngineV3(data_dir=run_dir)

        context = {
            "metrics":    {"avg_watch_pct": 0.35, "ctr": 0.04, "drops": [2, 4]},
            "bandit":     {"scene": bandit.suggest_strategy_adjustments(),
                           "packaging": thompson.suggest_strategy_adjustments()},
            "prediction": {"risks": [{"scene_idx": 1, "probability": 0.75}]},
            "history":    engine.load_history(),   # or []
        }

        strategy = engine.decide(context)
        # strategy.mode, strategy.actions, strategy.scene_policy, ...
    """

    STABILITY_WINDOW = 3   # consecutive same-mode cycles before mode_locked

    def __init__(self, data_dir: Path | None = None) -> None:
        self._store = DecisionHistoryStore(data_dir)

    # ── Public API ─────────────────────────────────────────────────────────────

    def decide(self, context: dict[str, Any]) -> UnifiedStrategy:
        """Main entry point.  context → UnifiedStrategy."""
        signals  = self._collect_signals(context)
        mode     = self._select_mode(signals)
        strategy = self._build_strategy(mode, signals)
        strategy = self._apply_stability(strategy, context)
        strategy = self._apply_corrections(strategy, signals)

        self._store.append(strategy)
        logger.info(
            f"DecisionEngineV3: mode={strategy.mode} pace={strategy.pace} "
            f"packaging={strategy.packaging_style} persona={strategy.persona} "
            f"actions={strategy.actions} locked={strategy.mode_locked}"
        )
        return strategy

    def load_history(self, n: int = 10) -> list[dict[str, Any]]:
        """Return the n most-recent decision records (for injecting into context)."""
        return self._store.load_recent(n)

    # ── Signal collection ──────────────────────────────────────────────────────

    def _collect_signals(self, context: dict[str, Any]) -> dict[str, Any]:
        """Flatten all context layers into one uniform signal dict."""
        metrics            = context.get("metrics")            or {}
        strategic_metrics  = context.get("strategic_metrics")  or {}
        bandit             = context.get("bandit")             or {}
        pred               = context.get("prediction")         or {}

        high_risk = [
            r for r in (pred.get("risks") or [])
            if r.get("probability", 0.0) >= HIGH_RISK_THRESHOLD
        ]

        short_retention = float(metrics.get("avg_watch_pct", 0.0))
        long_retention  = float(strategic_metrics.get("avg_watch_pct", short_retention))

        return {
            "retention":           short_retention,
            "strategic_retention": long_retention,
            "ctr":                 float(metrics.get("ctr",           0.0)),
            "drops":               list(metrics.get("drops",          [])),
            "retention_30s":       float(metrics.get("retention_30s", 0.0)),
            "scene_bandit":        bandit.get("scene")     or {},
            "packaging_bandit":    bandit.get("packaging") or {},
            "predicted_risks":     high_risk,
        }

    # ── Mode selection ─────────────────────────────────────────────────────────

    def _select_mode(self, s: dict[str, Any]) -> str:
        """Rule priority: retention → CTR → drops → stable.

        Hysteresis: a poor 7-day retention only triggers "growth" when the
        30-day baseline also confirms weakness (r30 < 0.45).  A single bad
        week shouldn't flip the entire strategy.
        """
        r7  = s["retention"]
        r30 = s.get("strategic_retention") or r7
        ctr = s["ctr"]

        if r7 < 0.4 and r30 < 0.45:
            return "growth"
        if r7 < 0.4:
            return "packaging_focus"
        if ctr < 0.05:
            return "packaging_focus"
        if len(s["drops"]) > 2:
            return "retention_fix"
        return "stable"

    # ── Strategy construction ──────────────────────────────────────────────────

    def _build_strategy(self, mode: str, s: dict[str, Any]) -> UnifiedStrategy:
        packaging_style = self._select_packaging(s)

        if mode == "growth":
            return UnifiedStrategy(
                mode                = "growth",
                pace                = "fast",
                hook_aggressiveness = 0.9,
                scene_policy        = self._select_scene_policy(s),
                packaging_style     = packaging_style,
                persona             = "energetic",
            )

        if mode == "packaging_focus":
            return UnifiedStrategy(
                mode                = "packaging_focus",
                pace                = "normal",
                hook_aggressiveness = 1.0,
                scene_policy        = self._select_scene_policy(s),
                packaging_style     = "conflict",   # force strongest packaging
                persona             = "curious",
            )

        if mode == "retention_fix":
            return UnifiedStrategy(
                mode                = "retention_fix",
                pace                = "fast",
                hook_aggressiveness = 0.8,
                scene_policy        = self._safe_scene_policy(),
                packaging_style     = packaging_style,
                persona             = "clear_explainer",
            )

        return self._stable_strategy(s)

    def _stable_strategy(self, s: dict[str, Any] | None = None) -> UnifiedStrategy:
        return UnifiedStrategy(
            mode                = "stable",
            pace                = "normal",
            hook_aggressiveness = 0.5,
            scene_policy        = self._blend_with_default((s or {}).get("scene_bandit") or {}),
            packaging_style     = self._select_packaging(s or {}),
            persona             = "balanced",
        )

    # ── Scene policy helpers ───────────────────────────────────────────────────

    def _select_scene_policy(self, signals: dict[str, Any]) -> dict[str, float]:
        """Bandit is an advisor, not a boss.

        When retention is very low: trust the bandit fully (need change).
        Otherwise: blend bandit suggestion with the safe default.
        """
        suggestion = signals.get("scene_bandit") or {}
        if signals.get("retention", 1.0) < 0.4:
            bandit_mix = suggestion.get("scene_mix") or {}
            return bandit_mix if bandit_mix else dict(DEFAULT_SCENE_MIX)
        return self._blend_with_default(suggestion)

    def _blend_with_default(self, suggestion: dict[str, Any]) -> dict[str, float]:
        """50 / 50 average of bandit recommendation and DEFAULT_SCENE_MIX."""
        bandit_mix = suggestion.get("scene_mix") or {}
        if not bandit_mix:
            return dict(DEFAULT_SCENE_MIX)
        all_keys = set(DEFAULT_SCENE_MIX) | set(bandit_mix)
        blended  = {
            k: (DEFAULT_SCENE_MIX.get(k, 0.0) + bandit_mix.get(k, 0.0)) / 2
            for k in all_keys
        }
        total = sum(blended.values())
        if total > 0:
            blended = {k: round(v / total, 4) for k, v in blended.items()}
        return blended

    def _safe_scene_policy(self) -> dict[str, float]:
        return dict(SAFE_SCENE_MIX)

    # ── Packaging style ────────────────────────────────────────────────────────

    def _select_packaging(self, signals: dict[str, Any]) -> str:
        """Force 'conflict' when CTR is weak; otherwise follow packaging bandit."""
        if signals.get("ctr", 1.0) < 0.05:
            return "conflict"
        bandit = signals.get("packaging_bandit") or {}
        preferred = bandit.get("preferred_variant_type", "")
        return preferred if preferred in _VALID_PACKAGING else "curiosity"

    # ── Stability layer ────────────────────────────────────────────────────────

    def _apply_stability(
        self,
        strategy: UnifiedStrategy,
        context:  dict[str, Any],
    ) -> UnifiedStrategy:
        """Lock mode when the last N decisions shared the same mode.

        mode_locked=True signals to the orchestrator that this mode has been
        stable long enough to start committing to deeper structural changes.
        """
        history = context.get("history") or []
        if len(history) >= self.STABILITY_WINDOW:
            last = history[-self.STABILITY_WINDOW :]
            modes = [h.get("mode", "") for h in last]
            if len(set(modes)) == 1 and modes[0] == strategy.mode:
                strategy.mode_locked = True
                logger.debug(
                    f"DecisionEngineV3: mode={strategy.mode!r} locked "
                    f"({self.STABILITY_WINDOW} consecutive cycles)"
                )
        return strategy

    # ── Correction actions ─────────────────────────────────────────────────────

    def _apply_corrections(
        self,
        strategy: UnifiedStrategy,
        signals:  dict[str, Any],
    ) -> UnifiedStrategy:
        """Append concrete pipeline action commands to the strategy."""
        actions: set[str] = set(strategy.actions)

        if signals["drops"]:
            actions |= {"shorten_scenes", "add_hooks", "increase_avatar"}

        if signals["predicted_risks"]:
            actions.add("preemptive_fix")

        strategy.actions = sorted(actions)   # deterministic order
        return strategy
