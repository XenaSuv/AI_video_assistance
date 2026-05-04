"""Decision Engine v2 — channel-level content strategy from performance metrics.

v1 (DecisionEngine) operates at story level: which angle/format per news item.
v2 (DecisionEngineV2) operates at channel level: system-wide behaviour mode.

Flow
----
PerformanceStore.get_channel_metrics()
    ↓
DecisionEngineV2.decide()   →   StrategyConfig
    ↓
SceneStrategyEngine.apply_strategy(config)
    → scene_mix biases pick_best() for each scene
editorial_brain (mode → ContentStrategy)
video_generator  (pace, hook_aggressiveness)

StrategyConfig fields
---------------------
mode               "growth" | "safe" | "experimental"
video_length       "short"  | "medium" | "long"
avatar_frequency   0.0–1.0  share of scenes with talking-head
hook_aggressiveness 0.0–1.0  0 = mild hooks, 1 = maximum urgency/shock
scene_mix          dict[scene_type, fraction]  must sum ≈ 1.0
pace               "fast" | "normal" | "slow"

Mode selection logic
--------------------
avg_watch_pct < 0.4          → growth      (aggressive retention tactics)
recent_trend == "down"        → experimental (try something different)
avg_watch_pct > 0.55          → safe         (stabilise what's working)
else                          → growth      (default push for improvement)

Performance store
-----------------
data/performance_store.json  — append-only list of
  { video_id, timestamp, metrics, strategy_mode }
Used to compute channel metrics and detect trends across recent videos.
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ── Contract types ────────────────────────────────────────────────────────────

_VALID_MODES        = frozenset({"growth", "safe", "experimental"})
_VALID_LENGTHS      = frozenset({"short", "medium", "long"})
_VALID_PACES        = frozenset({"fast", "normal", "slow"})
_VALID_SCENE_TYPES  = frozenset({"image", "text_overlay", "infographic", "diagram", "cutaway", "avatar"})


@dataclass
class ScenePolicy:
    """Scene-level execution policy carried inside StrategyConfig.

    This is the contract surface between Decision Engine and Scene Variety Engine:
    Decision Engine writes it; SceneVarietyEngineV2.assign_from_config() reads it.

    Fields
    ------
    mix
        Weighted distribution of scene types.  ``SceneVarietyEngineV2`` draws from
        this when assigning types, filtered by each scene's intent.  Normalised to
        sum 1.0 in __post_init__.
    pattern_interrupt_frequency
        Insert a ``"cutaway"`` every *N* scenes (1-indexed, scene 0 excluded).
        Set to 0 to disable.
    priority_intents
        Scenes whose intent appears here always get the best-scoring type for that
        intent (via ``pick_best``), bypassing the random weighted draw.
    avoid_repetition
        When True, ``_enforce_variety`` prevents consecutive identical types.
    """

    mix: dict[str, float]      = field(default_factory=dict)
    pattern_interrupt_frequency: int  = 3
    priority_intents: list[str]       = field(default_factory=lambda: ["hook"])
    avoid_repetition: bool            = True

    def __post_init__(self) -> None:
        if self.mix:
            total = sum(self.mix.values())
            if total > 0 and abs(total - 1.0) > 0.01:
                self.mix = {k: v / total for k, v in self.mix.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "mix":                          self.mix,
            "pattern_interrupt_frequency":  self.pattern_interrupt_frequency,
            "priority_intents":             self.priority_intents,
            "avoid_repetition":             self.avoid_repetition,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScenePolicy":
        return cls(
            mix                          = d.get("mix",                          {}),
            pattern_interrupt_frequency  = int(d.get("pattern_interrupt_frequency", 3)),
            priority_intents             = d.get("priority_intents",             ["hook"]),
            avoid_repetition             = bool(d.get("avoid_repetition",         True)),
        )


@dataclass
class StrategyConfig:
    """System-wide generation strategy returned by DecisionEngineV2.decide().

    ``scene_policy`` is the unified contract passed to SceneVarietyEngineV2.
    ``scene_mix`` is kept as a convenience alias synced with ``scene_policy.mix``.
    """

    mode: str                         = "growth"
    video_length: str                 = "medium"
    avatar_frequency: float           = 0.2
    hook_aggressiveness: float        = 0.7
    scene_mix: dict[str, float]       = field(default_factory=dict)
    pace: str                         = "normal"
    scene_policy: ScenePolicy         = field(default_factory=ScenePolicy)

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {self.mode!r}")
        if self.video_length not in _VALID_LENGTHS:
            raise ValueError(f"video_length must be one of {_VALID_LENGTHS}, got {self.video_length!r}")
        if self.pace not in _VALID_PACES:
            raise ValueError(f"pace must be one of {_VALID_PACES}, got {self.pace!r}")
        if not 0.0 <= self.avatar_frequency <= 1.0:
            raise ValueError(f"avatar_frequency must be 0.0–1.0, got {self.avatar_frequency}")
        if not 0.0 <= self.hook_aggressiveness <= 1.0:
            raise ValueError(f"hook_aggressiveness must be 0.0–1.0, got {self.hook_aggressiveness}")

        # Sync scene_mix ↔ scene_policy.mix (caller may set either)
        if self.scene_policy.mix and not self.scene_mix:
            self.scene_mix = dict(self.scene_policy.mix)
        elif self.scene_mix and not self.scene_policy.mix:
            total = sum(self.scene_mix.values())
            if total > 0 and abs(total - 1.0) > 0.01:
                self.scene_mix = {k: v / total for k, v in self.scene_mix.items()}
            self.scene_policy.mix = dict(self.scene_mix)
        elif self.scene_mix:
            # Both set: normalise scene_mix, let scene_policy.mix stand as-is
            total = sum(self.scene_mix.values())
            if total > 0 and abs(total - 1.0) > 0.01:
                self.scene_mix = {k: v / total for k, v in self.scene_mix.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode":                self.mode,
            "video_length":        self.video_length,
            "avatar_frequency":    self.avatar_frequency,
            "hook_aggressiveness": self.hook_aggressiveness,
            "scene_mix":           self.scene_mix,
            "pace":                self.pace,
            "scene_policy":        self.scene_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategyConfig":
        sp_raw = d.get("scene_policy", {})
        return cls(
            mode                = d.get("mode",                "growth"),
            video_length        = d.get("video_length",        "medium"),
            avatar_frequency    = float(d.get("avatar_frequency",    0.2)),
            hook_aggressiveness = float(d.get("hook_aggressiveness", 0.7)),
            scene_mix           = d.get("scene_mix",           {}),
            pace                = d.get("pace",                "normal"),
            scene_policy        = ScenePolicy.from_dict(sp_raw) if sp_raw else ScenePolicy(),
        )


# ── PerformanceStore ──────────────────────────────────────────────────────────

_DEFAULT_METRICS: dict[str, Any] = {
    "avg_watch_pct":  0.50,
    "hook_retention": 0.50,
    "drop_points":    [],
    "ctr":            0.04,
    "recent_trend":   "stable",
}


class PerformanceStore:
    """Append-only JSON log of channel metrics + strategy used per video."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = data_dir or settings.data_dir
        self._path = self._data_dir / "performance_store.json"

    def save(
        self,
        video_id: str,
        metrics: dict[str, Any],
        strategy_mode: str,
    ) -> None:
        """Append one entry to the store."""
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
        """Return the *n* most recent entries (newest last)."""
        return self._load_raw()[-n:]

    def get_channel_metrics(self) -> dict[str, Any]:
        """Aggregate recent entries into a single channel-level metrics dict."""
        recent = self.load_recent()
        if not recent:
            return dict(_DEFAULT_METRICS)

        watch_pcts   = [e["metrics"].get("avg_watch_pct",  0.5)  for e in recent]
        hook_rets    = [e["metrics"].get("hook_retention",  0.5)  for e in recent]
        ctrs         = [e["metrics"].get("ctr",             0.04) for e in recent]

        trend = self._compute_trend(watch_pcts)

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


# ── DecisionEngineV2 ──────────────────────────────────────────────────────────

class DecisionEngineV2:
    """Channel-level strategy engine: metrics → StrategyConfig.

    Complements v1 DecisionEngine (per-story angle/format weights) with a
    system-wide mode that governs pacing, hook strength, and scene composition.
    """

    def __init__(
        self,
        performance_store: PerformanceStore | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self._store = performance_store or PerformanceStore(data_dir=data_dir)

    # ── Public API ─────────────────────────────────────────────────────────────

    def decide(self, metrics: dict[str, Any] | None = None) -> StrategyConfig:
        """Produce a StrategyConfig from channel metrics.

        *metrics* can be injected directly (e.g. from YouTube Analytics);
        when None, the store's aggregated metrics are used.
        """
        if metrics is None:
            metrics = self._load_metrics()

        mode   = self._select_mode(metrics)
        config = self._build_strategy(mode, metrics)

        logger.info(
            f"DecisionEngineV2: mode={config.mode}, pace={config.pace}, "
            f"hook={config.hook_aggressiveness:.1f}, "
            f"avg_watch_pct={metrics.get('avg_watch_pct', '?')}"
        )
        return config

    def record(
        self,
        video_id: str,
        metrics: dict[str, Any],
        strategy: StrategyConfig,
    ) -> None:
        """Persist post-publish metrics for future decide() calls."""
        self._store.save(video_id, metrics, strategy_mode=strategy.mode)

    # ── Metrics loading ────────────────────────────────────────────────────────

    def _load_metrics(self) -> dict[str, Any]:
        return self._store.get_channel_metrics()

    # ── Mode selection ─────────────────────────────────────────────────────────

    def _select_mode(self, metrics: dict[str, Any]) -> str:
        avg_watch = metrics.get("avg_watch_pct", 0.5)
        trend     = metrics.get("recent_trend",  "stable")

        if avg_watch < 0.4:
            return "growth"
        if trend == "down":
            return "experimental"
        if avg_watch > 0.55:
            return "safe"
        return "growth"

    # ── Strategy builders ──────────────────────────────────────────────────────

    def _build_strategy(self, mode: str, metrics: dict[str, Any]) -> StrategyConfig:
        if mode == "growth":
            return self._growth_strategy(metrics)
        if mode == "safe":
            return self._safe_strategy(metrics)
        return self._experimental_strategy(metrics)

    def _growth_strategy(self, metrics: dict[str, Any]) -> StrategyConfig:
        """Aggressive retention: short, fast, high-energy hooks, dynamic scenes."""
        mix = {
            "text_overlay": 0.25,
            "image":        0.20,
            "infographic":  0.20,
            "diagram":      0.20,
            "cutaway":      0.15,
        }
        return StrategyConfig(
            mode                = "growth",
            video_length        = "short",
            avatar_frequency    = 0.30,
            hook_aggressiveness = 0.90,
            scene_mix           = mix,
            pace                = "fast",
            scene_policy        = ScenePolicy(
                mix                         = mix,
                pattern_interrupt_frequency = 2,
                priority_intents            = ["hook", "shock"],
                avoid_repetition            = True,
            ),
        )

    def _safe_strategy(self, metrics: dict[str, Any]) -> StrategyConfig:
        """Stabilise: longer, calmer, image-heavy, proven format."""
        mix = {
            "image":        0.40,
            "diagram":      0.20,
            "infographic":  0.20,
            "cutaway":      0.10,
            "text_overlay": 0.10,
        }
        return StrategyConfig(
            mode                = "safe",
            video_length        = "long",
            avatar_frequency    = 0.15,
            hook_aggressiveness = 0.50,
            scene_mix           = mix,
            pace                = "normal",
            scene_policy        = ScenePolicy(
                mix                         = mix,
                pattern_interrupt_frequency = 4,
                priority_intents            = ["hook"],
                avoid_repetition            = True,
            ),
        )

    def _experimental_strategy(self, metrics: dict[str, Any]) -> StrategyConfig:
        """Randomised exploration: vary length, hooks, and scene composition."""
        mix = self._random_scene_mix()
        return StrategyConfig(
            mode                = "experimental",
            video_length        = random.choice(["short", "medium"]),
            avatar_frequency    = round(random.uniform(0.10, 0.40), 2),
            hook_aggressiveness = round(random.uniform(0.50, 1.00), 2),
            scene_mix           = mix,
            pace                = random.choice(["fast", "normal"]),
            scene_policy        = ScenePolicy(
                mix                         = mix,
                pattern_interrupt_frequency = random.choice([2, 3]),
                priority_intents            = ["hook"],
                avoid_repetition            = True,
            ),
        )

    def _random_scene_mix(self) -> dict[str, float]:
        """Generate a random scene mix that sums to 1.0."""
        types   = ["image", "text_overlay", "infographic", "diagram", "cutaway"]
        weights = [random.random() for _ in types]
        total   = sum(weights)
        return {t: round(w / total, 4) for t, w in zip(types, weights)}

    def update_scene_mix_from_feedback(self, strat_engine: Any) -> dict[str, float]:
        """Derive a normalised scene_mix from SceneStrategyEngine's EMA scores.

        Returns the mix dict (caller may log or pass it to record()).
        Typical call: after a video run, pass the SceneStrategyEngine used so its
        updated scores shape the next decide() call's scene policy.
        """
        scores = strat_engine.get_scores()
        total  = sum(scores.values())
        if total <= 0:
            return {}
        mix = {t: round(s / total, 4) for t, s in scores.items()}
        logger.info(f"DecisionEngineV2.update_scene_mix_from_feedback: {mix}")
        return mix
