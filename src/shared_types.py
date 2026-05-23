"""Shared dataclasses for cross-module data exchange.

These replace dict[str, Any] at module boundaries so that attribute access
is type-checked and IDE-navigable.  Modules should import only what they need.

Serialization note: types used in JSON persistence (FeedbackRecord) keep
to_dict() / from_dict() helpers so the storage format is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── ScenePolicy / StrategyConfig ──────────────────────────────────────────────
# Contract surface between the decision layer and SceneVarietyEngineV2.
# Written by the decision engine; read by assign_from_config().

_VALID_MODES        = frozenset({"growth", "safe", "experimental"})
_VALID_LENGTHS      = frozenset({"short", "medium", "long"})
_VALID_PACES        = frozenset({"fast", "normal", "slow"})
_VALID_SCENE_TYPES  = frozenset({"image", "text_overlay", "infographic", "diagram", "cutaway", "avatar"})


@dataclass
class ScenePolicy:
    """Scene-level execution policy carried inside StrategyConfig.

    Decision Engine writes it; SceneVarietyEngineV2.assign_from_config() reads it.
    """

    mix: dict[str, float]               = field(default_factory=dict)
    pattern_interrupt_frequency: int    = 3
    priority_intents: list[str]         = field(default_factory=lambda: ["hook"])
    avoid_repetition: bool              = True

    def __post_init__(self) -> None:
        if self.mix:
            total = sum(self.mix.values())
            if total > 0 and abs(total - 1.0) > 0.01:
                self.mix = {k: v / total for k, v in self.mix.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "mix":                         self.mix,
            "pattern_interrupt_frequency": self.pattern_interrupt_frequency,
            "priority_intents":            self.priority_intents,
            "avoid_repetition":            self.avoid_repetition,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScenePolicy":
        return cls(
            mix                         = d.get("mix",                         {}),
            pattern_interrupt_frequency = int(d.get("pattern_interrupt_frequency", 3)),
            priority_intents            = d.get("priority_intents",            ["hook"]),
            avoid_repetition            = bool(d.get("avoid_repetition",        True)),
        )


@dataclass
class StrategyConfig:
    """System-wide generation strategy.

    ``scene_policy`` is the unified contract passed to SceneVarietyEngineV2.
    ``scene_mix`` is a convenience alias synced with ``scene_policy.mix``.
    """

    mode: str                    = "growth"
    video_length: str            = "medium"
    avatar_frequency: float      = 0.2
    hook_aggressiveness: float   = 0.7
    scene_mix: dict[str, float]  = field(default_factory=dict)
    pace: str                    = "normal"
    scene_policy: ScenePolicy    = field(default_factory=ScenePolicy)

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
        if self.scene_policy.mix and not self.scene_mix:
            self.scene_mix = dict(self.scene_policy.mix)
        elif self.scene_mix and not self.scene_policy.mix:
            total = sum(self.scene_mix.values())
            if total > 0 and abs(total - 1.0) > 0.01:
                self.scene_mix = {k: v / total for k, v in self.scene_mix.items()}
            self.scene_policy.mix = dict(self.scene_mix)
        elif self.scene_mix:
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


# ── DecisionEngineV3 output ───────────────────────────────────────────────────

@dataclass
class ContentStrategy:
    """Strategic decisions produced by DecisionEngineV3.decide()."""
    angle_weights: dict[str, float]
    format_weights: dict[str, float]
    exploration_rate: float
    mode: str               # "growth" | "safe" | "balanced"
    confidence: float


# ── HookOptimizer output ──────────────────────────────────────────────────────

@dataclass
class HookOptimizationResult:
    """Optimized hook patterns produced by HookOptimizer.run()."""
    recommended_patterns: list[str]
    avoid_patterns: list[str]
    context: dict[str, Any]  # editorial context snapshot (metadata only)
    exploration_enabled: bool
    style_bias: dict[str, float]


# ── MicroHookAgent output ─────────────────────────────────────────────────────

@dataclass
class InsertedHook:
    """A single micro-hook injected into the script."""
    position: int
    type: str
    text: str
    optimized: bool = False


@dataclass
class MicroHookResult:
    """Result of MicroHookAgent.run()."""
    final_script: str
    inserted_hooks: list[InsertedHook]
    optimization: HookOptimizationResult


# ── HumanizerAgent output ─────────────────────────────────────────────────────

@dataclass
class HumanizationResult:
    """Result of HumanizerAgent.run() / humanize_script()."""
    final_script: str
    changes: list[str]


# ── FeedbackAnalyzer output ───────────────────────────────────────────────────

@dataclass
class WindowStats:
    """Aggregated metrics for a specific time window (7-day or lifetime)."""
    sample_size: int
    avg_hook_score: float
    avg_watch_time: float
    success_rate: float


@dataclass
class PerformanceStats:
    """Aggregated performance for a single angle or format.

    Flat fields (sample_size, avg_hook_score, avg_watch_time, success_rate)
    represent lifetime aggregates and are kept for backward compatibility.
    Use ``recent_7d`` and ``lifetime`` for window-aware comparisons.
    """
    category: str           # angle key or format name
    sample_size: int        # lifetime total — backward-compat alias
    avg_hook_score: float   # lifetime — backward-compat alias
    avg_watch_time: float   # lifetime — backward-compat alias
    success_rate: float     # lifetime — backward-compat alias
    # Rolling windows — None when no data in that window
    recent_7d: WindowStats | None = None
    lifetime: WindowStats | None = None
