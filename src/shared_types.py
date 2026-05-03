"""Shared dataclasses for cross-module data exchange.

These replace dict[str, Any] at module boundaries so that attribute access
is type-checked and IDE-navigable.  Modules should import only what they need.

Serialization note: types used in JSON persistence (FeedbackRecord) keep
to_dict() / from_dict() helpers so the storage format is unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ── DecisionEngine output ─────────────────────────────────────────────────────

@dataclass
class ContentStrategy:
    """Strategic decisions produced by DecisionEngine.decide()."""
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
    context: dict           # editorial context snapshot (metadata only)
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
class PerformanceStats:
    """Aggregated performance for a single angle or format."""
    category: str           # angle key or format name
    sample_size: int
    avg_hook_score: float
    avg_watch_time: float
    success_rate: float
