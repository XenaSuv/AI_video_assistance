"""A/B Testing Engine — sequential title/thumbnail variant testing with CTR feedback.

Flow
----
PackagingEngine.generate_variants(editorial_plan, strategy_config)
    ↓
ABTestingEngine.run_test(video_id, variants)
    ├── apply variant A  → wait → collect CTR + retention_30s
    ├── apply variant B  → wait → collect CTR + retention_30s
    └── (repeat for N variants)
    ↓
_select_winner()  →  score = ctr * retention_30s (avoids clickbait trap)
    ↓
apply winner permanently
    ↓
ABTestStore.save() + suggest_strategy_adjustments()
    ↓
PackagingStore + DecisionEngineV2 feedback loop

Variant types
-------------
curiosity  — open loop, "wait — nobody saw this coming"
conflict   — explicit tension, "THE PROBLEM" (high_contrast thumbnail)
simple     — clean explainer, "EXPLAINED" (neutral thumbnail)

YouTube client interface
------------------------
ABTestingEngine accepts any object with:
    update_video(video_id, title, thumbnail_concept)  → None
    get_video_analytics(video_id)  → {"ctr": float, "retention_30s": float}

Pass youtube_client=None for a dry-run (no real API calls, metrics = 0.0).

Data store
----------
data/ab_test_results.json — append-only list of ABTestResult dicts.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class ABTestVariant:
    """One packaging variant being tested.

    All four content fields (type, title, thumbnail, hook) are set at creation.
    Metric fields (start_time, ctr, retention_30s, score) are filled in by
    ABTestingEngine during run_test().
    """
    id: str                    = "A"         # "A" | "B" | "C" | …
    type: str                  = "curiosity" # curiosity | conflict | simple
    title: str                 = ""
    thumbnail: dict[str, str]  = field(default_factory=dict)
    hook: str                  = ""
    start_time: str | None     = None
    ctr: float                 = 0.0
    retention_30s: float       = 0.0
    score: float               = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":            self.id,
            "type":          self.type,
            "title":         self.title,
            "thumbnail":     self.thumbnail,
            "hook":          self.hook,
            "start_time":    self.start_time,
            "ctr":           self.ctr,
            "retention_30s": self.retention_30s,
            "score":         self.score,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ABTestVariant":
        return cls(
            id            = d.get("id",            "A"),
            type          = d.get("type",          "curiosity"),
            title         = d.get("title",         ""),
            thumbnail     = d.get("thumbnail",     {}),
            hook          = d.get("hook",          ""),
            start_time    = d.get("start_time"),
            ctr           = float(d.get("ctr",           0.0)),
            retention_30s = float(d.get("retention_30s", 0.0)),
            score         = float(d.get("score",         0.0)),
        )


@dataclass
class ABTestResult:
    """Complete record of one A/B test run."""
    video_id: str
    variants: list[ABTestVariant]     = field(default_factory=list)
    winner: ABTestVariant | None      = None
    completed_at: str | None          = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id":     self.video_id,
            "variants":     [v.to_dict() for v in self.variants],
            "winner":       self.winner.to_dict() if self.winner else None,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ABTestResult":
        variants = [ABTestVariant.from_dict(v) for v in d.get("variants", [])]
        winner_d = d.get("winner")
        return cls(
            video_id     = d.get("video_id", ""),
            variants     = variants,
            winner       = ABTestVariant.from_dict(winner_d) if winner_d else None,
            completed_at = d.get("completed_at"),
        )


# ── ABTestStore ───────────────────────────────────────────────────────────────

class ABTestStore:
    """Append-only JSON log of completed A/B test results for feedback learning."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._path = (data_dir or settings.data_dir) / "ab_test_results.json"

    def save(self, result: ABTestResult) -> None:
        """Append one completed test to the store."""
        entries = self._load_raw()
        entries.append(result.to_dict())
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(entries, indent=2))
        logger.debug(f"ABTestStore: saved result for {result.video_id!r}")

    def load_recent(self, n: int = 20) -> list[ABTestResult]:
        """Return the *n* most recent test results."""
        return [ABTestResult.from_dict(d) for d in self._load_raw()[-n:]]

    def get_variant_type_performance(self) -> dict[str, float]:
        """Return average score per variant type across all tests."""
        entries = self._load_raw()
        groups: dict[str, list[float]] = defaultdict(list)
        for e in entries:
            winner = e.get("winner")
            if winner:
                groups[winner.get("type", "unknown")].append(
                    float(winner.get("score", winner.get("ctr", 0)))
                )
        return {t: round(sum(v) / len(v), 4) for t, v in groups.items()}

    def get_winning_patterns(self, top_n: int = 3) -> dict[str, float]:
        """Return the top *top_n* variant types by average winner score."""
        perf = self.get_variant_type_performance()
        ranked = sorted(perf.items(), key=lambda x: x[1], reverse=True)
        return dict(ranked[:top_n])

    def _load_raw(self) -> list[dict[str, Any]]:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                if isinstance(data, list):
                    return data
        except Exception as exc:
            logger.warning(f"ABTestStore: could not read store ({exc})")
        return []


# ── ABTestingEngine ───────────────────────────────────────────────────────────

class ABTestingEngine:
    """Sequential A/B test runner for video title / thumbnail variants.

    Each variant is exposed for *wait_hours* before metrics are collected.
    Score = CTR × retention_30s prevents clickbait variants winning on clicks
    alone.  When retention data is unavailable (0.0), score falls back to CTR.

    The YouTube client is injected for testability:
        youtube_client=None  → dry-run (no API calls, metrics stay 0.0)
    """

    def __init__(
        self,
        youtube_client: Any | None = None,
        wait_hours: float = 2.0,
        data_dir: Path | None = None,
    ) -> None:
        self.youtube      = youtube_client
        self._wait_hours  = wait_hours
        self._store       = ABTestStore(data_dir=data_dir)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run_test(
        self,
        video_id: str,
        variants: list[ABTestVariant],
        *,
        wait_fn: Callable[[float], None] | None = None,
    ) -> ABTestResult:
        """Run a sequential test across all variants; apply the winner.

        *wait_fn* overrides the default ``time.sleep`` — pass
        ``lambda h: None`` in unit tests to skip real waits.
        """
        _wait = wait_fn or self._real_wait

        if not variants:
            raise ValueError("run_test: variants list must not be empty")

        for i, variant in enumerate(variants):
            variant.id         = chr(ord("A") + i)
            variant.start_time = datetime.now(timezone.utc).isoformat()

            self._apply_variant(video_id, variant)
            _wait(self._wait_hours)

            metrics              = self._get_metrics(video_id)
            variant.ctr          = float(metrics.get("ctr", 0.0))
            variant.retention_30s = float(metrics.get("retention_30s", 0.0))
            variant.score        = self._score(variant)

            logger.info(
                f"ABTest [{video_id}] variant {variant.id} ({variant.type}): "
                f"ctr={variant.ctr:.4f}  ret30={variant.retention_30s:.2f}  "
                f"score={variant.score:.4f}"
            )

        winner = self._select_winner(variants)
        self._apply_variant(video_id, winner)

        result = ABTestResult(
            video_id     = video_id,
            variants     = variants,
            winner       = winner,
            completed_at = datetime.now(timezone.utc).isoformat(),
        )

        self._store.save(result)
        logger.info(
            f"ABTest [{video_id}] WINNER: {winner.id} ({winner.type}) "
            f"score={winner.score:.4f}"
        )
        return result

    def suggest_strategy_adjustments(self) -> dict[str, Any]:
        """Analyse historical test results and return StrategyConfig adjustments.

        Returns a dict the caller can apply to the next DecisionEngineV2 run::

            adjustments = engine.suggest_strategy_adjustments()
            # e.g. {"preferred_variant_type": "conflict",
            #        "hook_aggressiveness_delta": 0.1}
        """
        perf = self._store.get_variant_type_performance()
        if not perf:
            return {}

        best_type = max(perf, key=lambda k: perf[k])
        delta     = {
            "conflict":  +0.10,
            "curiosity": +0.05,
            "simple":    -0.05,
        }.get(best_type, 0.0)

        adjustments = {
            "preferred_variant_type":      best_type,
            "hook_aggressiveness_delta":   delta,
        }
        logger.info(f"ABTest: strategy adjustments → {adjustments}")
        return adjustments

    def apply_winner_feedback(
        self,
        result: ABTestResult,
        packaging_store: Any,
    ) -> None:
        """Persist the winner's CTR to PackagingStore for title/thumbnail learning.

        *packaging_store* is a PackagingStore instance.  The winner's variant
        type maps to the matching title_pattern so PackagingStore.get_best_patterns()
        can surface it.
        """
        if result.winner is None:
            return

        from src.packaging_engine import PackagingResult

        winner = result.winner
        pr = PackagingResult(
            idea          = "",
            hook          = winner.hook,
            title         = winner.title,
            thumbnail     = winner.thumbnail,
            title_pattern = _VARIANT_TYPE_TO_PATTERN.get(winner.type, "default"),
        )
        packaging_store.record(result.video_id, pr, ctr=winner.ctr)
        logger.info(
            f"ABTest: winner feedback recorded — type={winner.type!r} "
            f"ctr={winner.ctr:.4f} → PackagingStore"
        )

    # ── Core logic ─────────────────────────────────────────────────────────────

    def _select_winner(self, variants: list[ABTestVariant]) -> ABTestVariant:
        """Return the variant with the highest composite score."""
        return max(variants, key=lambda v: v.score)

    def _score(self, variant: ABTestVariant) -> float:
        """score = ctr × retention_30s; falls back to ctr when retention = 0."""
        if variant.retention_30s > 0:
            return round(variant.ctr * variant.retention_30s, 6)
        return round(variant.ctr, 6)

    # ── YouTube client wrappers ────────────────────────────────────────────────

    def _apply_variant(self, video_id: str, variant: ABTestVariant) -> None:
        """Push title + thumbnail concept to YouTube (or log in dry-run mode)."""
        if self.youtube is None:
            logger.debug(
                f"ABTest [DRY-RUN] apply {variant.id} ({variant.type}) "
                f"title={variant.title!r}"
            )
            return
        try:
            self.youtube.update_video(
                video_id          = video_id,
                title             = variant.title,
                thumbnail_concept = variant.thumbnail,
            )
        except Exception as exc:
            logger.warning(f"ABTest: _apply_variant failed ({exc})")

    def _get_metrics(self, video_id: str) -> dict[str, float]:
        """Fetch CTR + 30-second retention from YouTube Analytics."""
        if self.youtube is None:
            return {"ctr": 0.0, "retention_30s": 0.0}
        try:
            return self.youtube.get_video_analytics(video_id)
        except Exception as exc:
            logger.warning(f"ABTest: _get_metrics failed ({exc})")
            return {"ctr": 0.0, "retention_30s": 0.0}

    # ── Wait ──────────────────────────────────────────────────────────────────

    def _real_wait(self, hours: float) -> None:
        import time
        logger.info(f"ABTest: waiting {hours:.1f} h for impressions to accumulate…")
        time.sleep(hours * 3600)


# ── Variant-type → title_pattern mapping (for PackagingStore bridge) ──────────

_VARIANT_TYPE_TO_PATTERN: dict[str, str] = {
    "curiosity": "game_changer",
    "conflict":  "hype_problem",
    "simple":    "practical",
}
