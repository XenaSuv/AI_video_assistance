"""Sequence Learning Engine — sequence patterns → retention.

Individual scene quality matters, but *order* matters more:

    hook → explain → explain → [DROP] ❌
    hook → twist   → explain → retained ✅

This engine treats each video's scene-intent sequence as a learning signal.
Sliding-window pattern extraction converts sequences into overlapping n-grams,
which are then aggregated across all videos to surface which transitions
actually drive retention.

Key concepts
------------
Sequence   — ordered list of scene intents for one video:
             e.g. ["hook", "avatar", "explain", "twist", "example"]

Pattern    — a contiguous sub-sequence of fixed window size:
             window=3 → ("hook", "avatar", "explain"),
                         ("avatar", "explain", "twist"), ...

Aggregation — group all observations by pattern, compute avg_retention.

Best-next   — given the last N intents already placed, find the next intent
              that most often leads to high retention.

Applications
------------
Auto-correction:   if ("explain", "explain", "explain") is a bad pattern,
                   replace the 3rd scene's intent with "twist".

Sequence generation: greedy chain from "hook" using best_next() — an
                     "AI director" that builds retention-optimal orderings.

Drop predictor boost: bad patterns → higher predicted risk at that position.

Decision engine bias: top patterns → strategy["sequence_bias"].

Data file
---------
    data/sequence_learning.jsonl — one record per video, append-only.
"""
from __future__ import annotations

import json
import random as _stdlib_random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

# ── Constants ─────────────────────────────────────────────────────────────────

BAD_RETENTION_THRESHOLD  = 0.4
GOOD_RETENTION_THRESHOLD = 0.6
BAD_PATTERN_RISK_BOOST   = 0.20   # added to drop-predictor probability
_FALLBACK_INTENTS        = ["explain", "twist", "data", "example", "hook"]


# ── Standalone helpers ────────────────────────────────────────────────────────

def extract_patterns(sequence: list[str], window: int = 3) -> list[tuple[str, ...]]:
    """Sliding-window n-gram extraction over a sequence of intents.

    >>> extract_patterns(["hook", "explain", "twist", "example"], window=3)
    [('hook', 'explain', 'twist'), ('explain', 'twist', 'example')]
    """
    if len(sequence) < window:
        return []
    return [
        tuple(sequence[i: i + window])
        for i in range(len(sequence) - window + 1)
    ]


def _intent_from(scene: Any) -> str:
    """Extract intent from a scene object or dict (handles both formats)."""
    if isinstance(scene, dict):
        return str(scene.get("intent", scene.get("scene_intent", "unknown")))
    return str(getattr(scene, "scene_intent", getattr(scene, "intent", "unknown")))


def _set_intent(scene: Any, value: str) -> None:
    """Set intent on a scene object or dict."""
    if isinstance(scene, dict):
        scene["intent"] = value
    elif hasattr(scene, "scene_intent"):
        scene.scene_intent = value
    elif hasattr(scene, "intent"):
        scene.intent = value


# ── Persistence ───────────────────────────────────────────────────────────────

class SequenceLearningStore:
    """Append-only JSONL store for sequence records."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._path = (data_dir or settings.data_dir) / "sequence_learning.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"SequenceLearningStore: could not append ({exc})")

    def load(self) -> list[dict[str, Any]]:
        try:
            if not self._path.exists():
                return []
            lines = self._path.read_text().strip().splitlines()
            return [json.loads(ln) for ln in lines if ln.strip()]
        except Exception as exc:
            logger.warning(f"SequenceLearningStore: could not load ({exc})")
            return []


# ── SequenceLearningEngine ────────────────────────────────────────────────────

class SequenceLearningEngine:
    """Learn which scene-intent orderings drive retention.

    Typical lifecycle::

        engine = SequenceLearningEngine(data_dir=run_dir)

        # After a video with analytics:
        engine.store_sequence(run_history_record)

        # Before generating the next video:
        seq    = engine.generate_sequence(length=8)
        scenes = engine.auto_correct_sequence(scenes)

        # Decision Engine integration:
        strategy["sequence_bias"] = engine.get_sequence_bias(n=3)

        # Drop Predictor integration:
        risk_boost = engine.get_risk_boost(("explain", "explain", "explain"))
    """

    def __init__(
        self,
        data_dir:  Path | None = None,
        min_count: int         = 2,
        window:    int         = 3,
        seed:      int | None  = None,
    ) -> None:
        self._store    = SequenceLearningStore(data_dir)
        self.min_count = min_count
        self.window    = window
        self._rng      = _stdlib_random.Random(seed)

    # ── Ingestion ──────────────────────────────────────────────────────────────

    def store_sequence(self, run: dict[str, Any]) -> None:
        """Extract the intent sequence and performance from a run_history record.

        Skips records without performance data (analytics not yet arrived).
        """
        perf = run.get("performance")
        if not perf:
            logger.debug("SequenceLearningEngine: skipping — no performance data")
            return

        scenes = run.get("scenes") or []
        seq    = [_intent_from(s) for s in scenes]

        if not seq:
            logger.debug("SequenceLearningEngine: skipping — empty scene list")
            return

        record: dict[str, Any] = {
            "sequence":  seq,
            "retention": float(perf.get("avg_watch_pct", 0.0)),
            "drops":     list(perf.get("drops", [])),
            "_ts":       datetime.now(timezone.utc).isoformat(),
        }
        self._store.append(record)
        logger.info(
            f"SequenceLearningEngine: stored sequence {seq} "
            f"ret={record['retention']:.3f} drops={record['drops']}"
        )

    # ── Pattern aggregation ────────────────────────────────────────────────────

    def aggregate_patterns(
        self,
        window:    int | None = None,
        min_count: int | None = None,
    ) -> list[dict[str, Any]]:
        """Group all stored sequences by n-gram pattern; return per-pattern stats."""
        w     = window    or self.window
        m     = min_count or self.min_count
        data  = self._store.load()

        ret_lists: dict[tuple, list[float]] = defaultdict(list)
        for item in data:
            for pat in extract_patterns(item["sequence"], window=w):
                ret_lists[pat].append(float(item["retention"]))

        results = []
        for pat, vals in ret_lists.items():
            if len(vals) >= m:
                results.append({
                    "pattern":       pat,
                    "avg_retention": round(sum(vals) / len(vals), 4),
                    "count":         len(vals),
                })
        return results

    def get_top_patterns(
        self,
        min_count: int | None = None,
        window:    int | None = None,
        threshold: float      = 0.0,
    ) -> list[dict[str, Any]]:
        """Return patterns with avg_retention ≥ threshold, sorted best-first."""
        data     = self.aggregate_patterns(window=window, min_count=min_count)
        filtered = [x for x in data if x["avg_retention"] >= threshold]
        return sorted(filtered, key=lambda x: x["avg_retention"], reverse=True)

    def get_bad_patterns(
        self,
        threshold: float      = BAD_RETENTION_THRESHOLD,
        min_count: int | None = None,
        window:    int | None = None,
    ) -> list[dict[str, Any]]:
        """Return patterns below the retention threshold (to actively avoid)."""
        data     = self.aggregate_patterns(window=window, min_count=min_count)
        filtered = [x for x in data if x["avg_retention"] < threshold]
        return sorted(filtered, key=lambda x: x["avg_retention"])

    def bad_pattern_set(
        self,
        threshold: float      = BAD_RETENTION_THRESHOLD,
        min_count: int | None = None,
        window:    int | None = None,
    ) -> frozenset[tuple]:
        """Return bad pattern tuples as a frozenset for O(1) lookup."""
        return frozenset(x["pattern"] for x in self.get_bad_patterns(
            threshold=threshold, min_count=min_count, window=window
        ))

    # ── Next-scene prediction ──────────────────────────────────────────────────

    def best_next(
        self,
        prefix:    list[str] | tuple[str, ...],
        min_count: int | None = None,
    ) -> str:
        """Given the last N intents, return the intent most likely to retain viewers.

        Searches for (window=len(prefix)+1) patterns that start with prefix,
        ranks by avg_retention, and returns the best next element.

        Falls back to a random fallback intent when no qualified pattern exists.
        """
        prefix_t = tuple(prefix)
        target_w = len(prefix_t) + 1
        patterns = self.aggregate_patterns(window=target_w, min_count=min_count)

        candidates: list[tuple[float, str]] = []
        for stats in patterns:
            pat = stats["pattern"]
            if pat[:-1] == prefix_t:
                candidates.append((stats["avg_retention"], pat[-1]))

        if candidates:
            best_ret, best_intent = max(candidates, key=lambda x: x[0])
            logger.debug(
                f"SequenceLearningEngine: best_next({list(prefix)}) "
                f"→ {best_intent!r} (ret={best_ret:.3f})"
            )
            return best_intent

        fallback = self._rng.choice(_FALLBACK_INTENTS)
        logger.debug(
            f"SequenceLearningEngine: best_next({list(prefix)}) — no data, "
            f"fallback={fallback!r}"
        )
        return fallback

    # ── Sequence generation ────────────────────────────────────────────────────

    def generate_sequence(
        self,
        length: int          = 8,
        start:  list[str] | None = None,
    ) -> list[str]:
        """Greedy sequence generation using best_next().

        Starts from *start* (defaults to ["hook"]) and extends one intent at a
        time, always picking the next element that has historically led to the
        highest retention given the last two intents.
        """
        seq = list(start) if start else ["hook"]

        while len(seq) < length:
            prefix  = seq[-2:] if len(seq) >= 2 else seq[-1:]
            next_i  = self.best_next(prefix)
            seq.append(next_i)

        logger.info(f"SequenceLearningEngine: generated sequence {seq}")
        return seq

    # ── Auto-correction ────────────────────────────────────────────────────────

    def auto_correct_sequence(
        self,
        scenes:    list[Any],
        window:    int   = 3,
        threshold: float = BAD_RETENTION_THRESHOLD,
        replacement: str = "twist",
    ) -> int:
        """Replace the last element of each bad n-gram window with *replacement*.

        Mutates *scenes* in place (works with both dicts and Scene objects).
        Returns the number of corrections applied.
        """
        if len(scenes) < window:
            return 0

        bad_pats   = self.bad_pattern_set(threshold=threshold, window=window)
        corrections = 0

        for i in range(len(scenes) - window + 1):
            window_intents = tuple(_intent_from(s) for s in scenes[i: i + window])
            if window_intents in bad_pats:
                _set_intent(scenes[i + window - 1], replacement)
                logger.info(
                    f"SequenceLearningEngine: corrected pattern {window_intents} "
                    f"at position {i + window - 1} → {replacement!r}"
                )
                corrections += 1

        return corrections

    # ── Integration helpers ────────────────────────────────────────────────────

    def get_risk_boost(
        self,
        pattern:   tuple[str, ...],
        threshold: float = BAD_RETENTION_THRESHOLD,
    ) -> float:
        """Return additional risk score to add to DropPredictor probability.

        Returns BAD_PATTERN_RISK_BOOST when pattern is known-bad, else 0.0.
        """
        if pattern in self.bad_pattern_set(threshold=threshold):
            return BAD_PATTERN_RISK_BOOST
        return 0.0

    def get_sequence_bias(self, n: int = 3) -> list[tuple[str, ...]]:
        """Return the top-N best pattern tuples for strategy["sequence_bias"]."""
        top = self.get_top_patterns()
        return [x["pattern"] for x in top[:n]]
