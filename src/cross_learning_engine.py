"""Cross-Learning Engine — combination-level pattern learning.

Individual bandit arms optimise one dimension at a time (hook style, scene
policy, etc.).  This engine records the *full combination* that ran for each
video and learns which combinations of factors actually drive retention.

Key insight
-----------
    conflict hook + avatar + fast pace → 🔥
    explain  hook + image  + slow pace → 💀

The winning element alone is not enough; what matters is how elements interact.

Combination key
---------------
    {hook_type, scene_type, persona, pace, topic}

Each observation is one video: that key → (retention, ctr).
After enough observations (min_count ≥ 3) the engine can surface the best
proven combinations and override / guide DecisionEngineV3.

Exploration / exploitation
--------------------------
    recommend() uses epsilon-greedy with exploration_rate=0.2 (20% random).
    This prevents the system from converging on a local optimum.

Integration
-----------
    # After run_cycle:
    engine.learn(run_history_record)

    # Before next decide():
    best = engine.recommend(context={"topic": "AI news"})
    if best:
        strategy.persona = best["persona"]
        strategy.pace    = best["pace"]

Data file
---------
    data/cross_learning.jsonl — append-only, one record per line.
"""
from __future__ import annotations

import json
import random as _stdlib_random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

# ── Constants ─────────────────────────────────────────────────────────────────

_HOOK_TYPES  = ["conflict", "curiosity", "simple"]
_SCENE_TYPES = ["avatar", "image", "diagram", "text_overlay"]
_PERSONAS    = ["energetic", "curious", "clear_explainer", "balanced"]
_PACES       = ["fast", "normal", "slow"]


# ── Persistence ───────────────────────────────────────────────────────────────

class CrossLearningStore:
    """Append-only JSONL store for combination observations."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._path = (data_dir or settings.data_dir) / "cross_learning.jsonl"

    def append(self, record: dict[str, Any]) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"CrossLearningStore: could not append ({exc})")

    def load(self) -> list[dict[str, Any]]:
        try:
            if not self._path.exists():
                return []
            lines = self._path.read_text().strip().splitlines()
            return [json.loads(ln) for ln in lines if ln.strip()]
        except Exception as exc:
            logger.warning(f"CrossLearningStore: could not load ({exc})")
            return []


# ── CrossLearningEngine ───────────────────────────────────────────────────────

class CrossLearningEngine:
    """Learn which combinations of (hook, scene, persona, pace, topic) win.

    Lifecycle::

        engine = CrossLearningEngine(data_dir=run_dir)

        # After a video cycle (once analytics arrive):
        engine.learn(run_history_record)

        # Feed best combo back into strategy:
        best = engine.recommend({"topic": "AI news"})
        if best:
            strategy["persona"] = best["persona"]
            strategy["pace"]    = best["pace"]
    """

    def __init__(
        self,
        data_dir:         Path | None = None,
        exploration_rate: float       = 0.2,
        seed:             int | None  = None,
    ) -> None:
        self._store           = CrossLearningStore(data_dir)
        self.exploration_rate = exploration_rate
        self._rng             = _stdlib_random.Random(seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def learn(self, run_data: dict[str, Any]) -> None:
        """Extract the combo + result from a run-history record and persist it.

        Skips records without performance data (analytics not yet available).
        """
        record = self._extract_record(run_data)
        if record is None:
            logger.debug("CrossLearningEngine: skipping — no performance data yet")
            return
        self._store.append(record)
        logger.info(
            f"CrossLearningEngine: learned combo {record['combo']} → "
            f"ret={record['result']['retention']:.3f} "
            f"ctr={record['result']['ctr']:.3f}"
        )

    def recommend(
        self,
        context:          dict[str, Any] | None = None,
        min_count:        int                   = 3,
        exploration_rate: float | None          = None,
    ) -> dict[str, Any] | None:
        """Return the best combo for the given context (epsilon-greedy).

        Returns None when there is not enough data to recommend anything.
        Exploration returns a random combo dict (not persisted — caller decides).
        """
        rate = exploration_rate if exploration_rate is not None else self.exploration_rate
        if self._rng.random() < rate:
            combo = self._random_combo()
            logger.debug(f"CrossLearningEngine: exploring → {combo}")
            return combo

        topic = (context or {}).get("topic")
        top   = self.get_top_combinations(min_count=min_count, topic=topic)

        if not top and topic:
            top = self.get_top_combinations(min_count=min_count)

        if not top:
            logger.debug("CrossLearningEngine: no qualified combos yet")
            return None

        best = top[0]["combo"]
        logger.info(
            f"CrossLearningEngine: recommending {best} "
            f"(avg_ret={top[0]['avg_retention']:.3f}, n={top[0]['count']})"
        )
        return best

    def aggregate(self) -> list[dict[str, Any]]:
        """Group all observations by combo key, compute per-group statistics."""
        data = self._store.load()

        retention_lists: dict[tuple, list[float]] = defaultdict(list)
        ctr_lists:       dict[tuple, list[float]] = defaultdict(list)

        for item in data:
            key = tuple(sorted(item["combo"].items()))
            retention_lists[key].append(float(item["result"].get("retention", 0.0)))
            ctr_lists[key].append(      float(item["result"].get("ctr",       0.0)))

        results = []
        for key, ret_vals in retention_lists.items():
            ctr_vals = ctr_lists[key]
            results.append({
                "combo":         dict(key),
                "avg_retention": round(sum(ret_vals) / len(ret_vals), 4),
                "avg_ctr":       round(sum(ctr_vals) / len(ctr_vals), 4),
                "count":         len(ret_vals),
            })

        return results

    def get_top_combinations(
        self,
        min_count: int       = 3,
        topic:     str | None = None,
    ) -> list[dict[str, Any]]:
        """Return combos with ≥ min_count observations, sorted by avg_retention."""
        data = self.aggregate()
        filtered = [x for x in data if x["count"] >= min_count]
        if topic:
            filtered = [x for x in filtered if x["combo"].get("topic") == topic]
        return sorted(filtered, key=lambda x: x["avg_retention"], reverse=True)

    def get_worst_combinations(self, min_count: int = 3) -> list[dict[str, Any]]:
        """Return the lowest-retention combos — useful for debugging."""
        data = self.aggregate()
        filtered = [x for x in data if x["count"] >= min_count]
        return sorted(filtered, key=lambda x: x["avg_retention"])

    # ── Extraction helpers ─────────────────────────────────────────────────────

    def _extract_record(self, run: dict[str, Any]) -> dict[str, Any] | None:
        """Build a {combo, result} record from a run_history entry.

        Returns None if performance data is absent (analytics not yet arrived).
        """
        perf = run.get("performance")
        if not perf:
            return None

        strategy  = run.get("strategy")  or {}
        packaging = run.get("packaging") or {}
        editorial = run.get("editorial") or {}

        combo: dict[str, Any] = {
            "hook_type":  packaging.get("hook_type",  "curiosity"),
            "scene_type": self._dominant_scene(run),
            "persona":    strategy.get("persona",     "balanced"),
            "pace":       strategy.get("pace",        "normal"),
            "topic":      editorial.get("topic",      "general"),
        }
        result: dict[str, float] = {
            "retention": float(perf.get("avg_watch_pct", 0.0)),
            "ctr":       float(perf.get("ctr",           0.0)),
        }
        return {
            "combo":  combo,
            "result": result,
            "_ts":    datetime.now(timezone.utc).isoformat(),
        }

    def _dominant_scene(self, run: dict[str, Any]) -> str:
        """Return the most-used scene type in this run."""
        scenes = run.get("scenes") or []
        if not scenes:
            return "unknown"
        counts = Counter(s.get("type", "unknown") for s in scenes)
        return counts.most_common(1)[0][0]

    def _random_combo(self) -> dict[str, Any]:
        """Random combo for exploration — drawn uniformly from known options."""
        return {
            "hook_type":  self._rng.choice(_HOOK_TYPES),
            "scene_type": self._rng.choice(_SCENE_TYPES),
            "persona":    self._rng.choice(_PERSONAS),
            "pace":       self._rng.choice(_PACES),
        }
