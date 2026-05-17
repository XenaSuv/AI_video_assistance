"""Shorts Experiment Engine — rapid hypothesis testing via YouTube Shorts.

Every Short is an experiment, not content.  This engine treats Shorts as a
10x-faster learning signal: 10 shorts = 10 hook/pacing/persona experiments
that would otherwise take 10 daily videos (10 days) to learn.

Experiment anatomy (25s max):
    0–3s   hook    — the variable under test
    3–15s  core    — condensed key point
    15–25s payoff  — CTA / memorable closer

Primary metric: retention_3s (% who stay past 3 seconds).
When a hook crosses the winner threshold (default 0.70), the engine:
  1. Injects its text into the next run's hook_candidates list
  2. Boosts hook_aggressiveness in DecisionEngineV3 context
  3. Records the winning combination in CrossLearningEngine

Data files:
    data/shorts_experiments.jsonl   — one record per experiment (generated)
    data/shorts_results.jsonl       — one record per result (collected)
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.auto_action_engine import AutoActionEngine
from src.cross_learning_engine import CrossLearningEngine
from src.youtube_analytics import get_video_metrics


# ── Constants ─────────────────────────────────────────────────────────────────

_HOOK_TEMPLATES: dict[str, list[str]] = {
    "conflict": [
        "This changes everything about {topic}...",
        "Here's the problem with {topic} nobody talks about",
        "This might actually fail — and here's why",
        "{topic} is under attack. Here's what's really happening.",
    ],
    "curiosity": [
        "Nobody is talking about this {topic} development",
        "The secret behind {topic} that experts hide",
        "Wait — you didn't know this about {topic}?",
        "This {topic} fact will surprise you",
    ],
    "shock": [
        "They secretly banned {topic}. Here's the proof.",
        "{topic} just crossed a line that can't be uncrossed",
        "This {topic} number is genuinely terrifying",
    ],
    "question": [
        "What if {topic} is already smarter than we think?",
        "Is {topic} about to replace everything?",
        "Can {topic} actually solve this problem?",
    ],
    "negative": [
        "Why {topic} is more overhyped than you realise",
        "{topic} isn't working — here's the data",
        "The hidden downside of {topic} everyone ignores",
    ],
}

_WINNER_THRESHOLD     = 0.70   # retention_3s above this = strong signal
_MIN_SIGNAL_COUNT     = 3      # min winners before signal is trusted
_HOOK_AGG_BOOST       = 0.3   # added to hook_aggressiveness on strong signal
_MAX_VARIANTS_PER_RUN = 5


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ShortsExperiment:
    experiment_id: str
    story_title:   str
    hook_type:     str
    hook_text:     str
    core_text:     str
    payoff_text:   str
    created_at:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    video_id:      str = ""      # filled after upload
    video_path:    str = ""      # filled after generation

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShortsExperiment":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ShortsExperimentResult:
    experiment_id:   str
    video_id:        str
    retention_3s:    float = 0.0   # fraction of viewers past 3s; primary metric
    avg_watch_pct:   float = 0.0   # 0–1
    completion_rate: float = 0.0   # 0–1 (watched to end)
    views:           int   = 0
    collected_at:    str   = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_winner:       bool  = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ShortsExperimentResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Persistence ───────────────────────────────────────────────────────────────

class ShortsExperimentStore:
    """Append-only JSONL backing store for experiments and results."""

    def __init__(self, data_dir: Path | None = None) -> None:
        base = data_dir or settings.data_dir
        self._exp_path    = base / "shorts_experiments.jsonl"
        self._result_path = base / "shorts_results.jsonl"
        base.mkdir(parents=True, exist_ok=True)

    # ── experiments ───────────────────────────────────────────────────────────

    def save_experiment(self, exp: ShortsExperiment) -> None:
        self._append(self._exp_path, exp.to_dict())

    def update_experiment(self, experiment_id: str, updates: dict[str, Any]) -> None:
        """Rewrite the experiments file updating matching record in-place."""
        records = self._load_all(self._exp_path)
        changed = False
        for r in records:
            if r.get("experiment_id") == experiment_id:
                r.update(updates)
                changed = True
        if changed:
            self._exp_path.write_text(
                "\n".join(json.dumps(r) for r in records) + "\n"
            )

    def load_experiments(self) -> list[ShortsExperiment]:
        return [ShortsExperiment.from_dict(r) for r in self._load_all(self._exp_path)]

    def load_experiment(self, experiment_id: str) -> ShortsExperiment | None:
        for r in self._load_all(self._exp_path):
            if r.get("experiment_id") == experiment_id:
                return ShortsExperiment.from_dict(r)
        return None

    # ── results ───────────────────────────────────────────────────────────────

    def save_result(self, result: ShortsExperimentResult) -> None:
        self._append(self._result_path, result.to_dict())

    def load_results(self) -> list[ShortsExperimentResult]:
        return [ShortsExperimentResult.from_dict(r) for r in self._load_all(self._result_path)]

    def has_result(self, experiment_id: str) -> bool:
        return any(
            r.get("experiment_id") == experiment_id
            for r in self._load_all(self._result_path)
        )

    # ── helpers ───────────────────────────────────────────────────────────────

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as exc:
            logger.warning(f"ShortsExperimentStore: write failed ({path.name}): {exc}")

    def _load_all(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        records: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:
                logger.debug(f"Skipped malformed record: {exc}")
        return records


# ── Engine ────────────────────────────────────────────────────────────────────

class ShortsExperimentEngine:
    """Generates, tracks, and learns from Shorts experiments.

    Usage in pipeline:
        # Before editorial_brain:
        best_hooks = engine.get_best_hooks()          # inject into hook_candidates
        signal     = engine.get_strong_signal()       # boost hook_aggressiveness

        # After EN upload:
        experiments = engine.generate(story)
        for exp in experiments:
            video_path = create_short_video(exp.hook_text + "\\n" + exp.core_text + "\\n" + exp.payoff_text)
            video_id   = upload_short(video_path, title=exp.hook_text[:100])
            engine.mark_uploaded(exp.experiment_id, video_id=video_id, video_path=str(video_path))

        # Deferred feedback (next run):
        engine.collect_analytics()
        engine.analyze()
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._store    = ShortsExperimentStore(data_dir)
        self._data_dir = data_dir or settings.data_dir

    # ── generation ────────────────────────────────────────────────────────────

    def generate(self, base_story: dict[str, Any]) -> list[ShortsExperiment]:
        """Return up to _MAX_VARIANTS_PER_RUN hook experiments for base_story.

        Selects one template per hook type family, filling {topic} from the
        story title.  Core and payoff are short placeholders — callers should
        replace them with OpenAI-generated text when available.
        """
        title  = base_story.get("title", "AI")
        topic  = _extract_topic(title)
        source = base_story.get("source", "")

        variants: list[ShortsExperiment] = []
        hook_types = list(_HOOK_TEMPLATES.keys())
        for hook_type in hook_types[:_MAX_VARIANTS_PER_RUN]:
            templates = _HOOK_TEMPLATES[hook_type]
            hook_text = templates[hash(title) % len(templates)].format(topic=topic)
            exp = ShortsExperiment(
                experiment_id = str(uuid.uuid4()),
                story_title   = title,
                hook_type     = hook_type,
                hook_text     = hook_text,
                core_text     = _make_core(title, source),
                payoff_text   = "Follow for daily AI updates.",
            )
            self._store.save_experiment(exp)
            variants.append(exp)
            logger.info(f"ShortsExperiment generated: [{hook_type}] {hook_text!r}")

        return variants

    def mark_uploaded(
        self,
        experiment_id: str,
        video_id: str,
        video_path: str = "",
    ) -> None:
        self._store.update_experiment(experiment_id, {
            "video_id":   video_id,
            "video_path": video_path,
        })
        logger.info(f"ShortsExperiment {experiment_id}: uploaded → {video_id}")

    # ── analytics collection ──────────────────────────────────────────────────

    def collect_analytics(self, min_age_hours: float = 4.0) -> int:
        """Fetch YouTube metrics for all experiments without results yet.

        Returns the number of new results collected.
        """
        experiments = self._store.load_experiments()
        collected = 0
        now = datetime.now(timezone.utc)

        for exp in experiments:
            if not exp.video_id:
                continue
            if self._store.has_result(exp.experiment_id):
                continue
            # Skip if experiment was uploaded too recently
            try:
                created = datetime.fromisoformat(exp.created_at)
                age_h = (now - created).total_seconds() / 3600
                if age_h < min_age_hours:
                    continue
            except Exception as exc:
                logger.debug(f"Skipped malformed record: {exc}")

            try:
                yt = get_video_metrics(exp.video_id)
                if not yt:
                    continue
                views    = int(yt.get("views", 0))
                avg_pct  = float(yt.get("avg_view_percentage", 0.0)) / 100.0
                # Approximate retention_3s from avg_watch_pct using a
                # heuristic: 3s / 25s ≈ 12% of the video, so if avg_watch_pct
                # > 0.12 it's likely most viewers stayed past 3s.  Without the
                # full retention curve we use avg_pct as a proxy.
                retention_3s = min(1.0, avg_pct * 2.5)  # heuristic upscale
                result = ShortsExperimentResult(
                    experiment_id   = exp.experiment_id,
                    video_id        = exp.video_id,
                    retention_3s    = round(retention_3s, 4),
                    avg_watch_pct   = round(avg_pct, 4),
                    completion_rate = round(avg_pct, 4),
                    views           = views,
                    is_winner       = retention_3s >= _WINNER_THRESHOLD,
                )
                self._store.save_result(result)
                collected += 1
                logger.info(
                    f"ShortsExperiment result: {exp.experiment_id} "
                    f"retention_3s={retention_3s:.2f} winner={result.is_winner}"
                )
            except Exception as exc:
                logger.warning(
                    f"ShortsExperiment: analytics collection failed for "
                    f"{exp.video_id}: {exc}"
                )

        return collected

    # ── signal extraction ─────────────────────────────────────────────────────

    def get_best_hooks(self, n: int = 5) -> list[str]:
        """Return the top-n hook texts ranked by retention_3s."""
        results = self._store.load_results()
        if not results:
            return []
        results_sorted = sorted(results, key=lambda r: r.retention_3s, reverse=True)
        experiments_by_id = {
            e.experiment_id: e for e in self._store.load_experiments()
        }
        hooks: list[str] = []
        for result in results_sorted[:n]:
            exp = experiments_by_id.get(result.experiment_id)
            if exp and exp.hook_text:
                hooks.append(exp.hook_text)
        return hooks

    def get_strong_signal(self) -> float | None:
        """Return hook_aggressiveness boost if enough winners exist, else None.

        Returns _HOOK_AGG_BOOST when _MIN_SIGNAL_COUNT or more experiments
        have crossed _WINNER_THRESHOLD, indicating a reliable pattern.
        """
        results = self._store.load_results()
        winners = [r for r in results if r.is_winner]
        if len(winners) >= _MIN_SIGNAL_COUNT:
            logger.info(
                f"ShortsExperiment: strong signal detected "
                f"({len(winners)} winners ≥ threshold {_WINNER_THRESHOLD})"
            )
            return _HOOK_AGG_BOOST
        return None

    def record_result(
        self,
        experiment_id: str,
        retention_3s: float,
        avg_watch_pct: float = 0.0,
        completion_rate: float = 0.0,
        views: int = 0,
    ) -> ShortsExperimentResult:
        """Manually record a result (used in tests and manual pipelines)."""
        exp = self._store.load_experiment(experiment_id)
        video_id = exp.video_id if exp else ""
        result = ShortsExperimentResult(
            experiment_id   = experiment_id,
            video_id        = video_id,
            retention_3s    = round(retention_3s, 4),
            avg_watch_pct   = round(avg_watch_pct, 4),
            completion_rate = round(completion_rate, 4),
            views           = views,
            is_winner       = retention_3s >= _WINNER_THRESHOLD,
        )
        self._store.save_result(result)
        return result

    # ── learning integration ──────────────────────────────────────────────────

    def analyze(self) -> dict[str, Any]:
        """Feed winning experiments into CrossLearningEngine and AutoActionEngine.

        Returns a summary of actions taken.
        """
        results      = self._store.load_results()
        experiments  = {e.experiment_id: e for e in self._store.load_experiments()}
        winners      = [r for r in results if r.is_winner]
        summary: dict[str, Any] = {
            "total_experiments": len(results),
            "winners":           len(winners),
            "best_hook_type":    None,
            "cross_learning":    False,
            "auto_action":       False,
        }
        if not winners:
            return summary

        # Best winner by retention_3s
        best_result = max(winners, key=lambda r: r.retention_3s)
        best_exp    = experiments.get(best_result.experiment_id)
        if not best_exp:
            return summary

        summary["best_hook_type"] = best_exp.hook_type

        # Feed into CrossLearningEngine
        try:
            CrossLearningEngine(data_dir=self._data_dir).learn({
                "strategy": {"pace": "fast", "persona": "energetic"},
                "packaging": {"hook_type": best_exp.hook_type},
                "editorial": {"topic": "AI news"},
                "scenes": [{"type": "text_overlay"}],
                "performance": {
                    "avg_watch_pct": best_result.avg_watch_pct,
                    "ctr": 0.0,
                },
            })
            summary["cross_learning"] = True
            logger.info(
                f"ShortsExperiment → CrossLearningEngine: "
                f"hook_type={best_exp.hook_type!r} "
                f"retention_3s={best_result.retention_3s:.2f}"
            )
        except Exception as exc:
            logger.warning(f"ShortsExperiment: CrossLearningEngine.learn() failed: {exc}")

        # Trigger AutoActionEngine signal when top performer exists
        try:
            engine = AutoActionEngine(run_type="daily")
            strategy = engine.load_strategy()
            strategy["hook_aggressiveness"] = min(
                1.0, strategy.get("hook_aggressiveness", 0.5) + _HOOK_AGG_BOOST
            )
            strategy["packaging_style"] = best_exp.hook_type
            engine.save_strategy(strategy)
            summary["auto_action"] = True
            logger.info(
                f"ShortsExperiment → AutoActionEngine: "
                f"boosted hook_aggressiveness, set packaging_style={best_exp.hook_type!r}"
            )
        except Exception as exc:
            logger.warning(f"ShortsExperiment: AutoActionEngine update failed: {exc}")

        return summary


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_topic(title: str) -> str:
    """Return a short topic phrase from the story title (first 4 words)."""
    words = title.split()
    return " ".join(words[:4]) if words else "AI"


def _make_core(title: str, source: str) -> str:
    """Build a one-sentence core for the Short from the story title."""
    base = title.rstrip(".!?")
    if source:
        return f"{base} — from {source}."
    return f"{base}."
