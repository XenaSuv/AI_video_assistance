"""Dynamic hook selection using a Thompson-sampling multi-armed bandit.

Each hook variant is treated as a Bernoulli arm with a Beta(alpha, beta) prior.
On every video we:
  1. Sample theta_i ~ Beta(alpha_i, beta_i) for every arm
  2. Pick the arm with the highest theta_i (Thompson sampling)
  3. Record the chosen hook in data/hook_performance.json
  4. Later, record_performance() is called with CTR / retention data which
     updates alpha/beta so better-performing hooks get picked more often.

Context key (e.g. "daily", "claude", "chatgpt", "gemini", "digest") keeps
each pipeline's history separate so arms don't bleed across content types.

Public API
----------
pick_hook(variants, data_dir, context_key) -> str
    Pick the best variant (or random if all are new). Saves the choice.

record_usage(hook_text, video_id, data_dir, context_key) -> None
    Persist which hook was used for a given video_id.

record_performance(video_id, ctr_pct, retention_pct, data_dir, context_key) -> None
    Update arm statistics once YouTube Analytics data is available.
    ctr_pct     – click-through rate (0-100)
    retention_pct – average % of video watched (0-100)
    The reward signal is (ctr_pct / 10 + retention_pct / 100) / 2, clamped to [0, 1].
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────── persistence ───────────────────

def _perf_file(data_dir: Path) -> Path:
    return data_dir / "hook_performance.json"


def _load(data_dir: Path) -> dict:
    p = _perf_file(data_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def _save(data: dict, data_dir: Path) -> None:
    _perf_file(data_dir).write_text(json.dumps(data, indent=2))


# ─────────────────── arm helpers ───────────────────

def _arm_key(hook_text: str) -> str:
    """Stable identity for a hook — first 80 chars (avoids huge keys)."""
    return hook_text[:80].strip()


def _get_arm(ctx: dict, key: str) -> dict:
    """Return arm stats, initialising with Beta(1,1) = uniform prior."""
    if key not in ctx:
        ctx[key] = {"alpha": 1.0, "beta": 1.0, "uses": 0, "wins": 0.0}
    return ctx[key]


def _sample(arm: dict) -> float:
    return random.betavariate(arm["alpha"], arm["beta"])


# ─────────────────── public API ───────────────────

def pick_hook(
    variants: list[str],
    data_dir: Path,
    context_key: str,
) -> str:
    """Select the best hook from *variants* using Thompson sampling.

    Falls back to a random pick if fewer than 2 variants are supplied.
    The chosen hook is returned; call record_usage() to persist the decision.
    """
    if not variants:
        raise ValueError("pick_hook requires at least one variant")
    if len(variants) == 1:
        return variants[0]

    data = _load(data_dir)
    ctx  = data.setdefault(context_key, {})

    # Thompson sampling: draw from each arm's posterior, pick highest
    scores = []
    for v in variants:
        arm    = _get_arm(ctx, _arm_key(v))
        scores.append(_sample(arm))

    chosen_idx  = scores.index(max(scores))
    chosen      = variants[chosen_idx]
    logger.info(
        f"[hook_selector/{context_key}] picked variant {chosen_idx} "
        f"(score {scores[chosen_idx]:.3f}): {chosen[:60]!r}"
    )
    _save(data, data_dir)
    return chosen


def record_usage(
    hook_text: str,
    video_id: str,
    data_dir: Path,
    context_key: str,
) -> None:
    """Persist which hook was used for *video_id* so performance can be linked later."""
    data = _load(data_dir)
    ctx  = data.setdefault(context_key, {})
    arm  = _get_arm(ctx, _arm_key(hook_text))
    arm["uses"] += 1

    # Keep a reverse-lookup video_id → hook snippet
    data.setdefault("_video_hooks", {})[video_id] = {
        "context": context_key,
        "hook":    hook_text[:80],
    }
    _save(data, data_dir)
    logger.info(f"[hook_selector] recorded usage: video={video_id}")


def record_performance(
    video_id: str,
    ctr_pct: float,
    retention_pct: float,
    data_dir: Path,
    context_key: str | None = None,
) -> None:
    """Update arm Beta parameters from YouTube Analytics data.

    Can be called manually or by a scheduled analytics job.
    reward = mean(ctr_pct/10, retention_pct/100), clamped to [0,1].
    """
    data = _load(data_dir)

    # Look up which hook this video used
    entry = data.get("_video_hooks", {}).get(video_id)
    if not entry:
        logger.warning(f"[hook_selector] no record for video_id={video_id}")
        return

    ctx_key  = context_key or entry["context"]
    hook_key = entry["hook"]

    ctx = data.setdefault(ctx_key, {})
    arm = _get_arm(ctx, hook_key)

    reward = min(1.0, max(0.0, (ctr_pct / 10.0 + retention_pct / 100.0) / 2.0))
    arm["alpha"] += reward
    arm["beta"]  += 1.0 - reward
    arm["wins"]  += reward

    logger.info(
        f"[hook_selector/{ctx_key}] updated arm {hook_key!r}: "
        f"reward={reward:.3f}, alpha={arm['alpha']:.2f}, beta={arm['beta']:.2f}"
    )
    _save(data, data_dir)


def top_hooks(data_dir: Path, context_key: str, n: int = 5) -> list[dict]:
    """Return the top-n hooks by estimated mean CTR for *context_key* (for reporting)."""
    data = _load(data_dir)
    ctx  = data.get(context_key, {})
    arms = [
        {
            "hook":     key,
            "mean":     a["alpha"] / (a["alpha"] + a["beta"]),
            "uses":     a["uses"],
            "wins":     a["wins"],
            "alpha":    a["alpha"],
            "beta":     a["beta"],
        }
        for key, a in ctx.items()
    ]
    return sorted(arms, key=lambda x: x["mean"], reverse=True)[:n]
