"""Quality gate: checks script, audio, and historical performance before publishing.

Two tiers:
  Hard (passed=False) — blocks publish; pipeline raises QualityGateError.
  Soft (warnings)     — logged but publish proceeds.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.ffmpeg_utils as ffmpeg_utils


class QualityGateError(RuntimeError):
    """Raised when a hard quality check fails."""


@dataclass
class QualityReport:
    passed: bool
    score: float                    # 0.0–1.0 composite
    issues: list[str] = field(default_factory=list)     # blocking
    warnings: list[str] = field(default_factory=list)   # informational


# ── individual checks ─────────────────────────────────────────────────────────

def check_script(script: Any, min_words_per_scene: int = 50, min_scenes: int = 3) -> QualityReport:
    """Verify the script has enough content."""
    issues: list[str] = []
    warnings: list[str] = []

    if not script.hook or len(script.hook.strip()) < 10:
        issues.append("Hook is empty or too short")

    if len(script.scenes) < min_scenes:
        issues.append(f"Too few scenes: {len(script.scenes)} < {min_scenes}")

    short_scenes = [
        s.idx for s in script.scenes
        if len(s.narration.split()) < min_words_per_scene
    ]
    if short_scenes:
        warnings.append(
            f"Scene(s) {short_scenes} have fewer than {min_words_per_scene} words"
        )

    total_words = sum(len(s.narration.split()) for s in script.scenes)
    if total_words < min_words_per_scene * min_scenes:
        issues.append(f"Script too short: {total_words} total words")

    passed = len(issues) == 0
    score = 1.0 if passed else 0.0
    return QualityReport(passed=passed, score=score, issues=issues, warnings=warnings)


def check_audio(script: Any, audio_dir: Path, min_total_sec: int = 60) -> QualityReport:
    """Verify all scene audio files exist and have sufficient duration."""
    issues: list[str] = []
    warnings: list[str] = []

    missing = []
    total_dur = 0.0
    for scene in script.scenes:
        mp3 = audio_dir / f"scene_{scene.idx:02d}.mp3"
        if not mp3.exists():
            missing.append(scene.idx)
            continue
        try:
            dur = ffmpeg_utils.duration(mp3)
            total_dur += dur
            if dur < 2.0:
                warnings.append(f"scene_{scene.idx:02d}.mp3 is very short ({dur:.1f}s)")
        except Exception as exc:
            warnings.append(f"scene_{scene.idx:02d}.mp3 unreadable: {exc}")

    if missing:
        issues.append(f"Missing audio for scene(s): {missing}")

    if total_dur < min_total_sec:
        issues.append(f"Total audio too short: {total_dur:.0f}s < {min_total_sec}s")

    passed = len(issues) == 0
    score = max(0.0, 1.0 - len(missing) / max(len(script.scenes), 1))
    return QualityReport(passed=passed, score=score, issues=issues, warnings=warnings)


def check_historical_performance(
    feedback_history: list[dict[str, Any]],
    min_hook_score: float = 0.45,
    min_samples: int = 5,
) -> QualityReport:
    """Warn if recent videos are consistently underperforming.

    Only acts when there are enough samples to be confident.
    Never blocks (soft check only) — we don't want to halt the pipeline
    on sparse data.
    """
    warnings: list[str] = []
    score = 1.0

    recent = [r for r in feedback_history if r.get("hook_score") is not None]
    recent = recent[-10:]  # only last 10 entries

    if len(recent) < min_samples:
        return QualityReport(passed=True, score=score, warnings=[
            f"Not enough feedback history ({len(recent)}/{min_samples}) to evaluate performance"
        ])

    avg_hook = sum(r["hook_score"] for r in recent) / len(recent)
    score = min(avg_hook / min_hook_score, 1.0)

    if avg_hook < min_hook_score:
        warnings.append(
            f"Recent hook scores averaging {avg_hook:.2f} "
            f"(threshold {min_hook_score}) — consider revising content strategy"
        )

    return QualityReport(passed=True, score=score, warnings=warnings)


# ── composite gate ────────────────────────────────────────────────────────────

def run_gate(
    script: Any,
    audio_dir: Path,
    feedback_history: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> QualityReport:
    """Run all checks and return a composite QualityReport.

    Raises QualityGateError if any hard check fails.
    """
    cfg = config or {}

    checks = [
        check_script(
            script,
            min_words_per_scene=cfg.get("min_words_per_scene", 50),
            min_scenes=cfg.get("min_scenes", 3),
        ),
        check_audio(
            script,
            audio_dir,
            min_total_sec=cfg.get("min_total_sec", 60),
        ),
        check_historical_performance(
            feedback_history,
            min_hook_score=cfg.get("min_hook_score", 0.45),
            min_samples=cfg.get("min_samples", 5),
        ),
    ]

    all_issues = [issue for c in checks for issue in c.issues]
    all_warnings = [w for c in checks for w in c.warnings]
    composite_score = sum(c.score for c in checks) / len(checks)
    passed = len(all_issues) == 0

    report = QualityReport(
        passed=passed,
        score=round(composite_score, 2),
        issues=all_issues,
        warnings=all_warnings,
    )

    for w in report.warnings:
        logger.warning(f"Quality gate: {w}")

    if not passed:
        for issue in report.issues:
            logger.error(f"Quality gate FAILED: {issue}")
        raise QualityGateError(
            f"Quality gate blocked publish — {len(all_issues)} issue(s): "
            + "; ".join(all_issues)
        )

    logger.info(f"Quality gate passed (score={report.score:.2f})")
    return report
