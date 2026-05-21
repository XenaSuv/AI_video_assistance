"""Stateless helper functions shared across pipeline modules.

Extracted from pipeline_orchestrator to keep that module focused on
orchestration logic. All functions here are pure or do simple file I/O.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.ffmpeg_utils as ffmpeg_utils
from config import settings
from src.decision_engine_v3 import DecisionEngineV3, PerformanceStore, UnifiedStrategy
from src.retention_correction_engine import Correction
from src.script_generator import Scene, VideoScript
from src.shared_types import ContentStrategy

# ── Stateless helpers ─────────────────────────────────────────────────────────

def _get_intro_duration(intro_path: Path | None) -> float:
    """Return duration in seconds of the intro clip, or 0.0 if not present."""
    if not intro_path or not intro_path.exists():
        return 0.0
    return ffmpeg_utils.duration(intro_path)


def _get_shared_outro() -> Path | None:
    """Return the shared outro clip used across all pipelines."""
    outro_path = settings.source_dir / "ai-news-outro.mp4"
    return outro_path if outro_path.exists() else None


def _load_cached_script(path: Path) -> VideoScript | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return VideoScript(
        title=data["title"],
        description=data["description"],
        tags=data["tags"],
        hook=data["hook"],
        hook_variants=data.get("hook_variants", []),
        scenes=[
            Scene(idx=i, **{k: v for k, v in s.items() if k != "idx"})
            for i, s in enumerate(data["scenes"])
        ],
    )


def _load_audio_durations(script: VideoScript, audio_dir: Path) -> None:
    """Populate scene.duration_sec from existing mp3 files."""
    for s in script.scenes:
        p = audio_dir / f"scene_{s.idx:02d}.mp3"
        s.duration_sec = int(ffmpeg_utils.duration(p)) + 1


def _setup_logging(run_dir: Path) -> None:
    from src.json_log_sink import JsonSink
    from src.log_filter import scrub
    logger.remove()
    logger.add(
        sys.stderr, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        filter=scrub,
    )
    logger.add(run_dir / "run.log", level="DEBUG", rotation="10 MB", filter=scrub)
    logger.add(JsonSink(run_dir / "run.jsonl"), level="DEBUG", filter=scrub)


def _needs_video_rebuild(video_path: Path) -> bool:
    """Return True when a cached final video is missing or built with stale rules."""
    assembled_dir = video_path.parent / "assembled"
    legacy_end_card = any(
        path.exists()
        for path in (
            assembled_dir / "end_card.png",
            assembled_dir / "end_card.mp4",
        )
    )
    legacy_title_cards = any(assembled_dir.glob("title_*.mp4"))
    intro_outro_audio_missing = any(
        path.exists() and not ffmpeg_utils.has_audio_stream(path)
        for path in (
            assembled_dir / "intro_resized.mp4",
            assembled_dir / "outro_resized.mp4",
            assembled_dir / "body_with_outro.mp4",
        )
    )
    if not video_path.exists():
        return True
    if legacy_end_card or legacy_title_cards or intro_outro_audio_missing:
        if legacy_end_card:
            reason = "legacy generated end card"
        elif legacy_title_cards:
            reason = "legacy standalone title cards"
        else:
            reason = "intro/outro cache without audio stream"
        logger.warning(f"Cached video uses {reason}; rebuilding: {video_path}")
        video_path.unlink(missing_ok=True)
        for cached in (
            assembled_dir / "content.mp4",
            assembled_dir / "content_music.mp4",
            assembled_dir / "body_with_outro.mp4",
            assembled_dir / "body_with_music.mp4",
            assembled_dir / "end_card.png",
            assembled_dir / "end_card.mp4",
            assembled_dir / "intro_with_audio.mp4",
            assembled_dir / "outro_with_audio.mp4",
        ):
            cached.unlink(missing_ok=True)
        for cached in assembled_dir.glob("title_*.mp4"):
            cached.unlink(missing_ok=True)
        return True
    try:
        if ffmpeg_utils.has_audio_stream(video_path):
            return False
    except Exception as exc:
        logger.warning(f"Cached video probe failed for {video_path.name}: {exc}")

    logger.warning(f"Cached video has no audio stream; rebuilding: {video_path}")
    video_path.unlink(missing_ok=True)
    for cached in (
        assembled_dir / "content.mp4",
        assembled_dir / "content_music.mp4",
        assembled_dir / "end_card.png",
        assembled_dir / "end_card.mp4",
    ):
        cached.unlink(missing_ok=True)
    for cached in assembled_dir.glob("title_*.mp4"):
        cached.unlink(missing_ok=True)
    return True


def _unified_strategy_to_content_strategy(strategy: UnifiedStrategy) -> ContentStrategy:
    """Bridge DecisionEngineV3 UnifiedStrategy to ContentStrategy for editorial_brain.

    v3 operates at channel level (mode, pace, hook_aggressiveness, packaging_style).
    ContentStrategy is story-level (angle_weights, format_weights, exploration_rate).

    angle_weights and format_weights are left empty — editorial_brain already
    reads its own FeedbackAnalyzer data to score those, so no signal is lost.

    exploration_rate is derived from hook_aggressiveness: a more aggressive hook
    posture means less random exploration (we're committing to proven tactics).
    confidence is tied to mode: retention_fix → high, stable → medium-high,
    growth → medium, packaging_focus → medium.
    """
    mode_map = {
        "growth":          "growth",
        "packaging_focus": "growth",    # aggressive posture, same energy as growth
        "retention_fix":   "safe",      # conservative: fix before pushing
        "stable":          "balanced",
    }
    mode = mode_map.get(strategy.mode, "balanced")
    exploration_rate = round(max(0.1, min(0.4, 0.4 - strategy.hook_aggressiveness * 0.3)), 2)
    confidence = {
        "retention_fix":   0.8,
        "stable":          0.7,
        "growth":          0.5,
        "packaging_focus": 0.6,
    }.get(strategy.mode, 0.5)
    return ContentStrategy(
        angle_weights={},
        format_weights={},
        exploration_rate=exploration_rate,
        mode=mode,
        confidence=confidence,
    )


def _classify_hook_type(hook: str) -> str:
    """Return the variant type (conflict / curiosity / simple) for a hook string."""
    lower = hook.lower()
    if any(k in lower for k in (
        "problem", "threat", "fail", "wrong", "danger", "crisis",
        "battle", "war", " vs ", "collapse", "kill",
    )):
        return "conflict"
    if any(k in lower for k in (
        "nobody", "secret", "surprising", "wait", "weird", "actually",
        "hidden", "but here", "?", "you won't", "you didn't",
    )):
        return "curiosity"
    return "simple"


def _build_scene_map(scenes: list[dict[str, Any]], curve_len: int) -> dict[int, dict[str, Any]]:
    """Map retention-curve bucket indices to scene metadata.

    Distributes curve indices proportionally across scenes by duration_sec.
    Falls back to equal slices when durations are unavailable.
    """
    if not scenes or curve_len == 0:
        return {}
    durations = [s.get("duration_sec", 60) for s in scenes]
    total = sum(durations) or len(scenes)
    scene_map: dict[int, dict[str, Any]] = {}
    bucket = 0
    for scene_idx, (scene, dur) in enumerate(zip(scenes, durations, strict=False)):
        n_buckets = max(1, round(curve_len * dur / total))
        for b in range(bucket, min(bucket + n_buckets, curve_len)):
            scene_map[b] = {
                "scene_idx":  scene_idx,
                "scene_type": scene.get("scene_type", "image"),
                "intent":     scene.get("scene_intent", scene.get("intent", "explain")),
            }
        bucket += n_buckets
    return scene_map


def _load_retention_state(data_dir: Path) -> dict[str, Any]:
    """Load persisted retention correction state written by the deferred feedback loop."""
    path = data_dir / "retention_state.json"
    try:
        if path.exists():
            data: dict[str, Any] = json.loads(path.read_text())
            return data
    except Exception as exc:
        logger.debug(f"_load_retention_state: {exc}")
    return {}


def _save_retention_state(
    data_dir: Path,
    corrections: list[Correction],
    adjustments: dict[str, Any],
) -> None:
    """Persist retention correction results so the next pipeline run can use them."""
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "retention_state.json").write_text(json.dumps({
        "corrections": [c.to_dict() for c in corrections],
        "adjustments": adjustments,
    }, indent=2))


def _build_v3_context(
    perf_store: "PerformanceStore",
    v3_engine: "DecisionEngineV3",
    thompson_preferred_type: str | None,
    predicted_risks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble the context dict passed to DecisionEngineV3.decide().

    Separated from _step_script so it can be tested without standing up a full
    PipelineOrchestrator instance.
    """
    return {
        "metrics": perf_store.get_channel_metrics(),
        "bandit": {
            "scene": {},
            "packaging": {
                "preferred_variant_type": thompson_preferred_type or "",
            },
        },
        "prediction": {"risks": predicted_risks},
        "history": v3_engine.load_history(),
    }
