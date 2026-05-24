"""Per-run dependency bundle for the daily pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.checkpoint import PipelineCheckpoint
    from src.deduplicator import SeenStories
    from src.live_state import LiveState
    from src.pipeline_observer import PipelineObserver


@dataclass
class PipelineContext:
    """Bundles per-run infrastructure so step constructors take one ctx arg.

    Build once at the top of PipelineOrchestrator.run() and pass down to
    _ScriptStep, _MediaBuilder, and _PublishStep instead of threading four
    separate kwargs through every constructor.
    """

    run_dir:  Path
    cp:       "PipelineCheckpoint"
    observer: "PipelineObserver"
    live:     "LiveState"
    seen:     "SeenStories"
