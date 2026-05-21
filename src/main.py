"""Daily pipeline entry point.

Orchestration logic lives in PipelineOrchestrator. This module provides the
run_pipeline() function used by the scheduler and a CLI entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline_orchestrator import PipelineOrchestrator


def run_pipeline(dry_run: bool = False, skip_upload: bool = False) -> dict[str, Any]:
    """Execute the full daily pipeline. Returns a summary dict."""
    return PipelineOrchestrator().run(dry_run=dry_run, skip_upload=skip_upload)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Scrape only")
    ap.add_argument("--skip-upload", action="store_true", help="Build video but don't upload")
    args = ap.parse_args()

    run_pipeline(dry_run=args.dry_run, skip_upload=args.skip_upload)
