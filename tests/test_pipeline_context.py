"""Tests for src/pipeline_context.py — PipelineContext dataclass."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.pipeline_context import PipelineContext


def _make_ctx(tmp_path: Path) -> PipelineContext:
    return PipelineContext(
        run_dir=tmp_path,
        cp=MagicMock(),
        observer=MagicMock(),
        live=MagicMock(),
        seen=MagicMock(),
    )


class TestPipelineContext:
    def test_fields_stored(self, tmp_path):
        cp = MagicMock()
        observer = MagicMock()
        live = MagicMock()
        seen = MagicMock()
        ctx = PipelineContext(
            run_dir=tmp_path, cp=cp, observer=observer, live=live, seen=seen
        )
        assert ctx.run_dir is tmp_path
        assert ctx.cp is cp
        assert ctx.observer is observer
        assert ctx.live is live
        assert ctx.seen is seen

    def test_run_dir_is_path(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        assert isinstance(ctx.run_dir, Path)

    def test_fields_are_mutable(self, tmp_path):
        ctx = _make_ctx(tmp_path)
        new_cp = MagicMock()
        ctx.cp = new_cp
        assert ctx.cp is new_cp

    def test_two_contexts_are_independent(self, tmp_path):
        ctx_a = _make_ctx(tmp_path)
        ctx_b = _make_ctx(tmp_path)
        ctx_a.cp = MagicMock()
        assert ctx_a.cp is not ctx_b.cp
