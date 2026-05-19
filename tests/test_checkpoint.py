"""Tests for PipelineCheckpoint."""
import json
from pathlib import Path

import pytest

from src.checkpoint import PipelineCheckpoint


@pytest.fixture
def cp(tmp_path):
    return PipelineCheckpoint(tmp_path)


class TestIsMarkDone:
    def test_fresh_checkpoint_is_not_done(self, cp):
        assert not cp.is_done("scrape")

    def test_marked_step_is_done(self, cp):
        cp.mark_done("scrape")
        assert cp.is_done("scrape")

    def test_other_steps_not_affected(self, cp):
        cp.mark_done("scrape")
        assert not cp.is_done("script")

    def test_metadata_stored_and_retrieved(self, cp):
        cp.mark_done("scrape", {"num_items": 42})
        assert cp.metadata("scrape")["num_items"] == 42

    def test_metadata_excludes_status_and_ts(self, cp):
        cp.mark_done("scrape", {"num_items": 5})
        meta = cp.metadata("scrape")
        assert "status" not in meta
        assert "ts" not in meta

    def test_empty_metadata_for_unknown_step(self, cp):
        assert cp.metadata("nonexistent") == {}


class TestPersistence:
    def test_survives_reload(self, tmp_path):
        cp1 = PipelineCheckpoint(tmp_path)
        cp1.mark_done("scrape", {"n": 10})

        cp2 = PipelineCheckpoint(tmp_path)
        assert cp2.is_done("scrape")
        assert cp2.metadata("scrape")["n"] == 10

    def test_checkpoint_file_is_valid_json(self, tmp_path):
        cp = PipelineCheckpoint(tmp_path)
        cp.mark_done("scrape")
        data = json.loads((tmp_path / "checkpoint.json").read_text())
        assert data["steps"]["scrape"]["status"] == "done"

    def test_corrupted_file_starts_fresh(self, tmp_path):
        (tmp_path / "checkpoint.json").write_text("not json{{")
        cp = PipelineCheckpoint(tmp_path)
        assert not cp.is_done("scrape")


class TestCompletedSteps:
    def test_empty_initially(self, cp):
        assert cp.completed_steps() == []

    def test_reflects_marked_steps_in_order(self, cp):
        cp.mark_done("scrape")
        cp.mark_done("script")
        cp.mark_done("voice")
        assert cp.completed_steps() == ["scrape", "script", "voice"]


class TestReset:
    def test_reset_single_step(self, cp):
        cp.mark_done("scrape")
        cp.mark_done("script")
        cp.reset("scrape")
        assert not cp.is_done("scrape")
        assert cp.is_done("script")

    def test_reset_all_steps(self, cp):
        cp.mark_done("scrape")
        cp.mark_done("script")
        cp.reset()
        assert cp.completed_steps() == []

    def test_reset_persists(self, tmp_path):
        cp1 = PipelineCheckpoint(tmp_path)
        cp1.mark_done("scrape")
        cp1.reset("scrape")

        cp2 = PipelineCheckpoint(tmp_path)
        assert not cp2.is_done("scrape")
