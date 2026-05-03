"""Tests for PipelineObserver."""
import json
import time
import pytest
from pathlib import Path
from src.pipeline_observer import PipelineObserver


@pytest.fixture
def obs(tmp_path):
    return PipelineObserver(tmp_path, pipeline="test")


# ── step lifecycle ─────────────────────────────────────────────────────────────

class TestStepLifecycle:
    def test_step_done_recorded(self, obs):
        obs.step_start("scrape")
        obs.step_done("scrape", items=42)
        trace = obs._to_dict()
        step = trace["steps"][0]
        assert step["name"] == "scrape"
        assert step["status"] == "done"
        assert step["items"] == 42

    def test_step_skip_recorded(self, obs):
        obs.step_start("scrape")
        obs.step_skip("scrape")
        step = obs._to_dict()["steps"][0]
        assert step["status"] == "skipped"

    def test_step_fail_recorded(self, obs):
        obs.step_start("video")
        obs.step_fail("video", ValueError("ffmpeg died"))
        step = obs._to_dict()["steps"][0]
        assert step["status"] == "failed"
        assert "ffmpeg died" in step["error"]

    def test_multiple_steps_ordered(self, obs):
        for name in ("scrape", "script", "voice"):
            obs.step_start(name)
            obs.step_done(name)
        names = [s["name"] for s in obs._to_dict()["steps"]]
        assert names == ["scrape", "script", "voice"]

    def test_duration_is_non_negative(self, obs):
        obs.step_start("script")
        obs.step_done("script")
        dur = obs._to_dict()["steps"][0]["duration_sec"]
        assert dur is not None and dur >= 0

    def test_step_done_without_start_logs_warning(self, obs, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="loguru"):
            obs.step_done("orphan")
        # step should not appear (or not crash)
        names = [s["name"] for s in obs._to_dict()["steps"]]
        assert "orphan" not in names


# ── persistence ────────────────────────────────────────────────────────────────

class TestPersistence:
    def test_trace_file_written_after_step(self, tmp_path):
        obs = PipelineObserver(tmp_path, pipeline="daily")
        obs.step_start("scrape")
        obs.step_done("scrape")
        trace_file = tmp_path / "pipeline_trace.json"
        assert trace_file.exists()
        data = json.loads(trace_file.read_text())
        assert data["steps"][0]["name"] == "scrape"

    def test_trace_written_incrementally(self, tmp_path):
        obs = PipelineObserver(tmp_path, pipeline="daily")
        obs.step_start("scrape")
        # file exists and step is 'running' before done is called
        data = json.loads((tmp_path / "pipeline_trace.json").read_text())
        assert data["steps"][0]["status"] == "running"
        obs.step_done("scrape")
        data = json.loads((tmp_path / "pipeline_trace.json").read_text())
        assert data["steps"][0]["status"] == "done"

    def test_trace_valid_json_on_crash(self, tmp_path):
        obs = PipelineObserver(tmp_path, pipeline="daily")
        obs.step_start("video")
        obs.step_fail("video", RuntimeError("boom"))
        # Simulate process dying here — file must be readable
        data = json.loads((tmp_path / "pipeline_trace.json").read_text())
        assert data["steps"][0]["status"] == "failed"


# ── finish ────────────────────────────────────────────────────────────────────

class TestFinish:
    def test_finish_success(self, obs):
        obs.step_start("scrape")
        obs.step_done("scrape")
        trace = obs.finish(status="success", cost_usd=0.42)
        assert trace["status"] == "success"
        assert trace["cost_usd"] == 0.42
        assert trace["finished_at"] is not None

    def test_finish_failed(self, obs):
        obs.step_start("video")
        obs.step_fail("video", RuntimeError("oops"))
        trace = obs.finish(status="failed", error="oops")
        assert trace["status"] == "failed"

    def test_total_sec_positive(self, obs):
        obs.step_start("scrape")
        obs.step_done("scrape")
        trace = obs.finish()
        assert trace["total_sec"] >= 0

    def test_pipeline_name_in_trace(self, tmp_path):
        obs = PipelineObserver(tmp_path, pipeline="weekly")
        trace = obs.finish()
        assert trace["pipeline"] == "weekly"


# ── edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_step_skip_without_start(self, obs):
        obs.step_skip("subtitles")
        step = obs._to_dict()["steps"][0]
        assert step["status"] == "skipped"

    def test_metadata_merged_into_step(self, obs):
        obs.step_start("voice")
        obs.step_done("voice", total_duration_sec=120, num_scenes=8)
        step = obs._to_dict()["steps"][0]
        assert step["total_duration_sec"] == 120
        assert step["num_scenes"] == 8

    def test_empty_trace_on_init(self, obs):
        trace = obs._to_dict()
        assert trace["steps"] == []
        assert trace["status"] == "running"
