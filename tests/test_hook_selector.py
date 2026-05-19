"""Tests for hook_selector module — Thompson sampling hook selection."""
from __future__ import annotations

import json

import pytest

from src.hook_selector import pick_hook, record_performance, record_usage, top_hooks

# ── pick_hook() ───────────────────────────────────────────────────────────────

class TestPickHook:
    def test_raises_on_empty_variants(self, tmp_path):
        with pytest.raises(ValueError):
            pick_hook([], tmp_path, "test")

    def test_returns_single_variant_directly(self, tmp_path):
        result = pick_hook(["only hook"], tmp_path, "test")
        assert result == "only hook"

    def test_picks_one_of_two_variants(self, tmp_path):
        variants = ["hook A", "hook B"]
        result = pick_hook(variants, tmp_path, "daily")
        assert result in variants

    def test_picks_one_of_many_variants(self, tmp_path):
        variants = [f"hook variant {i}" for i in range(5)]
        result = pick_hook(variants, tmp_path, "weekly")
        assert result in variants

    def test_creates_performance_file(self, tmp_path):
        pick_hook(["hook one", "hook two"], tmp_path, "daily")
        perf_file = tmp_path / "hook_performance.json"
        assert perf_file.exists()

    def test_saves_context_key_in_file(self, tmp_path):
        pick_hook(["hook one", "hook two"], tmp_path, "my_context")
        data = json.loads((tmp_path / "hook_performance.json").read_text())
        assert "my_context" in data

    def test_multiple_calls_are_stable(self, tmp_path):
        variants = ["hook A", "hook B", "hook C"]
        for _ in range(5):
            result = pick_hook(variants, tmp_path, "daily")
            assert result in variants


# ── record_usage() ────────────────────────────────────────────────────────────

class TestRecordUsage:
    def test_increments_uses_count(self, tmp_path):
        record_usage("My hook text here", "vid_001", tmp_path, "daily")
        data = json.loads((tmp_path / "hook_performance.json").read_text())
        arm = data["daily"]["My hook text here"]
        assert arm["uses"] == 1

    def test_stores_video_hook_mapping(self, tmp_path):
        record_usage("My hook text here", "vid_001", tmp_path, "daily")
        data = json.loads((tmp_path / "hook_performance.json").read_text())
        assert "vid_001" in data["_video_hooks"]
        assert data["_video_hooks"]["vid_001"]["context"] == "daily"

    def test_multiple_usages_accumulate(self, tmp_path):
        record_usage("My hook text here", "vid_001", tmp_path, "daily")
        record_usage("My hook text here", "vid_002", tmp_path, "daily")
        data = json.loads((tmp_path / "hook_performance.json").read_text())
        arm = data["daily"]["My hook text here"]
        assert arm["uses"] == 2

    def test_different_hooks_stored_separately(self, tmp_path):
        record_usage("First hook text here", "vid_001", tmp_path, "daily")
        record_usage("Second hook text here", "vid_002", tmp_path, "daily")
        data = json.loads((tmp_path / "hook_performance.json").read_text())
        assert "First hook text here" in data["daily"]
        assert "Second hook text here" in data["daily"]

    def test_long_hook_truncated_to_80_chars(self, tmp_path):
        long_hook = "x" * 100
        record_usage(long_hook, "vid_001", tmp_path, "daily")
        data = json.loads((tmp_path / "hook_performance.json").read_text())
        key = long_hook[:80]
        assert key in data["daily"]

    def test_separate_context_keys(self, tmp_path):
        record_usage("Same hook text here", "vid_001", tmp_path, "daily")
        record_usage("Same hook text here", "vid_002", tmp_path, "weekly")
        data = json.loads((tmp_path / "hook_performance.json").read_text())
        assert "daily" in data
        assert "weekly" in data


# ── record_performance() ──────────────────────────────────────────────────────

class TestRecordPerformance:
    def _setup(self, tmp_path, hook="My hook text here", video_id="vid_001", context="daily"):
        record_usage(hook, video_id, tmp_path, context)
        return hook

    def test_updates_alpha_on_good_performance(self, tmp_path):
        self._setup(tmp_path)
        data_before = json.loads((tmp_path / "hook_performance.json").read_text())
        alpha_before = data_before["daily"]["My hook text here"]["alpha"]

        record_performance("vid_001", ctr_pct=5.0, retention_pct=70.0, data_dir=tmp_path)

        data_after = json.loads((tmp_path / "hook_performance.json").read_text())
        alpha_after = data_after["daily"]["My hook text here"]["alpha"]
        assert alpha_after > alpha_before

    def test_updates_beta_on_poor_performance(self, tmp_path):
        self._setup(tmp_path)
        data_before = json.loads((tmp_path / "hook_performance.json").read_text())
        beta_before = data_before["daily"]["My hook text here"]["beta"]

        record_performance("vid_001", ctr_pct=0.0, retention_pct=0.0, data_dir=tmp_path)

        data_after = json.loads((tmp_path / "hook_performance.json").read_text())
        beta_after = data_after["daily"]["My hook text here"]["beta"]
        assert beta_after > beta_before

    def test_reward_is_clamped_to_0_1(self, tmp_path):
        self._setup(tmp_path)
        # Very high values should clamp to 1.0
        record_performance("vid_001", ctr_pct=100.0, retention_pct=100.0, data_dir=tmp_path)
        data = json.loads((tmp_path / "hook_performance.json").read_text())
        arm = data["daily"]["My hook text here"]
        # alpha should be 1 (prior) + 1.0 (reward clamped) = 2.0
        assert abs(arm["alpha"] - 2.0) < 1e-9

    def test_no_record_logs_warning_and_returns(self, tmp_path):
        # Should not raise even if video_id not found
        record_performance("nonexistent_vid", ctr_pct=5.0, retention_pct=60.0, data_dir=tmp_path)

    def test_explicit_context_key_overrides_stored(self, tmp_path):
        record_usage("My hook text here", "vid_001", tmp_path, "daily")
        # Call with explicit context_key different from stored "daily"
        record_performance("vid_001", ctr_pct=3.0, retention_pct=50.0,
                           data_dir=tmp_path, context_key="daily")
        data = json.loads((tmp_path / "hook_performance.json").read_text())
        arm = data["daily"]["My hook text here"]
        assert arm["alpha"] > 1.0

    def test_wins_accumulate_correctly(self, tmp_path):
        self._setup(tmp_path)
        record_performance("vid_001", ctr_pct=10.0, retention_pct=100.0, data_dir=tmp_path)
        data = json.loads((tmp_path / "hook_performance.json").read_text())
        arm = data["daily"]["My hook text here"]
        # reward = min(1, (10/10 + 100/100) / 2) = min(1, 1.0) = 1.0
        assert abs(arm["wins"] - 1.0) < 1e-9


# ── top_hooks() ───────────────────────────────────────────────────────────────

class TestTopHooks:
    def test_returns_empty_list_when_no_data(self, tmp_path):
        result = top_hooks(tmp_path, "daily")
        assert result == []

    def test_returns_hooks_sorted_by_mean(self, tmp_path):
        record_usage("Best hook text here long", "vid_001", tmp_path, "daily")
        record_usage("Worst hook text here x", "vid_002", tmp_path, "daily")
        # Give vid_001 a good performance (high reward → high alpha)
        record_performance("vid_001", ctr_pct=10.0, retention_pct=100.0, data_dir=tmp_path)
        # Give vid_002 a bad performance (low reward → high beta)
        record_performance("vid_002", ctr_pct=0.0, retention_pct=0.0, data_dir=tmp_path)

        result = top_hooks(tmp_path, "daily", n=2)
        assert len(result) == 2
        # Best hook should be first
        assert result[0]["mean"] >= result[1]["mean"]

    def test_returns_up_to_n_hooks(self, tmp_path):
        for i in range(7):
            record_usage(f"Hook number {i} text here", f"vid_{i:03d}", tmp_path, "daily")
        result = top_hooks(tmp_path, "daily", n=3)
        assert len(result) <= 3

    def test_returns_all_when_fewer_than_n(self, tmp_path):
        record_usage("Hook one text here now", "vid_001", tmp_path, "daily")
        record_usage("Hook two text here now", "vid_002", tmp_path, "daily")
        result = top_hooks(tmp_path, "daily", n=10)
        assert len(result) == 2

    def test_result_contains_expected_fields(self, tmp_path):
        record_usage("My hook text here long", "vid_001", tmp_path, "daily")
        result = top_hooks(tmp_path, "daily", n=5)
        assert len(result) == 1
        item = result[0]
        assert "hook" in item
        assert "mean" in item
        assert "uses" in item
        assert "wins" in item

    def test_unknown_context_key_returns_empty(self, tmp_path):
        record_usage("My hook text", "vid_001", tmp_path, "daily")
        result = top_hooks(tmp_path, "other_context", n=5)
        assert result == []

    def test_mean_is_between_0_and_1(self, tmp_path):
        record_usage("My hook text here now", "vid_001", tmp_path, "daily")
        result = top_hooks(tmp_path, "daily")
        for item in result:
            assert 0.0 <= item["mean"] <= 1.0
