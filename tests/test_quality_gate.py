"""Tests for quality gate checks."""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.quality_gate import (
    QualityGateError,
    QualityReport,
    check_audio,
    check_historical_performance,
    check_script,
    run_gate,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _scene(idx: int, words: int = 100) -> MagicMock:
    s = MagicMock()
    s.idx = idx
    s.narration = " ".join(["word"] * words)
    return s


def _script(num_scenes: int = 5, words_per_scene: int = 100, hook: str = "Great hook here") -> MagicMock:
    s = MagicMock()
    s.hook = hook
    s.scenes = [_scene(i, words_per_scene) for i in range(num_scenes)]
    return s


# ── check_script ──────────────────────────────────────────────────────────────

class TestCheckScript:
    def test_passes_with_good_script(self):
        report = check_script(_script())
        assert report.passed

    def test_fails_with_no_hook(self):
        report = check_script(_script(hook=""))
        assert not report.passed
        assert any("Hook" in i for i in report.issues)

    def test_fails_with_too_few_scenes(self):
        report = check_script(_script(num_scenes=2), min_scenes=3)
        assert not report.passed
        assert any("scenes" in i for i in report.issues)

    def test_warns_on_short_scenes(self):
        report = check_script(_script(num_scenes=5, words_per_scene=30), min_words_per_scene=50)
        assert report.passed  # soft check — doesn't block
        assert len(report.warnings) > 0

    def test_fails_when_total_words_too_low(self):
        report = check_script(_script(num_scenes=3, words_per_scene=10), min_words_per_scene=50, min_scenes=3)
        assert not report.passed


# ── check_audio ───────────────────────────────────────────────────────────────

class TestCheckAudio:
    def test_passes_when_all_files_exist(self, tmp_path):
        script = _script(num_scenes=3)
        for i in range(3):
            (tmp_path / f"scene_{i:02d}.mp3").touch()

        with patch("src.quality_gate.ffmpeg_utils.duration", return_value=30.0):
            report = check_audio(script, tmp_path, min_total_sec=60)

        assert report.passed

    def test_fails_when_file_missing(self, tmp_path):
        script = _script(num_scenes=3)
        (tmp_path / "scene_00.mp3").touch()
        # scene_01.mp3 and scene_02.mp3 missing

        with patch("src.quality_gate.ffmpeg_utils.duration", return_value=30.0):
            report = check_audio(script, tmp_path)

        assert not report.passed
        assert any("Missing audio" in i for i in report.issues)

    def test_fails_when_total_duration_too_short(self, tmp_path):
        script = _script(num_scenes=3)
        for i in range(3):
            (tmp_path / f"scene_{i:02d}.mp3").touch()

        with patch("src.quality_gate.ffmpeg_utils.duration", return_value=5.0):
            report = check_audio(script, tmp_path, min_total_sec=60)

        assert not report.passed
        assert any("Total audio" in i for i in report.issues)

    def test_warns_on_very_short_scene(self, tmp_path):
        script = _script(num_scenes=2)
        for i in range(2):
            (tmp_path / f"scene_{i:02d}.mp3").touch()

        def fake_duration(path):
            return 1.0 if "00" in str(path) else 60.0

        with patch("src.quality_gate.ffmpeg_utils.duration", side_effect=fake_duration):
            report = check_audio(script, tmp_path, min_total_sec=30)

        assert report.passed
        assert any("very short" in w for w in report.warnings)


# ── check_historical_performance ──────────────────────────────────────────────

class TestCheckHistoricalPerformance:
    def test_passes_with_good_history(self):
        history = [{"hook_score": 0.7}] * 8
        report = check_historical_performance(history, min_hook_score=0.45, min_samples=5)
        assert report.passed
        assert not report.warnings

    def test_warns_when_scores_below_threshold(self):
        history = [{"hook_score": 0.3}] * 8
        report = check_historical_performance(history, min_hook_score=0.45, min_samples=5)
        assert report.passed  # soft — never blocks
        assert any("averaging" in w for w in report.warnings)

    def test_passes_with_not_enough_samples(self):
        history = [{"hook_score": 0.2}] * 3
        report = check_historical_performance(history, min_hook_score=0.45, min_samples=5)
        assert report.passed
        assert any("Not enough" in w for w in report.warnings)

    def test_only_uses_last_10_entries(self):
        # 15 old bad entries + 10 recent good entries
        old = [{"hook_score": 0.2}] * 15
        recent = [{"hook_score": 0.8}] * 10
        report = check_historical_performance(old + recent, min_hook_score=0.45, min_samples=5)
        assert not report.warnings  # recent entries are good


# ── run_gate ──────────────────────────────────────────────────────────────────

class TestRunGate:
    def test_passes_with_all_good(self, tmp_path):
        script = _script(num_scenes=5)
        for i in range(5):
            (tmp_path / f"scene_{i:02d}.mp3").touch()
        history = [{"hook_score": 0.7}] * 8

        with patch("src.quality_gate.ffmpeg_utils.duration", return_value=30.0):
            report = run_gate(script, tmp_path, history)

        assert report.passed
        assert report.score > 0.5

    def test_raises_on_hard_failure(self, tmp_path):
        script = _script(num_scenes=5)
        # No audio files — hard failure

        with patch("src.quality_gate.ffmpeg_utils.duration", return_value=30.0):
            with pytest.raises(QualityGateError, match="(?i)quality gate"):
                run_gate(script, tmp_path, [])

    def test_summary_has_score(self, tmp_path):
        script = _script(num_scenes=5)
        for i in range(5):
            (tmp_path / f"scene_{i:02d}.mp3").touch()

        with patch("src.quality_gate.ffmpeg_utils.duration", return_value=30.0):
            report = run_gate(script, tmp_path, [])

        assert 0.0 <= report.score <= 1.0

    def test_config_overrides_thresholds(self, tmp_path):
        # Script with only 2 scenes — normally fails min_scenes=3
        script = _script(num_scenes=2)
        for i in range(2):
            (tmp_path / f"scene_{i:02d}.mp3").touch()
        history = [{"hook_score": 0.7}] * 8

        with patch("src.quality_gate.ffmpeg_utils.duration", return_value=60.0):
            report = run_gate(script, tmp_path, history, config={"min_scenes": 2})

        assert report.passed
