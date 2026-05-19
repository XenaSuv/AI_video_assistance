"""Tests for subtitle_generator — _srt_time(), _transcribe_scene(),
and generate_subtitles() with mocked Whisper/OpenAI calls."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Stub openai before importing subtitle_generator (it imports OpenAI at module level)
_openai_mock = MagicMock()
sys.modules.setdefault("openai", _openai_mock)

from src.script_generator import Scene, VideoScript
from src.subtitle_generator import _srt_time, _transcribe_scene, generate_subtitles

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scene(idx: int, heading: str = "", narration: str = "", duration_sec: int = 30) -> Scene:
    return Scene(
        idx=idx,
        heading=heading or f"Scene {idx}",
        narration=narration or f"Narration for scene {idx}.",
        visual_prompt="",
        duration_sec=duration_sec,
    )


def _make_script(scenes: list[Scene]) -> VideoScript:
    return VideoScript(
        title="Test Script",
        description="desc",
        tags=["ai"],
        hook="Hook text",
        scenes=scenes,
    )


def _fake_segment(start: float, end: float, text: str) -> MagicMock:
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg


def _make_transcription(segments: list[MagicMock]) -> MagicMock:
    resp = MagicMock()
    resp.segments = segments
    return resp


# ── _srt_time ─────────────────────────────────────────────────────────────────

class TestSrtTime:
    def test_zero_seconds(self):
        assert _srt_time(0.0) == "00:00:00,000"

    def test_simple_seconds(self):
        assert _srt_time(5.5) == "00:00:05,500"

    def test_minutes(self):
        assert _srt_time(90.0) == "00:01:30,000"

    def test_hours(self):
        assert _srt_time(3661.0) == "01:01:01,000"

    def test_millisecond_precision(self):
        assert _srt_time(1.123) == "00:00:01,123"

    def test_exactly_one_minute(self):
        assert _srt_time(60.0) == "00:01:00,000"

    def test_fractional_ms_rounding(self):
        # 1.4995 → rounds to 500ms (or 499ms depending on float precision)
        result = _srt_time(1.4995)
        # The function uses round(), so 0.4995 * 1000 ≈ 499.5 → 500
        assert result.startswith("00:00:01,")

    def test_long_video(self):
        # 2h 30m 15.250s
        result = _srt_time(2 * 3600 + 30 * 60 + 15.25)
        assert result == "02:30:15,250"


# ── _transcribe_scene ─────────────────────────────────────────────────────────

class TestTranscribeScene:
    def test_returns_list_of_segment_dicts(self, tmp_path):
        audio_path = tmp_path / "scene_00.mp3"
        audio_path.write_bytes(b"fake_audio")

        fake_resp = _make_transcription([
            _fake_segment(0.0, 2.5, " Hello world"),
            _fake_segment(2.5, 5.0, " How are you"),
        ])
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = fake_resp

        result = _transcribe_scene(mock_client, audio_path)

        assert len(result) == 2
        assert result[0] == {"start": 0.0, "end": 2.5, "text": "Hello world"}
        assert result[1] == {"start": 2.5, "end": 5.0, "text": "How are you"}

    def test_strips_whitespace_from_text(self, tmp_path):
        audio_path = tmp_path / "scene_00.mp3"
        audio_path.write_bytes(b"x")

        fake_resp = _make_transcription([_fake_segment(0.0, 1.0, "  padded  ")])
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = fake_resp

        result = _transcribe_scene(mock_client, audio_path)
        assert result[0]["text"] == "padded"

    def test_calls_whisper_with_correct_params(self, tmp_path):
        audio_path = tmp_path / "scene_00.mp3"
        audio_path.write_bytes(b"audio")

        fake_resp = _make_transcription([])
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = fake_resp

        _transcribe_scene(mock_client, audio_path)

        call_kwargs = mock_client.audio.transcriptions.create.call_args[1]
        assert call_kwargs["model"] == "whisper-1"
        assert call_kwargs["response_format"] == "verbose_json"
        assert "segment" in call_kwargs["timestamp_granularities"]


# ── generate_subtitles ────────────────────────────────────────────────────────

class TestGenerateSubtitles:
    def test_returns_cached_path_if_exists(self, tmp_path):
        srt_path = tmp_path / "subtitles.srt"
        srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
        script = _make_script([_make_scene(0)])

        with patch("src.subtitle_generator.make_openai_client") as mock_openai:
            result = generate_subtitles(script, tmp_path, srt_path)

        mock_openai.assert_not_called()
        assert result == srt_path

    def test_writes_srt_file_for_single_scene(self, tmp_path):
        scene = _make_scene(0, duration_sec=10)
        script = _make_script([scene])
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        mp3 = audio_dir / "scene_00.mp3"
        mp3.write_bytes(b"audio")
        srt_path = tmp_path / "subtitles.srt"

        fake_resp = _make_transcription([
            _fake_segment(0.0, 3.0, "This is the first line."),
            _fake_segment(3.0, 6.0, "And here is the second."),
        ])
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = fake_resp

        with patch("src.subtitle_generator.make_openai_client", return_value=mock_client):
            result = generate_subtitles(script, audio_dir, srt_path)

        assert result == srt_path
        assert srt_path.exists()
        content = srt_path.read_text(encoding="utf-8")
        assert "00:00:00,000 --> 00:00:03,000" in content
        assert "This is the first line." in content
        assert "00:00:03,000 --> 00:00:06,000" in content
        assert "And here is the second." in content

    def test_intro_duration_offsets_timestamps(self, tmp_path):
        scene = _make_scene(0, duration_sec=10)
        script = _make_script([scene])
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        mp3 = audio_dir / "scene_00.mp3"
        mp3.write_bytes(b"audio")
        srt_path = tmp_path / "subtitles.srt"

        fake_resp = _make_transcription([_fake_segment(0.0, 2.0, "Hello")])
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = fake_resp

        with patch("src.subtitle_generator.make_openai_client", return_value=mock_client):
            generate_subtitles(script, audio_dir, srt_path, intro_duration=5.0)

        content = srt_path.read_text(encoding="utf-8")
        # start should be 5.0 + 0.0 = 5.0 → "00:00:05,000"
        assert "00:00:05,000 --> 00:00:07,000" in content

    def test_skips_missing_audio_and_advances_offset(self, tmp_path):
        scene0 = _make_scene(0, duration_sec=10)
        scene1 = _make_scene(1, duration_sec=10)
        script = _make_script([scene0, scene1])
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        # Only create audio for scene 1, not scene 0
        mp3_1 = audio_dir / "scene_01.mp3"
        mp3_1.write_bytes(b"audio")
        srt_path = tmp_path / "subtitles.srt"

        fake_resp = _make_transcription([_fake_segment(0.0, 3.0, "Scene one text")])
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = fake_resp

        with patch("src.subtitle_generator.make_openai_client", return_value=mock_client):
            generate_subtitles(script, audio_dir, srt_path)

        content = srt_path.read_text(encoding="utf-8")
        # Scene 0 is skipped (10s offset), scene 1 starts at 10s
        assert "00:00:10,000 --> 00:00:13,000" in content
        assert "Scene one text" in content

    def test_does_not_write_file_when_no_segments(self, tmp_path):
        scene = _make_scene(0, duration_sec=10)
        script = _make_script([scene])
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        mp3 = audio_dir / "scene_00.mp3"
        mp3.write_bytes(b"audio")
        srt_path = tmp_path / "subtitles.srt"

        # Transcription returns empty segments
        fake_resp = _make_transcription([])
        mock_client = MagicMock()
        mock_client.audio.transcriptions.create.return_value = fake_resp

        with patch("src.subtitle_generator.make_openai_client", return_value=mock_client):
            result = generate_subtitles(script, audio_dir, srt_path)

        assert not srt_path.exists()
        assert result == srt_path  # still returns the path

    def test_multi_scene_srt_numbering(self, tmp_path):
        scenes = [_make_scene(i, duration_sec=10) for i in range(3)]
        script = _make_script(scenes)
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        for i in range(3):
            (audio_dir / f"scene_0{i}.mp3").write_bytes(b"audio")
        srt_path = tmp_path / "subtitles.srt"

        def fake_transcribe(client, path):
            idx = int(path.stem.split("_")[1])
            return [{"start": 0.0, "end": 2.0, "text": f"Text for scene {idx}"}]

        mock_client = MagicMock()

        with patch("src.subtitle_generator.make_openai_client", return_value=mock_client):
            with patch("src.subtitle_generator._transcribe_scene", side_effect=fake_transcribe):
                generate_subtitles(script, audio_dir, srt_path)

        lines = srt_path.read_text(encoding="utf-8").split("\n")
        # First entry should be "1"
        assert lines[0] == "1"
        # Second entry should be "2"
        entry_2_idx = lines.index("2")
        assert entry_2_idx > 0

    def test_no_audio_for_any_scene_produces_no_srt(self, tmp_path):
        scene = _make_scene(0, duration_sec=10)
        script = _make_script([scene])
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        # No MP3 files created
        srt_path = tmp_path / "subtitles.srt"

        mock_client = MagicMock()
        with patch("src.subtitle_generator.make_openai_client", return_value=mock_client):
            result = generate_subtitles(script, audio_dir, srt_path)

        mock_client.audio.transcriptions.create.assert_not_called()
        assert not srt_path.exists()
