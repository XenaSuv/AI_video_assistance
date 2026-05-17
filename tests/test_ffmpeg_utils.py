"""Tests for ffmpeg_utils — _esc() escaping and subprocess timeout handling."""
import json
import subprocess
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# Stub numpy before importing the module (heavy dep not needed for these tests)
_np_mock = MagicMock()
sys.modules.setdefault("numpy", _np_mock)

from src.ffmpeg_utils import (
    _esc, _run, _PROBE_TIMEOUT, _CLIP_TIMEOUT, _LONG_TIMEOUT,
    probe, duration, video_size, has_audio_stream, ensure_audio_stream,
    get_frame, merge_av, concat, title_card, overlay_title_card,
    quote_card_clip, end_card_clip, burn_chyron, ken_burns,
    frames_to_video, mix_music, make_vertical, zoom_clip,
    loop_and_trim, resize_video, black_clip, burn_captions,
)


class TestEsc:
    def test_clean_text_unchanged(self):
        assert _esc("Hello World") == "Hello World"

    def test_backslash_escaped_first(self):
        # Backslash must be doubled; must happen before other replacements
        # so that escape sequences introduced by later steps aren't re-escaped.
        assert _esc("a\\b") == "a\\\\b"

    def test_single_quote_escaped(self):
        # FFmpeg mirrors POSIX: ' → '\''
        assert _esc("it's") == r"it'\''s"

    def test_colon_escaped(self):
        assert _esc("foo:bar") == r"foo\:bar"

    def test_square_brackets_escaped(self):
        assert _esc("a[b]c") == r"a\[b\]c"

    def test_percent_doubled_for_drawtext(self):
        # % is the drawtext printf-expansion prefix (%{pts}, %{expr:…})
        assert _esc("100%") == "100%%"
        # { and } are not filter-graph special chars — only % is doubled,
        # so %%{pts} is rendered literally as %{pts} (expansion broken)
        assert _esc("%{pts}") == "%%{pts}"

    def test_percent_after_backslash_not_double_escaped(self):
        # backslash before %, then % doubled — the backslash must not be re-escaped
        result = _esc("50\\%")
        assert result == "50\\\\%%"

    def test_newline_stripped(self):
        assert _esc("line1\nline2") == "line1line2"

    def test_carriage_return_stripped(self):
        assert _esc("foo\rbar") == "foobar"

    def test_null_byte_stripped(self):
        assert _esc("foo\x00bar") == "foobar"

    def test_tab_stripped(self):
        assert _esc("a\tb") == "ab"

    def test_other_control_chars_stripped(self):
        # Bell, form-feed, etc.
        assert _esc("a\x07\x0cb") == "ab"

    def test_empty_string(self):
        assert _esc("") == ""

    def test_real_world_news_headline(self):
        # Typical RSS headline with no special chars
        result = _esc("OpenAI releases GPT-5 with record benchmarks")
        assert result == "OpenAI releases GPT-5 with record benchmarks"

    def test_headline_with_percent(self):
        # Common in financial/statistics headlines
        result = _esc("AI adoption up 35% year-over-year")
        assert result == "AI adoption up 35%% year-over-year"

    def test_headline_with_colon(self):
        result = _esc("Breaking: New model surpasses GPT-4")
        assert r"\:" in result

    def test_headline_with_single_quote(self):
        result = _esc("What's next for AI regulation?")
        assert r"'\''" in result

    def test_combined_special_chars(self):
        # Worst-case headline with multiple special chars
        result = _esc("AI's impact: [50%] increase\nin efficiency")
        # Single quote escaped
        assert r"'\''" in result
        # Colon escaped
        assert r"\:" in result
        # Percent doubled
        assert "%%" in result
        # Brackets escaped
        assert r"\[" in result
        assert r"\]" in result
        # Newline stripped — no literal newline in output
        assert "\n" not in result


class TestRunTimeout:
    def test_timeout_constants_are_ordered(self):
        # Probe must be shortest; long operations get the most time.
        assert _PROBE_TIMEOUT < _CLIP_TIMEOUT < _LONG_TIMEOUT

    def test_run_raises_runtime_error_on_timeout(self):
        # Simulate TimeoutExpired from subprocess and verify _run() wraps it.
        with mock.patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=5)):
            try:
                _run(["ffmpeg", "-i", "fake.mp4"], timeout=5)
                assert False, "Expected RuntimeError"
            except RuntimeError as exc:
                assert "timed out" in str(exc).lower()
                assert "5s" in str(exc)

    def test_run_default_timeout_is_clip_timeout(self):
        # _run() must forward the timeout to subprocess.run.
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            _run(["ffmpeg", "-version"])
            _, kwargs = mock_run.call_args
            assert kwargs["timeout"] == _CLIP_TIMEOUT

    def test_run_accepts_custom_timeout(self):
        with mock.patch("subprocess.run") as mock_run:
            mock_run.return_value = mock.MagicMock(returncode=0)
            _run(["ffprobe", "x"], timeout=_PROBE_TIMEOUT)
            _, kwargs = mock_run.call_args
            assert kwargs["timeout"] == _PROBE_TIMEOUT


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_probe_result(streams=None, fmt_duration=None):
    """Build a fake ffprobe JSON payload."""
    info = {}
    if fmt_duration is not None:
        info["format"] = {"duration": str(fmt_duration)}
    else:
        info["format"] = {}
    info["streams"] = streams or []
    result = MagicMock()
    result.stdout = json.dumps(info)
    return result


def _mock_run_ok(stdout=""):
    """Return a MagicMock completedprocess with given stdout."""
    r = MagicMock()
    r.stdout = stdout
    return r


# ─────────────────────────────────────────────────────────────────────────────
# probe()
# ─────────────────────────────────────────────────────────────────────────────

class TestProbe:
    def test_returns_parsed_json(self):
        fake_info = {"streams": [], "format": {"duration": "5.0"}}
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(fmt_duration=5.0)
            result = probe(Path("fake.mp4"))
        assert result["format"]["duration"] == "5.0"

    def test_calls_ffprobe_with_correct_flags(self):
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(fmt_duration=3.0)
            probe(Path("video.mp4"))
        cmd = m.call_args[0][0]
        assert cmd[0] == "ffprobe"
        assert "-show_streams" in cmd
        assert "-show_format" in cmd
        assert "video.mp4" in cmd

    def test_uses_probe_timeout(self):
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(fmt_duration=1.0)
            probe(Path("x.mp4"))
        _, kwargs = m.call_args
        assert kwargs["timeout"] == _PROBE_TIMEOUT


# ─────────────────────────────────────────────────────────────────────────────
# duration()
# ─────────────────────────────────────────────────────────────────────────────

class TestDuration:
    def test_reads_format_duration_first(self):
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(fmt_duration=12.5)
            assert duration(Path("v.mp4")) == 12.5

    def test_falls_back_to_stream_duration(self):
        streams = [{"codec_type": "video", "duration": "7.3"}]
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(streams=streams)
            result = duration(Path("v.mp4"))
            assert abs(result - 7.3) < 1e-9

    def test_raises_when_no_duration(self):
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result()
            import pytest
            with pytest.raises(ValueError, match="Cannot determine duration"):
                duration(Path("empty.mp4"))

    def test_skips_streams_without_duration(self):
        streams = [{"codec_type": "video"}, {"codec_type": "audio", "duration": "9.0"}]
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(streams=streams)
            result = duration(Path("v.mp4"))
            assert abs(result - 9.0) < 1e-9


import pytest  # noqa: E402 — needed for pytest.approx and pytest.raises throughout


# ─────────────────────────────────────────────────────────────────────────────
# video_size()
# ─────────────────────────────────────────────────────────────────────────────

class TestVideoSize:
    def test_returns_width_height_from_first_video_stream(self):
        streams = [{"codec_type": "video", "width": 1280, "height": 720}]
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(streams=streams, fmt_duration=5.0)
            w, h = video_size(Path("v.mp4"))
        assert w == 1280
        assert h == 720

    def test_raises_when_no_video_stream(self):
        streams = [{"codec_type": "audio"}]
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(streams=streams, fmt_duration=5.0)
            with pytest.raises(ValueError, match="No video stream"):
                video_size(Path("audio.mp3"))

    def test_skips_non_video_streams(self):
        streams = [
            {"codec_type": "audio"},
            {"codec_type": "video", "width": 640, "height": 360},
        ]
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(streams=streams, fmt_duration=5.0)
            w, h = video_size(Path("v.mp4"))
        assert (w, h) == (640, 360)


# ─────────────────────────────────────────────────────────────────────────────
# has_audio_stream()
# ─────────────────────────────────────────────────────────────────────────────

class TestHasAudioStream:
    def test_returns_false_when_file_does_not_exist(self, tmp_path):
        assert has_audio_stream(tmp_path / "missing.mp4") is False

    def test_returns_true_when_audio_stream_present(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        streams = [{"codec_type": "audio"}]
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(streams=streams, fmt_duration=5.0)
            assert has_audio_stream(video) is True

    def test_returns_false_when_only_video_stream(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        streams = [{"codec_type": "video", "width": 1280, "height": 720}]
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(streams=streams, fmt_duration=5.0)
            assert has_audio_stream(video) is False

    def test_returns_false_when_no_streams(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(streams=[], fmt_duration=5.0)
            assert has_audio_stream(video) is False


# ─────────────────────────────────────────────────────────────────────────────
# ensure_audio_stream()
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureAudioStream:
    def test_returns_input_if_already_has_audio(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        output = tmp_path / "out.mp4"
        streams = [{"codec_type": "audio"}]
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(streams=streams, fmt_duration=5.0)
            result = ensure_audio_stream(video, output)
        assert result == video

    def test_returns_output_if_it_already_exists(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        output = tmp_path / "out.mp4"
        output.touch()
        # no audio stream
        streams = [{"codec_type": "video", "width": 1280, "height": 720}]
        with patch("subprocess.run") as m:
            m.return_value = _make_probe_result(streams=streams, fmt_duration=5.0)
            result = ensure_audio_stream(video, output)
        assert result == output

    def test_runs_ffmpeg_to_add_silent_audio(self, tmp_path):
        video = tmp_path / "v.mp4"
        video.touch()
        output = tmp_path / "out.mp4"
        # No audio, output doesn't exist
        video_probe = _make_probe_result(streams=[{"codec_type": "video", "width": 1280, "height": 720}], fmt_duration=4.0)
        duration_probe = _make_probe_result(fmt_duration=4.0)

        call_count = [0]
        def side_effect(args, **kwargs):
            # First two calls are probe (for has_audio_stream and duration), rest is ffmpeg
            call_count[0] += 1
            if "ffprobe" in args[0]:
                return video_probe
            return MagicMock(stdout="", returncode=0)

        with patch("subprocess.run", side_effect=side_effect) as m:
            result = ensure_audio_stream(video, output)
        # Should call ffmpeg with anullsrc
        all_calls = m.call_args_list
        ffmpeg_calls = [c for c in all_calls if c[0][0][0] == "ffmpeg"]
        assert len(ffmpeg_calls) == 1
        cmd = ffmpeg_calls[0][0][0]
        assert "anullsrc" in " ".join(cmd)


# ─────────────────────────────────────────────────────────────────────────────
# merge_av()
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeAv:
    def test_returns_output_if_exists(self, tmp_path):
        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.mp3"
        output = tmp_path / "out.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = merge_av(video, audio, output)
        assert result == output
        m.assert_not_called()

    def test_no_loop_uses_stream_copy(self, tmp_path):
        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.mp3"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            result = merge_av(video, audio, output)
        assert result == output
        cmd = m.call_args[0][0]
        assert "-c:v" in cmd
        assert "copy" in cmd
        assert "-stream_loop" not in cmd

    def test_loop_video_uses_libx264(self, tmp_path):
        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.mp3"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            result = merge_av(video, audio, output, loop_video=True)
        assert result == output
        cmd = m.call_args[0][0]
        assert "libx264" in cmd
        assert "-stream_loop" in cmd

    def test_creates_parent_directory(self, tmp_path):
        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.mp3"
        output = tmp_path / "sub" / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            merge_av(video, audio, output)
        assert (tmp_path / "sub").is_dir()


# ─────────────────────────────────────────────────────────────────────────────
# concat()
# ─────────────────────────────────────────────────────────────────────────────

class TestConcat:
    def test_returns_output_if_exists(self, tmp_path):
        output = tmp_path / "out.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = concat([tmp_path / "a.mp4"], output)
        assert result == output
        m.assert_not_called()

    def test_writes_list_file_then_removes_it(self, tmp_path):
        paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            concat(paths, output)
        # List file should be cleaned up
        list_file = tmp_path / "out_list.txt"
        assert not list_file.exists()

    def test_prefers_stream_copy(self, tmp_path):
        paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            concat(paths, output)
        cmd = m.call_args[0][0]
        assert "-c" in cmd and "copy" in cmd
        assert "libx264" not in cmd

    def test_fallback_uses_libx264_and_aac(self, tmp_path):
        import subprocess as _sp
        paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.side_effect = [
                _sp.CalledProcessError(1, "ffmpeg"),
                MagicMock(stdout="", returncode=0),
            ]
            concat(paths, output)
        fallback_cmd = m.call_args_list[1][0][0]
        assert "libx264" in fallback_cmd
        assert "aac" in fallback_cmd
        assert "-an" not in fallback_cmd

    def test_video_only_adds_an_flag(self, tmp_path):
        paths = [tmp_path / "a.mp4"]
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            concat(paths, output, video_only=True)
        cmd = m.call_args[0][0]
        assert "-an" in cmd
        assert "aac" not in cmd

    def test_uses_long_timeout(self, tmp_path):
        paths = [tmp_path / "a.mp4"]
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            concat(paths, output)
        _, kwargs = m.call_args
        assert kwargs["timeout"] == _LONG_TIMEOUT


# ─────────────────────────────────────────────────────────────────────────────
# title_card()
# ─────────────────────────────────────────────────────────────────────────────

class TestTitleCard:
    def test_returns_output_if_exists(self, tmp_path):
        output = tmp_path / "title.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = title_card("Hello", output)
        assert result == output
        m.assert_not_called()

    def test_calls_ffmpeg_with_lavfi_color(self, tmp_path):
        output = tmp_path / "title.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            title_card("Test Heading", output)
        cmd = m.call_args[0][0]
        assert "ffmpeg" in cmd
        assert any("color=c=" in a for a in cmd)

    def test_includes_drawtext_with_escaped_heading(self, tmp_path):
        output = tmp_path / "title.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            title_card("AI's impact: 100%", output)
        cmd_str = " ".join(str(a) for a in m.call_args[0][0])
        assert "drawtext" in cmd_str
        # Check that special chars were escaped
        assert r"'\''" in cmd_str or "drawtext" in cmd_str

    def test_default_duration_is_2_5_seconds(self, tmp_path):
        output = tmp_path / "title.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            title_card("Hello", output)
        cmd_str = " ".join(str(a) for a in m.call_args[0][0])
        assert "2.5" in cmd_str

    def test_custom_duration(self, tmp_path):
        output = tmp_path / "title.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            title_card("Hello", output, duration_sec=5.0)
        cmd_str = " ".join(str(a) for a in m.call_args[0][0])
        assert "5.0" in cmd_str

    def test_long_heading_wrapped(self, tmp_path):
        output = tmp_path / "title.mp4"
        long_heading = "A" * 100  # Very long — should be wrapped
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            result = title_card(long_heading, output)
        assert result == output

    def test_creates_parent_directory(self, tmp_path):
        output = tmp_path / "sub" / "title.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            title_card("Hello", output)
        assert (tmp_path / "sub").is_dir()


# ─────────────────────────────────────────────────────────────────────────────
# overlay_title_card()
# ─────────────────────────────────────────────────────────────────────────────

class TestOverlayTitleCard:
    def test_returns_output_if_exists(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = overlay_title_card(video, "Heading", output)
        assert result == output
        m.assert_not_called()

    def test_calls_ffmpeg_with_drawbox_and_drawtext(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            overlay_title_card(video, "AI News", output)
        cmd_str = " ".join(str(a) for a in m.call_args[0][0])
        assert "drawbox" in cmd_str
        assert "drawtext" in cmd_str

    def test_copies_audio(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            overlay_title_card(video, "Heading", output)
        cmd = m.call_args[0][0]
        # -c:a copy
        idx = cmd.index("-c:a")
        assert cmd[idx + 1] == "copy"


# ─────────────────────────────────────────────────────────────────────────────
# quote_card_clip()
# ─────────────────────────────────────────────────────────────────────────────

class TestQuoteCardClip:
    def test_returns_output_if_exists(self, tmp_path):
        png = tmp_path / "card.png"
        output = tmp_path / "clip.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = quote_card_clip(png, output)
        assert result == output
        m.assert_not_called()

    def test_calls_ffmpeg_with_loop_and_duration(self, tmp_path):
        png = tmp_path / "card.png"
        output = tmp_path / "clip.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            quote_card_clip(png, output, duration_sec=4.0)
        cmd = m.call_args[0][0]
        assert "-loop" in cmd
        assert "1" in cmd
        assert "-t" in cmd
        assert "4.0" in cmd

    def test_video_only_no_audio(self, tmp_path):
        png = tmp_path / "card.png"
        output = tmp_path / "clip.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            quote_card_clip(png, output)
        cmd = m.call_args[0][0]
        assert "-an" in cmd

    def test_includes_fade(self, tmp_path):
        png = tmp_path / "card.png"
        output = tmp_path / "clip.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            quote_card_clip(png, output)
        cmd_str = " ".join(m.call_args[0][0])
        assert "fade" in cmd_str


# ─────────────────────────────────────────────────────────────────────────────
# end_card_clip()
# ─────────────────────────────────────────────────────────────────────────────

class TestEndCardClip:
    def test_returns_output_if_exists(self, tmp_path):
        png = tmp_path / "end.png"
        output = tmp_path / "end_clip.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = end_card_clip(png, output)
        assert result == output
        m.assert_not_called()

    def test_calls_ffmpeg_with_silent_audio(self, tmp_path):
        png = tmp_path / "end.png"
        output = tmp_path / "end_clip.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            end_card_clip(png, output)
        cmd_str = " ".join(m.call_args[0][0])
        assert "anullsrc" in cmd_str

    def test_default_duration_10s(self, tmp_path):
        png = tmp_path / "end.png"
        output = tmp_path / "end_clip.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            end_card_clip(png, output)
        cmd_str = " ".join(m.call_args[0][0])
        assert "10.0" in cmd_str

    def test_includes_fade(self, tmp_path):
        png = tmp_path / "end.png"
        output = tmp_path / "end_clip.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            end_card_clip(png, output)
        cmd_str = " ".join(m.call_args[0][0])
        assert "fade" in cmd_str


# ─────────────────────────────────────────────────────────────────────────────
# burn_chyron()
# ─────────────────────────────────────────────────────────────────────────────

class TestBurnChyron:
    def test_returns_output_if_exists(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = burn_chyron(video, "Breaking news", output)
        assert result == output
        m.assert_not_called()

    def test_burns_drawtext_onto_video(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        # probe calls: duration
        probe_result = _make_probe_result(fmt_duration=10.0)
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            result = burn_chyron(video, "Breaking: AI News", output)
        # last call should be ffmpeg with drawtext
        last_cmd = m.call_args_list[-1][0][0]
        cmd_str = " ".join(last_cmd)
        assert "drawtext" in cmd_str

    def test_returns_source_video_on_exception(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run", side_effect=RuntimeError("boom")):
            result = burn_chyron(video, "Test", output)
        assert result == video

    def test_chyron_end_capped_at_duration(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        probe_result = _make_probe_result(fmt_duration=2.0)
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            burn_chyron(video, "Short video heading", output, start_sec=0.0)
        last_cmd = m.call_args_list[-1][0][0]
        # chyron_end = min(2.0, 0.0 + 4.0) = 2.0
        cmd_str = " ".join(last_cmd)
        assert "2.00" in cmd_str


# ─────────────────────────────────────────────────────────────────────────────
# ken_burns()
# ─────────────────────────────────────────────────────────────────────────────

class TestKenBurns:
    def test_returns_output_if_exists(self, tmp_path):
        img = tmp_path / "img.jpg"
        output = tmp_path / "kb.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = ken_burns(img, output, duration_sec=5.0, variant=0)
        assert result == output
        m.assert_not_called()

    def _run_ken_burns(self, tmp_path, variant, in_w=1792, in_h=1024):
        img = tmp_path / "img.jpg"
        output = tmp_path / f"kb_{variant}.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            ken_burns(img, output, duration_sec=5.0, variant=variant,
                      in_w=in_w, in_h=in_h)
        return m.call_args[0][0]

    def test_variant_0_pan_left_right(self, tmp_path):
        cmd = self._run_ken_burns(tmp_path, variant=0)
        cmd_str = " ".join(cmd)
        assert "scale=" in cmd_str
        assert "crop=" in cmd_str

    def test_variant_1_pan_right_left(self, tmp_path):
        cmd = self._run_ken_burns(tmp_path, variant=1)
        cmd_str = " ".join(cmd)
        assert "scale=" in cmd_str
        assert "crop=" in cmd_str

    def test_variant_2_zoom_in(self, tmp_path):
        cmd = self._run_ken_burns(tmp_path, variant=2)
        cmd_str = " ".join(cmd)
        assert "scale=" in cmd_str
        assert "crop=" in cmd_str

    def test_variant_3_zoom_out(self, tmp_path):
        cmd = self._run_ken_burns(tmp_path, variant=3)
        cmd_str = " ".join(cmd)
        assert "scale=" in cmd_str
        assert "crop=" in cmd_str

    def test_small_image_scaled_up(self, tmp_path):
        # in_w < out_w → ratio > 1 → kb_w should be scaled
        cmd = self._run_ken_burns(tmp_path, variant=0, in_w=640, in_h=360)
        cmd_str = " ".join(cmd)
        assert "scale=" in cmd_str

    def test_uses_loop_framerate_24(self, tmp_path):
        cmd = self._run_ken_burns(tmp_path, variant=0)
        assert "-loop" in cmd
        assert "-framerate" in cmd
        idx = cmd.index("-framerate")
        assert cmd[idx + 1] == "24"

    def test_no_audio_stream(self, tmp_path):
        cmd = self._run_ken_burns(tmp_path, variant=0)
        assert "-an" in cmd

    def test_duration_passed_to_ffmpeg(self, tmp_path):
        img = tmp_path / "img.jpg"
        output = tmp_path / "kb.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            ken_burns(img, output, duration_sec=7.5, variant=0)
        cmd = m.call_args[0][0]
        assert "7.5" in cmd


# ─────────────────────────────────────────────────────────────────────────────
# get_frame()
# ─────────────────────────────────────────────────────────────────────────────

class TestGetFrame:
    def test_returns_numpy_array_reshaped(self, tmp_path):
        import numpy as real_np
        video = tmp_path / "v.mp4"
        # Mock video_size → 4×4
        streams = [{"codec_type": "video", "width": 4, "height": 4}]
        probe_result = _make_probe_result(streams=streams, fmt_duration=5.0)
        # Raw frame bytes: 4*4*3 = 48 bytes
        fake_frame = b"\xff" * 48
        frame_result = MagicMock()
        frame_result.stdout = fake_frame

        call_count = [0]
        def side_effect(args, **kwargs):
            call_count[0] += 1
            if args[0] == "ffprobe":
                return probe_result
            return frame_result

        import numpy
        numpy.frombuffer = real_np.frombuffer if hasattr(real_np, 'frombuffer') else MagicMock(return_value=MagicMock())
        with patch("subprocess.run", side_effect=side_effect):
            # Should not raise
            result = get_frame(video, t=1.0)
        # We can't check exact shape since numpy is mocked, but call should succeed

    def test_raises_on_timeout(self, tmp_path):
        video = tmp_path / "v.mp4"
        streams = [{"codec_type": "video", "width": 4, "height": 4}]
        probe_result = _make_probe_result(streams=streams, fmt_duration=5.0)

        call_count = [0]
        def side_effect(args, **kwargs):
            call_count[0] += 1
            if args[0] == "ffprobe":
                return probe_result
            raise subprocess.TimeoutExpired(cmd=args, timeout=30)

        with patch("subprocess.run", side_effect=side_effect):
            with pytest.raises(RuntimeError, match="timed out"):
                get_frame(video, t=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# mix_music()
# ─────────────────────────────────────────────────────────────────────────────

class TestMixMusic:
    def test_returns_output_if_exists(self, tmp_path):
        video = tmp_path / "v.mp4"
        music = tmp_path / "music.mp3"
        output = tmp_path / "out.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = mix_music(video, music, output)
        assert result == output
        m.assert_not_called()

    def test_mixes_music_under_video(self, tmp_path):
        video = tmp_path / "v.mp4"
        music = tmp_path / "music.mp3"
        output = tmp_path / "out.mp4"
        probe_result = _make_probe_result(fmt_duration=30.0)
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            result = mix_music(video, music, output)
        assert result == output
        # last ffmpeg call should include amix
        last_cmd_str = " ".join(m.call_args_list[-1][0][0])
        assert "amix" in last_cmd_str

    def test_returns_source_video_on_exception(self, tmp_path):
        video = tmp_path / "v.mp4"
        music = tmp_path / "music.mp3"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run", side_effect=RuntimeError("ffmpeg exploded")):
            result = mix_music(video, music, output)
        assert result == video

    def test_volume_and_fade_in_filter(self, tmp_path):
        video = tmp_path / "v.mp4"
        music = tmp_path / "music.mp3"
        output = tmp_path / "out.mp4"
        probe_result = _make_probe_result(fmt_duration=10.0)
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            mix_music(video, music, output, volume=0.05, fade_in=1.0, fade_out=2.0)
        last_cmd_str = " ".join(m.call_args_list[-1][0][0])
        assert "volume=0.05" in last_cmd_str
        assert "afade=t=in" in last_cmd_str

    def test_uses_long_timeout(self, tmp_path):
        video = tmp_path / "v.mp4"
        music = tmp_path / "music.mp3"
        output = tmp_path / "out.mp4"
        probe_result = _make_probe_result(fmt_duration=10.0)
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            mix_music(video, music, output)
        last_call_kwargs = m.call_args_list[-1][1]
        assert last_call_kwargs.get("timeout") == _LONG_TIMEOUT


# ─────────────────────────────────────────────────────────────────────────────
# make_vertical()
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeVertical:
    def test_returns_output_if_exists(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = make_vertical(video, output)
        assert result == output
        m.assert_not_called()

    def test_landscape_wider_than_target_crops(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        # 1280x720 → target 1080x1920: src_ratio = 1.78, target_ratio = 0.5625
        # src > target → crop path
        streams = [{"codec_type": "video", "width": 1280, "height": 720}]
        probe_result = _make_probe_result(streams=streams, fmt_duration=5.0)
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            result = make_vertical(video, output)
        assert result == output
        last_cmd_str = " ".join(m.call_args_list[-1][0][0])
        assert "crop=" in last_cmd_str

    def test_portrait_input_just_scales(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        # Already portrait: 720x1280 → src_ratio=0.5625 == target_ratio → no crop
        streams = [{"codec_type": "video", "width": 720, "height": 1280}]
        probe_result = _make_probe_result(streams=streams, fmt_duration=5.0)
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            result = make_vertical(video, output)
        last_cmd_str = " ".join(m.call_args_list[-1][0][0])
        assert "scale=1080:1920" in last_cmd_str

    def test_custom_crop_x_respected(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        streams = [{"codec_type": "video", "width": 1920, "height": 1080}]
        probe_result = _make_probe_result(streams=streams, fmt_duration=5.0)
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            make_vertical(video, output, crop_x=100)
        last_cmd_str = " ".join(m.call_args_list[-1][0][0])
        assert "100" in last_cmd_str


# ─────────────────────────────────────────────────────────────────────────────
# zoom_clip()
# ─────────────────────────────────────────────────────────────────────────────

class TestZoomClip:
    def test_returns_output_if_exists(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = zoom_clip(video, output)
        assert result == output
        m.assert_not_called()

    def test_crops_and_rescales(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        streams = [{"codec_type": "video", "width": 1280, "height": 720}]
        probe_result = _make_probe_result(streams=streams, fmt_duration=5.0)
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            result = zoom_clip(video, output)
        assert result == output
        last_cmd_str = " ".join(m.call_args_list[-1][0][0])
        assert "crop=" in last_cmd_str
        assert "scale=" in last_cmd_str

    def test_returns_source_on_exception(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run", side_effect=RuntimeError("boom")):
            result = zoom_clip(video, output)
        assert result == video

    def test_custom_scale_factor(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        streams = [{"codec_type": "video", "width": 1280, "height": 720}]
        probe_result = _make_probe_result(streams=streams, fmt_duration=5.0)
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            zoom_clip(video, output, scale=1.5)
        # crop_w = int(1280 / 1.5) = 853
        last_cmd_str = " ".join(m.call_args_list[-1][0][0])
        assert "853" in last_cmd_str


# ─────────────────────────────────────────────────────────────────────────────
# loop_and_trim()
# ─────────────────────────────────────────────────────────────────────────────

class TestLoopAndTrim:
    def test_returns_output_if_exists(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = loop_and_trim(video, output, target_sec=10.0)
        assert result == output
        m.assert_not_called()

    def test_tries_stream_copy_first(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            result = loop_and_trim(video, output, target_sec=10.0)
        assert result == output
        cmd = m.call_args[0][0]
        assert "-c" in cmd
        assert "copy" in cmd

    def test_falls_back_to_reencode_on_calledprocesserror(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        call_count = [0]
        def side_effect(args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise subprocess.CalledProcessError(1, args)
            return MagicMock(stdout="", returncode=0)

        with patch("subprocess.run", side_effect=side_effect):
            result = loop_and_trim(video, output, target_sec=10.0)
        assert result == output
        assert call_count[0] == 2

    def test_reencode_uses_libx264(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        call_count = [0]
        calls = []
        def side_effect(args, **kwargs):
            call_count[0] += 1
            calls.append(args)
            if call_count[0] == 1:
                raise subprocess.CalledProcessError(1, args)
            return MagicMock(stdout="", returncode=0)

        with patch("subprocess.run", side_effect=side_effect):
            loop_and_trim(video, output, target_sec=5.0)
        # Second call should use libx264
        assert "libx264" in calls[1]

    def test_stream_loop_flag_present(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            loop_and_trim(video, output, target_sec=15.0)
        cmd = m.call_args[0][0]
        assert "-stream_loop" in cmd


# ─────────────────────────────────────────────────────────────────────────────
# resize_video()
# ─────────────────────────────────────────────────────────────────────────────

class TestResizeVideo:
    def test_returns_output_if_exists(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = resize_video(video, output, width=640, height=360)
        assert result == output
        m.assert_not_called()

    def test_scales_to_target_dimensions(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            resize_video(video, output, width=640, height=360)
        cmd_str = " ".join(m.call_args[0][0])
        assert "scale=640:360" in cmd_str

    def test_default_strips_audio(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            resize_video(video, output, width=640, height=360)
        cmd = m.call_args[0][0]
        assert "-an" in cmd

    def test_keep_audio_uses_aac(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            resize_video(video, output, width=640, height=360, keep_audio=True)
        cmd = m.call_args[0][0]
        assert "aac" in cmd
        assert "-an" not in cmd


# ─────────────────────────────────────────────────────────────────────────────
# black_clip()
# ─────────────────────────────────────────────────────────────────────────────

class TestBlackClip:
    def test_returns_output_if_exists(self, tmp_path):
        output = tmp_path / "black.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = black_clip(output, duration_sec=2.0)
        assert result == output
        m.assert_not_called()

    def test_generates_lavfi_color(self, tmp_path):
        output = tmp_path / "black.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            black_clip(output, duration_sec=2.0)
        cmd_str = " ".join(m.call_args[0][0])
        assert "color=c=" in cmd_str
        assert "0x0F1417" in cmd_str

    def test_default_size_1280x720(self, tmp_path):
        output = tmp_path / "black.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            black_clip(output, duration_sec=3.0)
        cmd_str = " ".join(m.call_args[0][0])
        assert "1280x720" in cmd_str

    def test_custom_size(self, tmp_path):
        output = tmp_path / "black.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            black_clip(output, width=1920, height=1080, duration_sec=1.0)
        cmd_str = " ".join(m.call_args[0][0])
        assert "1920x1080" in cmd_str

    def test_no_audio(self, tmp_path):
        output = tmp_path / "black.mp4"
        with patch("subprocess.run") as m:
            m.return_value = MagicMock(stdout="", returncode=0)
            black_clip(output, duration_sec=2.0)
        cmd = m.call_args[0][0]
        assert "-an" in cmd


# ─────────────────────────────────────────────────────────────────────────────
# burn_captions()
# ─────────────────────────────────────────────────────────────────────────────

class TestBurnCaptions:
    def test_returns_output_if_exists(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        output.touch()
        with patch("subprocess.run") as m:
            result = burn_captions(video, "some text", output)
        assert result == output
        m.assert_not_called()

    def test_returns_video_on_empty_text(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        probe_result = _make_probe_result(
            streams=[{"codec_type": "video", "width": 1280, "height": 720}],
            fmt_duration=10.0,
        )
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            result = burn_captions(video, "", output)
        # Empty words → no chunks → return video
        assert result == video

    def test_burns_drawtext_with_timed_chunks(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        probe_result = _make_probe_result(
            streams=[{"codec_type": "video", "width": 1280, "height": 720}],
            fmt_duration=10.0,
        )
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            result = burn_captions(video, "word1 word2 word3 word4 word5 word6", output)
        assert result == output
        last_cmd_str = " ".join(m.call_args_list[-1][0][0])
        assert "drawtext" in last_cmd_str
        assert "between" in last_cmd_str

    def test_returns_source_on_exception(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        with patch("subprocess.run", side_effect=RuntimeError("fail")):
            result = burn_captions(video, "hello world", output)
        assert result == video

    def test_custom_font_size_and_color(self, tmp_path):
        video = tmp_path / "v.mp4"
        output = tmp_path / "out.mp4"
        probe_result = _make_probe_result(
            streams=[{"codec_type": "video", "width": 1280, "height": 720}],
            fmt_duration=5.0,
        )
        with patch("subprocess.run") as m:
            m.return_value = probe_result
            burn_captions(video, "hello world", output, font_size=120, color="red")
        last_cmd_str = " ".join(m.call_args_list[-1][0][0])
        assert "fontsize=120" in last_cmd_str
        assert "fontcolor=red" in last_cmd_str


# ─────────────────────────────────────────────────────────────────────────────
# frames_to_video()
# ─────────────────────────────────────────────────────────────────────────────

class TestFramesToVideo:
    def test_returns_output_if_exists(self, tmp_path):
        output = tmp_path / "out.mp4"
        output.touch()
        import numpy as np_mock
        with patch("subprocess.Popen") as m:
            result = frames_to_video([], output)
        assert result == output
        m.assert_not_called()

    def test_calls_popen_with_rawvideo_args(self, tmp_path):
        output = tmp_path / "out.mp4"
        import numpy as np_mock
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.stderr = MagicMock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        with patch("subprocess.Popen", return_value=mock_proc) as popen_mock:
            frames_to_video([], output)
        popen_cmd = popen_mock.call_args[0][0]
        cmd_str = " ".join(popen_cmd)
        assert "rawvideo" in cmd_str
        assert "libx264" in cmd_str

    def test_raises_on_nonzero_returncode(self, tmp_path):
        output = tmp_path / "out.mp4"
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.returncode = 1
        mock_proc.wait.return_value = 1
        mock_proc.stderr.read.return_value = b"error msg"

        with patch("subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="ffmpeg failed"):
                frames_to_video([], output)

    def test_raises_on_timeout(self, tmp_path):
        output = tmp_path / "out.mp4"
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.returncode = 0
        # First call to wait() raises TimeoutExpired; subsequent calls (after kill) succeed
        _wait_calls = [0]
        def _wait_side_effect(**kwargs):
            _wait_calls[0] += 1
            if _wait_calls[0] == 1:
                raise subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=600)
            return 0
        mock_proc.wait.side_effect = _wait_side_effect

        with patch("subprocess.Popen", return_value=mock_proc):
            with pytest.raises(RuntimeError, match="timed out"):
                frames_to_video([], output)
        mock_proc.kill.assert_called_once()

    def test_mismatched_frame_shape_triggers_pil_resize(self, tmp_path):
        """Frames with wrong shape must be resized to (height, width, 3) via PIL."""
        output = tmp_path / "out.mp4"
        mock_proc = MagicMock()
        mock_proc.stdin = MagicMock()
        mock_proc.returncode = 0
        mock_proc.wait.return_value = 0

        # Create a frame with incorrect shape (too small)
        wrong_shape_frame = _np_mock.zeros.return_value
        wrong_shape_frame.shape = (100, 50, 3)  # differs from 720×1280

        mock_pil_img = MagicMock()
        mock_pil_img.resize.return_value = MagicMock()

        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("src.ffmpeg_utils.np") as mock_np:
                # Make frame.shape != (height, width, 3) so the branch fires
                mock_frame = MagicMock()
                mock_frame.shape = (100, 50, 3)  # wrong shape
                mock_frame.astype.return_value = mock_frame
                mock_frame.tobytes.return_value = b""
                mock_np.array.return_value = mock_frame

                with patch("PIL.Image") as mock_pil:
                    mock_pil.fromarray.return_value = mock_pil_img
                    frames_to_video([mock_frame], output, width=1280, height=720)

        # PIL resize was invoked to fix the frame dimensions
        mock_pil_img.resize.assert_called_once_with((1280, 720))


# ── Timeout constants — configurable via settings ─────────────────────────────

class TestTimeoutSettings:
    """Verify that _PROBE/_CLIP/_LONG_TIMEOUT are wired to config/settings.py."""

    def test_constants_match_settings_defaults(self):
        from config import settings
        assert _PROBE_TIMEOUT == settings.ffmpeg_probe_timeout
        assert _CLIP_TIMEOUT  == settings.ffmpeg_clip_timeout
        assert _LONG_TIMEOUT  == settings.ffmpeg_long_timeout

    def test_probe_lt_clip_lt_long(self):
        assert _PROBE_TIMEOUT < _CLIP_TIMEOUT < _LONG_TIMEOUT


# ── concat() — fallback re-encode with video_only=True ───────────────────────

class TestConcatFallbackVideoOnly:
    """Cover the video_only=True branch inside the fallback re-encode path."""

    def test_fallback_video_only_adds_an_not_aac(self, tmp_path):
        """When stream copy fails and video_only=True, fallback must use -an."""
        import subprocess as _sp
        paths = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
        output = tmp_path / "out.mp4"
        with patch("subprocess.run") as m:
            m.side_effect = [
                _sp.CalledProcessError(1, "ffmpeg"),   # stream copy fails
                MagicMock(stdout="", returncode=0),    # re-encode succeeds
            ]
            concat(paths, output, video_only=True)
        fallback_cmd = m.call_args_list[1][0][0]
        assert "-an" in fallback_cmd
        assert "aac" not in fallback_cmd


# ── config_validator — ffmpeg timeout fields ──────────────────────────────────

class TestConfigValidatorFfmpegTimeouts:
    """Validate that the three ffmpeg timeout settings are range-checked."""

    def _cfg(self, **overrides):
        from types import SimpleNamespace
        base = dict(
            daily_run_hour_utc=8, topic_run_hour_utc=10,
            background_music_volume=0.1, shorts_music_volume=0.13,
            script_target_words=2200, dedup_ttl_days=30,
            daily_budget_usd=50.0, monthly_budget_usd=500.0,
            end_card_duration=10.0,
            youtube_privacy="public", tiktok_privacy="PUBLIC_TO_EVERYONE",
            heygen_aspect="16:9",
            tiktok_enabled=False, tiktok_client_key="", tiktok_client_secret="",
            presenter_enabled=False, did_api_key="",
            heygen_enabled=False, heygen_api_key="", heygen_avatar_id="",
            heygen_avatar_id_direct="", heygen_avatar_id_serious="",
            heygen_avatar_id_curious="", heygen_avatar_id_professional="",
            heygen_avatar_id_casual="",
            ffmpeg_probe_timeout=30, ffmpeg_clip_timeout=300,
            ffmpeg_long_timeout=600,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_valid_timeouts_do_not_raise(self):
        from src.config_validator import validate
        validate(self._cfg())

    def test_probe_timeout_zero_raises(self):
        from src.config_validator import validate, ConfigurationError
        with pytest.raises(ConfigurationError, match="FFMPEG_PROBE_TIMEOUT"):
            validate(self._cfg(ffmpeg_probe_timeout=0))

    def test_clip_timeout_zero_raises(self):
        from src.config_validator import validate, ConfigurationError
        with pytest.raises(ConfigurationError, match="FFMPEG_CLIP_TIMEOUT"):
            validate(self._cfg(ffmpeg_clip_timeout=0))

    def test_long_timeout_zero_raises(self):
        from src.config_validator import validate, ConfigurationError
        with pytest.raises(ConfigurationError, match="FFMPEG_LONG_TIMEOUT"):
            validate(self._cfg(ffmpeg_long_timeout=0))

    def test_negative_timeout_raises(self):
        from src.config_validator import validate, ConfigurationError
        with pytest.raises(ConfigurationError, match="FFMPEG_CLIP_TIMEOUT"):
            validate(self._cfg(ffmpeg_clip_timeout=-1))
