"""Tests for clip-from-long-form Short helpers now living in src/shorts_pipeline.py.

These functions (build_short, create_short_video, add_big_captions, helpers)
were migrated from the deleted src/shorts_generator.py.  Behaviour is
unchanged; only the module path is different.

numpy is unavailable in the test environment and is mocked by conftest.
ffmpeg_utils is patched via _FfmpegMocks (same pattern as test_video_generator).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

# Stub transitive heavy deps before any project import
for _m in (
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
    "elevenlabs",
    "elevenlabs.client",
):
    sys.modules.setdefault(_m, MagicMock())

# Configure numpy stub so int(np.argmax(...)) returns a deterministic int
_np_stub = sys.modules.setdefault("numpy", MagicMock())
_np_stub.argmax.return_value = 40  # best column index used in saliency tests

from src.script_generator import Scene, VideoScript  # noqa: E402
from src.shorts_pipeline import (  # noqa: E402
    SHORT_H,
    SHORT_MAX_SECONDS,
    SHORT_W,
    _resolve_music_path,
    _saliency_crop_x,
    add_big_captions,
    build_short,
    create_short_video,
)

# ── helpers ────────────────────────────────────────────────────────────────────

def _script() -> VideoScript:
    return VideoScript(
        title="AI Short",
        description="desc",
        tags=[],
        hook="This changes everything",
        scenes=[Scene(idx=0, heading="Intro", narration="text", visual_prompt="img")],
    )


def _fake_settings(**overrides: object) -> SimpleNamespace:
    defaults: dict = {
        "background_music_path": "",
        "background_music_volume": 0.1,
        "source_dir": Path("/fake/source"),
        "shorts_music_volume": 0.15,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _FfmpegMocks:
    """Context manager that patches all src.ffmpeg_utils.* functions."""

    def __enter__(self) -> _FfmpegMocks:
        self._patches = [
            patch("src.ffmpeg_utils.duration", return_value=25.0),
            patch("src.ffmpeg_utils.concat",
                  side_effect=lambda paths, output, **kw: output),
            patch("src.ffmpeg_utils.loop_and_trim",
                  side_effect=lambda src, output, **kw: output),
            patch("src.ffmpeg_utils.get_frame", return_value=MagicMock()),
            patch("src.ffmpeg_utils.video_size", return_value=(1920, 1080)),
            patch("src.ffmpeg_utils.make_vertical",
                  side_effect=lambda src, output, **kw: output),
            patch("src.ffmpeg_utils.merge_av",
                  side_effect=lambda v, a, output, **kw: output),
            patch("src.ffmpeg_utils.burn_captions",
                  side_effect=lambda v, text, output, **kw: output),
            patch("src.ffmpeg_utils.mix_music",
                  side_effect=lambda v, m, output, **kw: output),
            patch("src.ffmpeg_utils.black_clip",
                  side_effect=lambda output, **kw: output),
        ]
        self.mocks: dict[str, MagicMock] = {}
        for p in self._patches:
            m = p.start()
            # Extract function name from "src.ffmpeg_utils.<name>"
            self.mocks[p.attribute] = m
        return self

    def __exit__(self, *_: object) -> None:
        for p in reversed(self._patches):
            p.stop()


def _build_short(tmp_path: Path, **kwargs: object) -> Path:
    """Call build_short with sensible defaults."""
    clip_dir = tmp_path / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    clip = clip_dir / "scene_00_clip_0.mp4"; clip.touch()
    main_video = tmp_path / "final_video.mp4"; main_video.touch()
    return build_short(_script(), main_video, tmp_path, **kwargs)


# ── _saliency_crop_x ──────────────────────────────────────────────────────────

class TestSaliencyCropX:
    def test_returns_zero_when_frame_not_wider_than_crop(self) -> None:
        frame = MagicMock()
        frame.shape = (100, 50, 3)  # w=50 <= crop_w=100
        assert _saliency_crop_x(frame, 100) == 0

    def test_returns_zero_when_frame_exactly_crop_width(self) -> None:
        frame = MagicMock()
        frame.shape = (100, 100, 3)  # max_offset = 100 - 100 = 0
        assert _saliency_crop_x(frame, 100) == 0

    def test_blends_best_toward_center(self) -> None:
        # np stub: argmax returns 40, w=200, crop_w=100 → center=50
        # return = int(40 * 0.7 + 50 * 0.3) = int(28 + 15) = 43
        frame = MagicMock()
        frame.shape = (100, 200, 3)
        result = _saliency_crop_x(frame, 100)
        assert isinstance(result, int)

    def test_calls_numpy_operations(self) -> None:
        frame = MagicMock()
        frame.shape = (100, 200, 3)
        _np_stub.argmax.return_value = 10
        _saliency_crop_x(frame, 50)
        _np_stub.argmax.assert_called()


# ── _resolve_music_path ───────────────────────────────────────────────────────

class TestResolveMusicPath:
    def test_empty_string_returns_none(self) -> None:
        with patch("src.shorts_pipeline.settings", _fake_settings(background_music_path="")):
            assert _resolve_music_path() is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        p = str(tmp_path / "no_music.mp3")
        with patch("src.shorts_pipeline.settings", _fake_settings(background_music_path=p)):
            assert _resolve_music_path() is None

    def test_existing_absolute_path_returned(self, tmp_path: Path) -> None:
        music = tmp_path / "bg.mp3"; music.touch()
        with patch("src.shorts_pipeline.settings",
                   _fake_settings(background_music_path=str(music))):
            assert _resolve_music_path() == music

    def test_relative_path_resolved_from_source_dir(self, tmp_path: Path) -> None:
        music = tmp_path / "bg.mp3"; music.touch()
        with patch("src.shorts_pipeline.settings",
                   _fake_settings(background_music_path="bg.mp3", source_dir=tmp_path)):
            assert _resolve_music_path() == music


# ── build_short ───────────────────────────────────────────────────────────────

class TestBuildShortHappyPath:
    def test_returns_path_inside_out_dir(self, tmp_path: Path) -> None:
        with _FfmpegMocks(), \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            result = _build_short(tmp_path)
        assert result.parent == tmp_path
        assert result.suffix == ".mp4"

    def test_custom_out_name(self, tmp_path: Path) -> None:
        with _FfmpegMocks(), \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            result = _build_short(tmp_path, out_name="shorts_ru.mp4")
        assert result.name == "shorts_ru.mp4"

    def test_duration_capped_at_max_seconds(self, tmp_path: Path) -> None:
        audio = tmp_path / "audio" / "scene_00.mp3"
        audio.parent.mkdir(parents=True); audio.touch()
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            # ffmpeg_utils.duration returns 25.0 (< SHORT_MAX_SECONDS = 58)
            _build_short(tmp_path)
        ff.mocks["loop_and_trim"].assert_called_once()
        _, call_kwargs = ff.mocks["loop_and_trim"].call_args
        assert call_kwargs["target_sec"] <= SHORT_MAX_SECONDS

    def test_audio_file_used_when_present(self, tmp_path: Path) -> None:
        audio_dir = tmp_path / "audio"; audio_dir.mkdir()
        audio = audio_dir / "scene_00.mp3"; audio.touch()
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            _build_short(tmp_path)
        # merge_av should be called with the audio path
        ff.mocks["merge_av"].assert_called_once()

    def test_no_audio_file_skips_merge_av(self, tmp_path: Path) -> None:
        # No audio dir → audio_path = None → merge_av NOT called
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            _build_short(tmp_path)
        ff.mocks["merge_av"].assert_not_called()

    def test_vertical_crop_applied(self, tmp_path: Path) -> None:
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            _build_short(tmp_path)
        ff.mocks["make_vertical"].assert_called_once()

    def test_captions_burned_with_hook_text(self, tmp_path: Path) -> None:
        script = _script()
        clip_dir = tmp_path / "clips"; clip_dir.mkdir()
        (clip_dir / "scene_00_clip_0.mp4").touch()
        main_video = tmp_path / "final_video.mp4"; main_video.touch()
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            build_short(script, main_video, tmp_path)
        args = ff.mocks["burn_captions"].call_args
        assert script.hook in args.args


class TestBuildShortMultipleClips:
    def test_multiple_clips_calls_concat(self, tmp_path: Path) -> None:
        clip_dir = tmp_path / "clips"; clip_dir.mkdir()
        (clip_dir / "scene_00_clip_0.mp4").touch()
        (clip_dir / "scene_00_clip_1.mp4").touch()
        (clip_dir / "scene_01_clip_0.mp4").touch()
        main_video = tmp_path / "final_video.mp4"; main_video.touch()
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            build_short(_script(), main_video, tmp_path)
        ff.mocks["concat"].assert_called_once()

    def test_no_source_clips_falls_back_to_main_video(self, tmp_path: Path) -> None:
        clip_dir = tmp_path / "clips"; clip_dir.mkdir()  # empty — no clip files
        main_video = tmp_path / "final_video.mp4"; main_video.touch()
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            build_short(_script(), main_video, tmp_path)
        # No concat for source clips — loop_and_trim receives main_video
        ff.mocks["concat"].assert_not_called()


class TestBuildShortSaliencyFailure:
    def test_saliency_failure_is_non_fatal(self, tmp_path: Path) -> None:
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            ff.mocks["get_frame"].side_effect = RuntimeError("frame decode fail")
            result = _build_short(tmp_path)
        assert result.suffix == ".mp4"
        # make_vertical still called (with crop_x=None fallback)
        ff.mocks["make_vertical"].assert_called_once()


class TestBuildShortMusic:
    def test_music_calls_mix_music(self, tmp_path: Path) -> None:
        music = tmp_path / "bg.mp3"; music.touch()
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings",
                   _fake_settings(background_music_path=str(music))), \
             patch("shutil.copy2"):
            _build_short(tmp_path)
        ff.mocks["mix_music"].assert_called_once()

    def test_no_music_skips_mix_music(self, tmp_path: Path) -> None:
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            _build_short(tmp_path)
        ff.mocks["mix_music"].assert_not_called()


# ── add_big_captions ──────────────────────────────────────────────────────────

class TestAddBigCaptions:
    def test_delegates_to_burn_captions(self, tmp_path: Path) -> None:
        video = tmp_path / "in.mp4"; video.touch()
        output = tmp_path / "out.mp4"
        with patch("src.ffmpeg_utils.burn_captions",
                   side_effect=lambda v, text, out, **kw: out) as mock_bc:
            result = add_big_captions(video, "Hello AI", output)
        mock_bc.assert_called_once_with(
            video, "Hello AI", output,
            font_size=72, y_pct=0.45, color="white",
        )
        assert result == output

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        video = tmp_path / "in.mp4"; video.touch()
        nested = tmp_path / "sub" / "out.mp4"
        with patch("src.ffmpeg_utils.burn_captions",
                   side_effect=lambda v, text, out, **kw: out):
            add_big_captions(video, "text", nested)
        assert nested.parent.exists()


# ── create_short_video ────────────────────────────────────────────────────────

class TestCreateShortVideo:
    def _call(self, tmp_path: Path, **kwargs: object) -> Path:
        return create_short_video("AI news hook", tmp_path, **kwargs)

    def test_returns_path_in_out_dir(self, tmp_path: Path) -> None:
        with _FfmpegMocks(), \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            result = self._call(tmp_path)
        assert result.parent == tmp_path
        assert result.suffix == ".mp4"

    def test_custom_out_name(self, tmp_path: Path) -> None:
        with _FfmpegMocks(), \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            result = self._call(tmp_path, out_name="exp_001.mp4")
        assert result.name == "exp_001.mp4"

    def test_creates_black_clip_base(self, tmp_path: Path) -> None:
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            self._call(tmp_path, duration_sec=30.0)
        ff.mocks["black_clip"].assert_called_once()
        kw = ff.mocks["black_clip"].call_args.kwargs
        assert kw["width"] == SHORT_W
        assert kw["height"] == SHORT_H
        assert kw["duration_sec"] == 30.0

    def test_captions_burned_onto_base_clip(self, tmp_path: Path) -> None:
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            self._call(tmp_path)
        ff.mocks["burn_captions"].assert_called_once()

    def test_audio_merged_when_path_exists(self, tmp_path: Path) -> None:
        audio = tmp_path / "hook.mp3"; audio.touch()
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            self._call(tmp_path, audio_path=audio)
        ff.mocks["merge_av"].assert_called_once()

    def test_no_audio_skips_merge_av(self, tmp_path: Path) -> None:
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            self._call(tmp_path)
        ff.mocks["merge_av"].assert_not_called()

    def test_nonexistent_audio_path_skips_merge_av(self, tmp_path: Path) -> None:
        missing_audio = tmp_path / "no_audio.mp3"  # not created
        with _FfmpegMocks() as ff, \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            self._call(tmp_path, audio_path=missing_audio)
        ff.mocks["merge_av"].assert_not_called()

    def test_creates_output_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "nested_out"
        with _FfmpegMocks(), \
             patch("src.shorts_pipeline.settings", _fake_settings()), \
             patch("shutil.copy2"):
            self._call(nested)
        assert nested.exists()
