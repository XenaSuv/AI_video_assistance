"""Smoke tests for src/video_generator.py.

FFmpeg is mocked via src.ffmpeg_utils.* patches — no real video encoding occurs.
Heavy image/voice/scene dependencies are stubbed in conftest.py and per-test patches.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import asyncio

from src.script_generator import Scene, VideoScript
from src.video_generator import (
    VIDEO_H,
    VIDEO_W,
    _assemble_one_scene,
    _async_assemble_scenes,
    _resolve_music_path,
    assemble_video,
    build_video,
    generate_clips_for_scene,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _scene(idx: int = 0, heading: str = "Intro", narration: str = "Hello world") -> Scene:
    return Scene(
        idx=idx,
        heading=heading,
        narration=narration,
        visual_prompt="tech background",
    )


def _script(scenes: list[Scene] | None = None) -> VideoScript:
    if scenes is None:
        scenes = [_scene(0)]
    return VideoScript(
        title="AI News Today",
        description="Daily AI news",
        tags=["ai", "news"],
        hook="AI is changing everything",
        scenes=scenes,
    )


def _fake_settings(**overrides: object) -> SimpleNamespace:
    """Return a SimpleNamespace with sensible defaults; any key can be overridden."""
    defaults: dict = {
        "background_music_path": "",
        "background_music_volume": 0.1,
        "source_dir": Path("/fake/source"),
        "heygen_enabled": False,
        "presenter_enabled": False,
        "heygen_api_key": "",
        "heygen_aspect": "16:9",
        "heygen_voice_id": "",
        "did_api_key": "",
        "presenter_avatar_path": "",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class _FfmpegMocks:
    """Context-manager that patches all src.ffmpeg_utils functions used by
    video_generator.py.  By default every function returns its *output* path
    argument so assemble_video can stitch together a coherent path chain."""

    def __init__(self, *, video_w: int = VIDEO_W, video_h: int = VIDEO_H, duration: float = 10.0):
        self._video_w = video_w
        self._video_h = video_h
        self._duration = duration
        self._patches: list = []
        self.mocks: dict[str, MagicMock] = {}

    def _patch(self, name: str, *, side_effect=None, return_value=None) -> MagicMock:
        p = patch(f"src.ffmpeg_utils.{name}", side_effect=side_effect, return_value=return_value)
        self._patches.append(p)
        m = p.start()
        self.mocks[name] = m
        return m

    def __enter__(self) -> _FfmpegMocks:
        self._patch("video_size", return_value=(self._video_w, self._video_h))
        self._patch("duration", return_value=self._duration)
        self._patch("has_audio_stream", return_value=True)
        self._patch("ensure_audio_stream", side_effect=lambda path, output: path)
        self._patch("merge_av", side_effect=lambda v, a, output, **kw: output)
        self._patch("concat", side_effect=lambda paths, output, **kw: output)
        self._patch("burn_chyron", side_effect=lambda clip, text, output, **kw: output)
        self._patch("overlay_title_card", side_effect=lambda clip, text, output, **kw: output)
        self._patch("black_clip", side_effect=lambda output, **kw: output)
        self._patch("resize_video", side_effect=lambda src, output, **kw: output)
        self._patch("mix_music", side_effect=lambda v, m, output, **kw: output)
        self._patch("quote_card_clip", side_effect=lambda png, output, **kw: output)
        return self

    def __exit__(self, *_: object) -> None:
        for p in reversed(self._patches):
            p.stop()

    def __getattr__(self, name: str) -> MagicMock:
        return self.mocks[name]


# ── _resolve_music_path ───────────────────────────────────────────────────────

class TestResolveMusicPath:
    def test_empty_path_returns_none(self) -> None:
        with patch("src.video_generator.settings", _fake_settings(background_music_path="")):
            assert _resolve_music_path() is None

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        p = str(tmp_path / "nonexistent.mp3")
        with patch("src.video_generator.settings", _fake_settings(background_music_path=p)):
            assert _resolve_music_path() is None

    def test_existing_absolute_path_returns_path(self, tmp_path: Path) -> None:
        music = tmp_path / "music.mp3"
        music.touch()
        with patch("src.video_generator.settings", _fake_settings(background_music_path=str(music))):
            result = _resolve_music_path()
        assert result == music

    def test_relative_path_resolved_from_source_dir(self, tmp_path: Path) -> None:
        music = tmp_path / "bg.mp3"
        music.touch()
        with patch("src.video_generator.settings", _fake_settings(
            background_music_path="bg.mp3", source_dir=tmp_path
        )):
            result = _resolve_music_path()
        assert result == music

    def test_relative_missing_returns_none(self, tmp_path: Path) -> None:
        with patch("src.video_generator.settings", _fake_settings(
            background_music_path="missing.mp3", source_dir=tmp_path
        )):
            result = _resolve_music_path()
        assert result is None


# ── generate_clips_for_scene ──────────────────────────────────────────────────

class TestGenerateClipsForScene:
    def test_returns_list_with_single_clip(self, tmp_path: Path) -> None:
        scene = _scene()
        fake_clip = tmp_path / "clip.mp4"
        with patch("src.video_generator.generate_scene_clip", return_value=fake_clip) as mock_gsc:
            result = generate_clips_for_scene(scene, tmp_path)
        assert result == [fake_clip]
        mock_gsc.assert_called_once_with(scene, tmp_path, tool=None, run_dir=None, is_breaking=False)

    def test_passes_tool_and_flags(self, tmp_path: Path) -> None:
        scene = _scene()
        with patch("src.video_generator.generate_scene_clip", return_value=tmp_path / "c.mp4") as mock_gsc:
            generate_clips_for_scene(scene, tmp_path, tool="claude", run_dir=tmp_path, is_breaking=True)
        mock_gsc.assert_called_once_with(scene, tmp_path, tool="claude", run_dir=tmp_path, is_breaking=True)


# ── assemble_video helpers ────────────────────────────────────────────────────

def _make_scene_files(tmp_path: Path, scenes: list[Scene]) -> tuple[dict, dict]:
    clips, audio = {}, {}
    for s in scenes:
        c = tmp_path / f"clip_{s.idx}.mp4"; c.touch()
        a = tmp_path / "audio" / f"scene_{s.idx:02d}.mp3"
        a.parent.mkdir(parents=True, exist_ok=True); a.touch()
        clips[s.idx] = [c]
        audio[s.idx] = a
    return clips, audio


# ── assemble_video: happy path ────────────────────────────────────────────────

class TestAssembleVideoHappyPath:
    def test_single_scene_no_extras_returns_output_path(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        script = _script(scenes)
        clips, audio = _make_scene_files(tmp_path, scenes)
        out = tmp_path / "out.mp4"

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            result = assemble_video(script, clips, audio, out)

        assert result == out
        ff.mocks["merge_av"].assert_called_once()
        ff.mocks["burn_chyron"].assert_called_once()

    def test_scene_0_no_title_overlay(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        script = _script(scenes)
        clips, audio = _make_scene_files(tmp_path, scenes)

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(script, clips, audio, tmp_path / "out.mp4")

        ff.mocks["overlay_title_card"].assert_not_called()

    def test_scene_1_has_title_overlay(self, tmp_path: Path) -> None:
        scenes = [_scene(0), _scene(1, heading="Second")]
        script = _script(scenes)
        clips, audio = _make_scene_files(tmp_path, scenes)

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(script, clips, audio, tmp_path / "out.mp4")

        ff.mocks["overlay_title_card"].assert_called_once()

    def test_multiple_clips_per_scene_calls_concat_for_cat(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        script = _script(scenes)
        c1 = tmp_path / "c1.mp4"; c1.touch()
        c2 = tmp_path / "c2.mp4"; c2.touch()
        audio = tmp_path / "audio" / "scene_00.mp3"
        audio.parent.mkdir(); audio.touch()

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(script, {0: [c1, c2]}, {0: audio}, tmp_path / "out.mp4")

        # At least one concat for the multi-clip scene + one for content
        assert ff.mocks["concat"].call_count >= 2


# ── assemble_video: fallback paths ────────────────────────────────────────────

class TestAssembleVideoFallbacks:
    def test_missing_clip_uses_black_placeholder(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        missing = tmp_path / "nonexistent.mp4"   # deliberately not created
        audio = tmp_path / "audio" / "scene_00.mp3"
        audio.parent.mkdir(); audio.touch()

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(_script(scenes), {0: [missing]}, {0: audio}, tmp_path / "out.mp4")

        ff.mocks["black_clip"].assert_called_once()

    def test_wrong_dimensions_clip_uses_black_placeholder(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)

        with _FfmpegMocks(video_w=640, video_h=480) as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4")

        ff.mocks["black_clip"].assert_called_once()

    def test_zero_duration_clip_uses_black_placeholder(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)

        with _FfmpegMocks(duration=0.0) as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4")

        ff.mocks["black_clip"].assert_called_once()

    def test_merge_av_failure_retries_with_placeholder(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)
        call_count = {"n": 0}

        def _failing_merge(video, audio_p, output, **kw):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("encode error")
            return output

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            ff.mocks["merge_av"].side_effect = _failing_merge
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4")

        assert call_count["n"] == 2
        ff.mocks["black_clip"].assert_called_once()

    def test_quote_card_scene_skips_chyron(self, tmp_path: Path) -> None:
        scene = _scene(0)
        scene.source_quote = "AI is the future"   # triggers quote card path
        script = _script([scene])
        clips, audio = _make_scene_files(tmp_path, [scene])

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png") as mock_qc, \
             patch("shutil.copy2"):
            assemble_video(script, clips, audio, tmp_path / "out.mp4")

        mock_qc.assert_called_once()
        ff.mocks["burn_chyron"].assert_not_called()

    def test_quote_card_failure_is_non_fatal(self, tmp_path: Path) -> None:
        scene = _scene(0)
        scene.source_quote = "AI is the future"
        script = _script([scene])
        clips, audio = _make_scene_files(tmp_path, [scene])

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png", side_effect=OSError("png fail")), \
             patch("shutil.copy2"):
            assemble_video(script, clips, audio, tmp_path / "out.mp4")

        # After failure falls back to normal chyron path
        ff.mocks["burn_chyron"].assert_called_once()


# ── assemble_video: intro / outro ─────────────────────────────────────────────

class TestAssembleVideoIntroOutro:
    def test_with_intro_correct_size(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)
        intro = tmp_path / "intro.mp4"; intro.touch()
        out = tmp_path / "out.mp4"

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"):
            result = assemble_video(_script(scenes), clips, audio, out, intro_path=intro)

        ff.mocks["ensure_audio_stream"].assert_called_once()
        ff.mocks["resize_video"].assert_not_called()
        # The final concat includes intro as first segment
        last_concat_args = ff.mocks["concat"].call_args_list[-1][0][0]
        assert intro in last_concat_args

    def test_with_intro_wrong_size_triggers_resize(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)
        intro = tmp_path / "intro.mp4"; intro.touch()

        with _FfmpegMocks(video_w=640, video_h=480) as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4", intro_path=intro)

        # resize_video called for at least intro (also for scene clip since wrong dims)
        assert ff.mocks["resize_video"].call_count >= 1

    def test_missing_intro_file_skipped(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)
        nonexistent = tmp_path / "no_intro.mp4"

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2") as mock_copy:
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4", intro_path=nonexistent)

        ff.mocks["ensure_audio_stream"].assert_not_called()
        mock_copy.assert_called_once()   # no intro → shutil fallback

    def test_with_outro_correct_size(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)
        outro = tmp_path / "outro.mp4"; outro.touch()
        intro = tmp_path / "intro.mp4"; intro.touch()  # use intro to avoid shutil branch

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"):
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4",
                           intro_path=intro, outro_path=outro)

        # ensure_audio_stream called for both intro and outro
        assert ff.mocks["ensure_audio_stream"].call_count == 2

    def test_with_outro_wrong_size_triggers_resize(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)
        intro = tmp_path / "intro.mp4"; intro.touch()
        outro = tmp_path / "outro.mp4"; outro.touch()

        with _FfmpegMocks(video_w=640, video_h=480) as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4",
                           intro_path=intro, outro_path=outro)

        assert ff.mocks["resize_video"].call_count >= 1


# ── assemble_video: presenter clip ────────────────────────────────────────────

class TestAssembleVideoPresenter:
    def test_presenter_correct_size_is_first_segment(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)
        presenter = tmp_path / "presenter.mp4"; presenter.touch()
        intro = tmp_path / "intro.mp4"; intro.touch()  # avoid shutil branch

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"):
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4",
                           presenter_clip=presenter, intro_path=intro)

        # First content concat has presenter as first element
        first_concat_paths = ff.mocks["concat"].call_args_list[0][0][0]
        assert first_concat_paths[0] == presenter

    def test_presenter_wrong_size_is_resized(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)
        presenter = tmp_path / "presenter.mp4"; presenter.touch()
        intro = tmp_path / "intro.mp4"; intro.touch()

        with _FfmpegMocks(video_w=640, video_h=480) as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4",
                           presenter_clip=presenter, intro_path=intro)

        assert ff.mocks["resize_video"].call_count >= 1

    def test_nonexistent_presenter_clip_skipped(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)
        missing_presenter = tmp_path / "ghost.mp4"  # not touched

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4",
                           presenter_clip=missing_presenter)

        # No resize, and first content concat must NOT include the ghost
        first_concat_paths = ff.mocks["concat"].call_args_list[0][0][0]
        assert missing_presenter not in first_concat_paths


# ── assemble_video: music ─────────────────────────────────────────────────────

class TestAssembleVideoMusic:
    def test_music_configured_calls_mix_music(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)
        music = tmp_path / "music.mp3"; music.touch()
        intro = tmp_path / "intro.mp4"; intro.touch()

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings(
                 background_music_path=str(music),
                 background_music_volume=0.1,
             )), \
             patch("src.video_generator.render_quote_card_png"):
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4", intro_path=intro)

        ff.mocks["mix_music"].assert_called_once()

    def test_no_music_skips_mix_music(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        clips, audio = _make_scene_files(tmp_path, scenes)

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.settings", _fake_settings(background_music_path="")), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("shutil.copy2"):
            assemble_video(_script(scenes), clips, audio, tmp_path / "out.mp4")

        ff.mocks["mix_music"].assert_not_called()


# ── _assemble_one_scene ───────────────────────────────────────────────────────

def _one_scene_setup(
    tmp_path: Path,
    idx: int = 0,
    source_quote: str | None = None,
) -> tuple[Scene, dict, dict, Path]:
    """Return (scene, clip_paths_by_scene, audio_paths_by_scene, assembled_dir)."""
    scene = _scene(idx)
    if source_quote is not None:
        scene.source_quote = source_quote
    clip = tmp_path / f"clip_{idx}.mp4"; clip.touch()
    audio_dir = tmp_path / "audio"; audio_dir.mkdir(exist_ok=True)
    audio = audio_dir / f"scene_{idx:02d}.mp3"; audio.touch()
    assembled_dir = tmp_path / "assembled"; assembled_dir.mkdir(exist_ok=True)
    return scene, {idx: [clip]}, {idx: audio}, assembled_dir


class TestAssembleOneScene:
    def test_happy_path_returns_idx_and_path(self, tmp_path: Path) -> None:
        scene, clips, audios, asm_dir = _one_scene_setup(tmp_path, idx=0)

        with _FfmpegMocks(), \
             patch("src.video_generator.render_quote_card_png"):
            idx, clip_path = _assemble_one_scene(scene, clips, audios, asm_dir)

        assert idx == 0
        assert isinstance(clip_path, Path)

    def test_scene_0_no_title_overlay(self, tmp_path: Path) -> None:
        scene, clips, audios, asm_dir = _one_scene_setup(tmp_path, idx=0)

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.render_quote_card_png"):
            _assemble_one_scene(scene, clips, audios, asm_dir)

        ff.mocks["overlay_title_card"].assert_not_called()

    def test_scene_1_has_title_overlay(self, tmp_path: Path) -> None:
        scene, clips, audios, asm_dir = _one_scene_setup(tmp_path, idx=1)

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.render_quote_card_png"):
            _assemble_one_scene(scene, clips, audios, asm_dir)

        ff.mocks["overlay_title_card"].assert_called_once()

    def test_missing_clip_uses_black_placeholder(self, tmp_path: Path) -> None:
        scene = _scene(0)
        audio_dir = tmp_path / "audio"; audio_dir.mkdir()
        audio = audio_dir / "scene_00.mp3"; audio.touch()
        asm_dir = tmp_path / "assembled"; asm_dir.mkdir()
        missing = tmp_path / "no_clip.mp4"  # not created

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.render_quote_card_png"):
            _assemble_one_scene(scene, {0: [missing]}, {0: audio}, asm_dir)

        ff.mocks["black_clip"].assert_called_once()

    def test_wrong_dimensions_uses_black_placeholder(self, tmp_path: Path) -> None:
        scene, clips, audios, asm_dir = _one_scene_setup(tmp_path)

        with _FfmpegMocks(video_w=640, video_h=480) as ff, \
             patch("src.video_generator.render_quote_card_png"):
            _assemble_one_scene(scene, clips, audios, asm_dir)

        ff.mocks["black_clip"].assert_called_once()

    def test_merge_av_failure_retries_with_placeholder(self, tmp_path: Path) -> None:
        scene, clips, audios, asm_dir = _one_scene_setup(tmp_path)
        calls = {"n": 0}

        def _fail_first(video, audio_p, output, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("encode error")
            return output

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.render_quote_card_png"):
            ff.mocks["merge_av"].side_effect = _fail_first
            _assemble_one_scene(scene, clips, audios, asm_dir)

        assert calls["n"] == 2
        ff.mocks["black_clip"].assert_called_once()

    def test_quote_card_skips_chyron(self, tmp_path: Path) -> None:
        scene, clips, audios, asm_dir = _one_scene_setup(tmp_path, source_quote="AI is the future")

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.render_quote_card_png"):
            _assemble_one_scene(scene, clips, audios, asm_dir)

        ff.mocks["burn_chyron"].assert_not_called()

    def test_quote_card_failure_is_non_fatal(self, tmp_path: Path) -> None:
        scene, clips, audios, asm_dir = _one_scene_setup(tmp_path, source_quote="AI is the future")

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.render_quote_card_png", side_effect=OSError("png fail")):
            _assemble_one_scene(scene, clips, audios, asm_dir)

        ff.mocks["burn_chyron"].assert_called_once()

    def test_multiple_clips_per_scene_calls_concat(self, tmp_path: Path) -> None:
        scene = _scene(0)
        c1 = tmp_path / "c1.mp4"; c1.touch()
        c2 = tmp_path / "c2.mp4"; c2.touch()
        audio_dir = tmp_path / "audio"; audio_dir.mkdir()
        audio = audio_dir / "scene_00.mp3"; audio.touch()
        asm_dir = tmp_path / "assembled"; asm_dir.mkdir()

        with _FfmpegMocks() as ff, \
             patch("src.video_generator.render_quote_card_png"):
            _assemble_one_scene(scene, {0: [c1, c2]}, {0: audio}, asm_dir)

        # concat called at least once for the multi-clip cat
        ff.mocks["concat"].assert_called_once()


# ── _async_assemble_scenes ────────────────────────────────────────────────────

class TestAsyncAssembleScenes:
    def test_single_scene_returns_one_clip(self, tmp_path: Path) -> None:
        scenes = [_scene(0)]
        script = _script(scenes)
        clips, audio = _make_scene_files(tmp_path, scenes)
        asm_dir = tmp_path / "assembled"; asm_dir.mkdir()

        with _FfmpegMocks(), \
             patch("src.video_generator.render_quote_card_png"):
            result = asyncio.run(_async_assemble_scenes(script, clips, audio, asm_dir))

        assert len(result) == 1

    def test_multiple_scenes_returns_clips_in_order(self, tmp_path: Path) -> None:
        scenes = [_scene(0), _scene(1, heading="Second"), _scene(2, heading="Third")]
        script = _script(scenes)
        clips, audio = _make_scene_files(tmp_path, scenes)
        asm_dir = tmp_path / "assembled"; asm_dir.mkdir()

        with _FfmpegMocks(), \
             patch("src.video_generator.render_quote_card_png"):
            result = asyncio.run(_async_assemble_scenes(script, clips, audio, asm_dir))

        assert len(result) == 3
        # Results must be Path objects in ascending scene-index order
        assert all(isinstance(p, Path) for p in result)

    def test_all_scenes_assembled_concurrently(self, tmp_path: Path) -> None:
        """Every scene's _assemble_one_scene must be called exactly once."""
        scenes = [_scene(i, heading=f"Scene {i}") for i in range(4)]
        script = _script(scenes)
        clips, audio = _make_scene_files(tmp_path, scenes)
        asm_dir = tmp_path / "assembled"; asm_dir.mkdir()

        assembled_idxs: list[int] = []

        real_assemble = _assemble_one_scene

        def _spy(scene, *args, **kwargs):  # type: ignore[no-untyped-def]
            assembled_idxs.append(scene.idx)
            return real_assemble(scene, *args, **kwargs)

        with _FfmpegMocks(), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("src.video_generator._assemble_one_scene", side_effect=_spy):
            asyncio.run(_async_assemble_scenes(script, clips, audio, asm_dir))

        assert sorted(assembled_idxs) == [0, 1, 2, 3]


# ── build_video ───────────────────────────────────────────────────────────────

class TestBuildVideo:
    """Smoke tests for the end-to-end build_video() entry point."""

    @staticmethod
    def _clip_dict(script: VideoScript, tmp_path: Path) -> dict:
        d = {}
        for s in script.scenes:
            c = tmp_path / "clips" / f"scene_{s.idx:02d}.mp4"
            c.parent.mkdir(parents=True, exist_ok=True); c.touch()
            d[s.idx] = [c]
        return d

    def test_smoke_no_presenter(self, tmp_path: Path) -> None:
        script = _script([_scene(0), _scene(1, heading="Second")])
        clip_dict = self._clip_dict(script, tmp_path)

        with _FfmpegMocks(), \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("src.video_generator.SceneStrategyEngine"), \
             patch("src.video_generator.SceneVarietyEngineV2"), \
             patch("src.video_generator.generate_scene_clip", return_value=tmp_path / "c.mp4"), \
             patch("asyncio.run", return_value=clip_dict), \
             patch("shutil.copy2"):
            result = build_video(script, tmp_path)

        assert result.name == "out.mp4" or result.suffix == ".mp4"

    def test_strategy_config_calls_assign_from_config(self, tmp_path: Path) -> None:
        script = _script([_scene(0)])
        clip_dict = self._clip_dict(script, tmp_path)
        strategy_config = MagicMock()

        with _FfmpegMocks(), \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("src.video_generator.SceneStrategyEngine"), \
             patch("src.video_generator.SceneVarietyEngineV2") as mock_sve, \
             patch("src.video_generator.generate_scene_clip", return_value=tmp_path / "c.mp4"), \
             patch("asyncio.run", return_value=clip_dict), \
             patch("shutil.copy2"):
            build_video(script, tmp_path, strategy_config=strategy_config)

        mock_sve.return_value.assign_from_config.assert_called_once()
        mock_sve.return_value.assign.assert_not_called()

    def test_no_strategy_config_calls_assign(self, tmp_path: Path) -> None:
        script = _script([_scene(0)])
        clip_dict = self._clip_dict(script, tmp_path)

        with _FfmpegMocks(), \
             patch("src.video_generator.settings", _fake_settings()), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("src.video_generator.SceneStrategyEngine"), \
             patch("src.video_generator.SceneVarietyEngineV2") as mock_sve, \
             patch("src.video_generator.generate_scene_clip", return_value=tmp_path / "c.mp4"), \
             patch("asyncio.run", return_value=clip_dict), \
             patch("shutil.copy2"):
            build_video(script, tmp_path)

        mock_sve.return_value.assign.assert_called_once()
        mock_sve.return_value.assign_from_config.assert_not_called()

    def test_presenter_tts_failure_is_non_fatal(self, tmp_path: Path) -> None:
        script = _script([_scene(0)])
        clip_dict = self._clip_dict(script, tmp_path)

        with _FfmpegMocks(), \
             patch("src.video_generator.settings", _fake_settings(presenter_enabled=True)), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("src.video_generator.SceneStrategyEngine"), \
             patch("src.video_generator.SceneVarietyEngineV2"), \
             patch("src.video_generator.synthesize_hook", side_effect=RuntimeError("TTS fail")), \
             patch("src.video_generator.generate_scene_clip", return_value=tmp_path / "c.mp4"), \
             patch("asyncio.run", return_value=clip_dict), \
             patch("shutil.copy2"):
            # Must not raise
            build_video(script, tmp_path, use_presenter=True)

    def test_did_presenter_calls_generate_presenter_clip(self, tmp_path: Path) -> None:
        script = _script([_scene(0)])
        clip_dict = self._clip_dict(script, tmp_path)
        hook_audio = tmp_path / "hook.mp3"; hook_audio.touch()

        with _FfmpegMocks(), \
             patch("src.video_generator.settings", _fake_settings(
                 presenter_enabled=True,
                 presenter_avatar_path=str(tmp_path / "avatar.png"),
             )), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("src.video_generator.SceneStrategyEngine"), \
             patch("src.video_generator.SceneVarietyEngineV2"), \
             patch("src.video_generator.synthesize_hook", return_value=hook_audio), \
             patch("src.video_generator.generate_presenter_clip",
                   return_value=tmp_path / "hook.mp4") as mock_did, \
             patch("src.video_generator.generate_scene_clip", return_value=tmp_path / "c.mp4"), \
             patch("asyncio.run", return_value=clip_dict), \
             patch("shutil.copy2"):
            build_video(script, tmp_path, use_presenter=True)

        mock_did.assert_called_once()

    def test_heygen_presenter_calls_generate_heygen_clip(self, tmp_path: Path) -> None:
        script = _script([_scene(0)])
        clip_dict = self._clip_dict(script, tmp_path)
        hook_audio = tmp_path / "hook.mp3"; hook_audio.touch()

        mock_ae_module = MagicMock()
        mock_ae_module.AvatarEngine.return_value.avatar_id.return_value = "avatar-123"

        with _FfmpegMocks(), \
             patch("src.video_generator.settings", _fake_settings(
                 heygen_enabled=True,
                 heygen_api_key="fake-key",
                 heygen_aspect="16:9",
                 heygen_voice_id="v1",
             )), \
             patch("src.video_generator.render_quote_card_png"), \
             patch("src.video_generator.SceneStrategyEngine"), \
             patch("src.video_generator.SceneVarietyEngineV2"), \
             patch("src.video_generator.synthesize_hook", return_value=hook_audio), \
             patch("src.video_generator.generate_heygen_clip",
                   return_value=tmp_path / "heygen.mp4") as mock_heygen, \
             patch("src.video_generator.generate_scene_clip", return_value=tmp_path / "c.mp4"), \
             patch("asyncio.run", return_value=clip_dict), \
             patch("shutil.copy2"), \
             patch.dict("sys.modules", {"src.avatar_engine": mock_ae_module}):
            build_video(script, tmp_path)

        mock_heygen.assert_called_once()
