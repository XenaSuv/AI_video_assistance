"""Tests for src/media_builder.py — _MediaBuilder class."""
from __future__ import annotations

import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, call, patch

# ── Stub heavy deps before any project import ─────────────────────────────────
for _m in (
    "elevenlabs",
    "elevenlabs.client",
    "src.video_generator",
    "src.voice_generator",
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
):
    sys.modules.setdefault(_m, MagicMock())

import pytest

from src.media_builder import _MediaBuilder
from src.script_generator import Scene, VideoScript

# ── Factories ─────────────────────────────────────────────────────────────────

def _mock_script(n_scenes: int = 3) -> VideoScript:
    scenes = [
        Scene(
            idx=i,
            heading=f"Scene {i}",
            narration="word " * 80,
            visual_prompt="some visual",
            duration_sec=30,
        )
        for i in range(n_scenes)
    ]
    return VideoScript(
        title="Test Episode",
        description="A test description.",
        tags=["ai", "test"],
        hook="This will change everything.",
        hook_variants=["Hook A", "Hook B"],
        scenes=scenes,
    )


def _builder(
    tmp_path: Path,
    *,
    script: VideoScript | None = None,
    cp: MagicMock | None = None,
    summary: dict | None = None,
    seen: MagicMock | None = None,
) -> _MediaBuilder:
    if script is None:
        script = _mock_script()
    if cp is None:
        cp = MagicMock()
        cp.is_done.return_value = False
        cp.metadata.return_value = {}
    if summary is None:
        summary = {}
    if seen is None:
        seen = MagicMock()

    observer = MagicMock()
    live = MagicMock()
    news = [MagicMock()]

    return _MediaBuilder(
        script=script,
        run_dir=tmp_path,
        cp=cp,
        observer=observer,
        live=live,
        seen=seen,
        news=news,
        summary=summary,
        thompson_preferred_type=None,
    )


# ── __init__ tests ────────────────────────────────────────────────────────────

class TestMediaBuilderInit:
    def test_output_attrs_all_none(self, tmp_path):
        b = _builder(tmp_path)
        assert b.subtitle_path is None
        assert b.en_intro is None
        assert b.en_outro is None
        assert b.long_video is None
        assert b.short_video is None
        assert b.thumbnail is None

    def test_init_stores_all_params(self, tmp_path):
        script = _mock_script(4)
        cp = MagicMock()
        summary = {"existing": True}
        seen = MagicMock()
        b = _builder(tmp_path, script=script, cp=cp, summary=summary, seen=seen)
        assert b._script is script
        assert b._run_dir == tmp_path
        assert b._cp is cp
        assert b._summary is summary
        assert b._seen is seen

    def test_thompson_preferred_type_stored(self, tmp_path):
        b = _builder(tmp_path)
        assert b._thompson_preferred_type is None


# ── build_voice tests ─────────────────────────────────────────────────────────

class TestBuildVoice:
    def test_checkpoint_done_calls_step_skip_and_load_durations(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = True
        summary = {}
        b = _builder(tmp_path, cp=cp, summary=summary)

        with patch("src.media_builder._load_audio_durations") as mock_load:
            b.build_voice()

        b._observer.step_skip.assert_called_once()
        mock_load.assert_called_once()
        assert "total_duration_sec" in summary

    def test_checkpoint_done_does_not_call_asyncio_run(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = True
        b = _builder(tmp_path, cp=cp)

        with ExitStack() as stack:
            mock_run = stack.enter_context(patch("src.media_builder.asyncio.run"))
            stack.enter_context(patch("src.media_builder._load_audio_durations"))
            b.build_voice()

        mock_run.assert_not_called()

    def test_audio_dir_missing_calls_asyncio_run(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        script = _mock_script(3)
        b = _builder(tmp_path, script=script, cp=cp)
        # audio dir does NOT exist in tmp_path

        with ExitStack() as stack:
            mock_run = stack.enter_context(patch("src.media_builder.asyncio.run"))
            stack.enter_context(patch("src.media_builder._load_audio_durations"))
            b.build_voice()

        mock_run.assert_called_once()

    def test_fewer_mp3s_than_scenes_calls_asyncio_run(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        (audio_dir / "scene_0.mp3").touch()  # only 1 mp3, but 3 scenes

        cp = MagicMock()
        cp.is_done.return_value = False
        script = _mock_script(3)
        b = _builder(tmp_path, script=script, cp=cp)

        with ExitStack() as stack:
            mock_run = stack.enter_context(patch("src.media_builder.asyncio.run"))
            stack.enter_context(patch("src.media_builder._load_audio_durations"))
            b.build_voice()

        mock_run.assert_called_once()

    def test_cached_mp3s_present_does_not_call_asyncio_run(self, tmp_path):
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        for i in range(3):
            (audio_dir / f"scene_{i}.mp3").touch()

        cp = MagicMock()
        cp.is_done.return_value = False
        script = _mock_script(3)
        b = _builder(tmp_path, script=script, cp=cp)

        with ExitStack() as stack:
            mock_run = stack.enter_context(patch("src.media_builder.asyncio.run"))
            mock_load = stack.enter_context(patch("src.media_builder._load_audio_durations"))
            b.build_voice()

        mock_run.assert_not_called()
        mock_load.assert_called_once()

    def test_fresh_run_marks_done_and_step_done(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        b = _builder(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.asyncio.run"))
            stack.enter_context(patch("src.media_builder._load_audio_durations"))
            b.build_voice()

        cp.mark_done.assert_called_once()
        b._observer.step_done.assert_called_once()

    def test_sets_total_duration_sec_in_summary(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = True
        script = _mock_script(3)  # each scene duration_sec=30 → total 90
        summary = {}
        b = _builder(tmp_path, script=script, cp=cp, summary=summary)

        with patch("src.media_builder._load_audio_durations"):
            b.build_voice()

        assert summary["total_duration_sec"] == 90


# ── build_subtitles tests ─────────────────────────────────────────────────────

class TestBuildSubtitles:
    def _patched_settings(self, tmp_path: Path) -> MagicMock:
        ms = MagicMock()
        ms.source_dir = tmp_path
        return ms

    def test_sets_en_intro_when_file_exists(self, tmp_path):
        intro = tmp_path / "ai-news-intro.mp4"
        intro.touch()
        ms = self._patched_settings(tmp_path)
        cp = MagicMock()
        cp.is_done.return_value = True
        b = _builder(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.settings", ms))
            stack.enter_context(patch("src.media_builder._get_shared_outro", return_value=None))
            stack.enter_context(patch("src.media_builder._get_intro_duration", return_value=5.0))
            b.build_subtitles()

        assert b.en_intro == intro

    def test_en_intro_none_when_file_missing(self, tmp_path):
        ms = self._patched_settings(tmp_path)
        # ai-news-intro.mp4 does NOT exist in tmp_path
        cp = MagicMock()
        cp.is_done.return_value = True
        b = _builder(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.settings", ms))
            stack.enter_context(patch("src.media_builder._get_shared_outro", return_value=None))
            b.build_subtitles()

        assert b.en_intro is None

    def test_en_outro_always_set_from_helper(self, tmp_path):
        ms = self._patched_settings(tmp_path)
        fake_outro = Path("/fake/outro.mp4")
        cp = MagicMock()
        cp.is_done.return_value = True
        b = _builder(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.settings", ms))
            stack.enter_context(patch("src.media_builder._get_shared_outro", return_value=fake_outro))
            b.build_subtitles()

        assert b.en_outro == fake_outro

    def test_cached_srt_pre_set_when_file_exists(self, tmp_path):
        srt = tmp_path / "subtitles.srt"
        srt.touch()
        ms = self._patched_settings(tmp_path)
        cp = MagicMock()
        cp.is_done.return_value = True
        b = _builder(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.settings", ms))
            stack.enter_context(patch("src.media_builder._get_shared_outro", return_value=None))
            b.build_subtitles()

        assert b.subtitle_path == srt

    def test_subtitle_path_none_when_no_srt(self, tmp_path):
        ms = self._patched_settings(tmp_path)
        cp = MagicMock()
        cp.is_done.return_value = True
        b = _builder(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.settings", ms))
            stack.enter_context(patch("src.media_builder._get_shared_outro", return_value=None))
            b.build_subtitles()

        assert b.subtitle_path is None

    def test_checkpoint_done_calls_step_skip(self, tmp_path):
        ms = self._patched_settings(tmp_path)
        cp = MagicMock()
        cp.is_done.return_value = True
        b = _builder(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.settings", ms))
            stack.enter_context(patch("src.media_builder._get_shared_outro", return_value=None))
            b.build_subtitles()

        b._observer.step_skip.assert_called_once_with("subtitles")

    def test_fresh_run_calls_generate_subtitles_and_marks_done(self, tmp_path):
        ms = self._patched_settings(tmp_path)
        cp = MagicMock()
        cp.is_done.return_value = False
        fake_srt = tmp_path / "subtitles.srt"
        b = _builder(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.settings", ms))
            stack.enter_context(patch("src.media_builder._get_shared_outro", return_value=None))
            stack.enter_context(patch("src.media_builder._get_intro_duration", return_value=0.0))
            mock_gen = stack.enter_context(
                patch("src.media_builder.generate_subtitles", return_value=fake_srt)
            )
            b.build_subtitles()

        mock_gen.assert_called_once()
        cp.mark_done.assert_called_once_with("subtitles")
        b._observer.step_done.assert_called_once_with("subtitles")

    def test_subtitle_failure_is_nonfatal(self, tmp_path):
        ms = self._patched_settings(tmp_path)
        cp = MagicMock()
        cp.is_done.return_value = False
        b = _builder(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.settings", ms))
            stack.enter_context(patch("src.media_builder._get_shared_outro", return_value=None))
            stack.enter_context(patch("src.media_builder._get_intro_duration", return_value=0.0))
            stack.enter_context(
                patch("src.media_builder.generate_subtitles", side_effect=RuntimeError("TTS down"))
            )
            b.build_subtitles()  # must NOT raise

        b._observer.step_fail.assert_called_once()
        cp.mark_done.assert_not_called()


# ── build_video tests ─────────────────────────────────────────────────────────

class TestBuildVideo:
    def test_long_video_always_set(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.build_video"))
            stack.enter_context(patch("src.media_builder._needs_video_rebuild", return_value=False))
            stack.enter_context(patch("src.media_builder.settings"))
            stack.enter_context(patch("src.broll_fetcher.get_pexels_credits", return_value=[]))
            b = _builder(tmp_path, cp=cp)
            b.build_video()

        assert b.long_video == tmp_path / "final_video.mp4"

    def test_short_video_is_none_always(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.build_video"))
            stack.enter_context(patch("src.media_builder._needs_video_rebuild", return_value=False))
            stack.enter_context(patch("src.media_builder.settings"))
            stack.enter_context(patch("src.broll_fetcher.get_pexels_credits", return_value=[]))
            b = _builder(tmp_path, cp=cp)
            b.build_video()

        assert b.short_video is None

    def test_not_done_calls_build_video_function(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False

        with ExitStack() as stack:
            mock_bv = stack.enter_context(patch("src.media_builder.build_video"))
            stack.enter_context(patch("src.media_builder._needs_video_rebuild", return_value=False))
            stack.enter_context(patch("src.media_builder.settings"))
            stack.enter_context(patch("src.broll_fetcher.get_pexels_credits", return_value=[]))
            b = _builder(tmp_path, cp=cp)
            b.build_video()

        mock_bv.assert_called_once()

    def test_done_and_no_rebuild_skips_build(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = True

        with ExitStack() as stack:
            mock_bv = stack.enter_context(patch("src.media_builder.build_video"))
            stack.enter_context(patch("src.media_builder._needs_video_rebuild", return_value=False))
            stack.enter_context(patch("src.media_builder.settings"))
            stack.enter_context(patch("src.broll_fetcher.get_pexels_credits", return_value=[]))
            b = _builder(tmp_path, cp=cp)
            b.build_video()

        mock_bv.assert_not_called()
        b._observer.step_skip.assert_called_once_with("video")

    def test_needs_rebuild_even_when_checkpoint_done(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = True

        with ExitStack() as stack:
            mock_bv = stack.enter_context(patch("src.media_builder.build_video"))
            stack.enter_context(patch("src.media_builder._needs_video_rebuild", return_value=True))
            stack.enter_context(patch("src.media_builder.settings"))
            stack.enter_context(patch("src.broll_fetcher.get_pexels_credits", return_value=[]))
            b = _builder(tmp_path, cp=cp)
            b.build_video()

        mock_bv.assert_called_once()

    def test_marks_seen_and_marks_done_on_fresh_build(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        seen = MagicMock()

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.build_video"))
            stack.enter_context(patch("src.media_builder._needs_video_rebuild", return_value=False))
            stack.enter_context(patch("src.media_builder.settings"))
            stack.enter_context(patch("src.broll_fetcher.get_pexels_credits", return_value=[]))
            b = _builder(tmp_path, cp=cp, seen=seen)
            b.build_video()

        seen.mark_featured.assert_called_once_with(b._news)
        cp.mark_done.assert_called_once()

    def test_pexels_credits_appended_to_description(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        script = _mock_script()

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.build_video"))
            stack.enter_context(patch("src.media_builder._needs_video_rebuild", return_value=False))
            stack.enter_context(patch("src.media_builder.settings"))
            stack.enter_context(
                patch("src.broll_fetcher.get_pexels_credits", return_value=["https://pexels.com/1"])
            )
            b = _builder(tmp_path, script=script, cp=cp)
            b.build_video()

        assert "Pexels" in b._script.description
        assert "https://pexels.com/1" in b._script.description

    def test_no_credits_does_not_modify_description(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        script = _mock_script()
        original_desc = script.description

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.build_video"))
            stack.enter_context(patch("src.media_builder._needs_video_rebuild", return_value=False))
            stack.enter_context(patch("src.media_builder.settings"))
            stack.enter_context(patch("src.broll_fetcher.get_pexels_credits", return_value=[]))
            b = _builder(tmp_path, script=script, cp=cp)
            b.build_video()

        assert b._script.description == original_desc


# ── build_thumbnail tests ─────────────────────────────────────────────────────

class TestBuildThumbnail:
    def _prep(self, tmp_path: Path, *, cp_done: bool = False, style: str = "bold"):
        cp = MagicMock()
        cp.is_done.return_value = cp_done
        cp.metadata.return_value = {"style": style}
        b = _builder(tmp_path, cp=cp)
        b.long_video = tmp_path / "final_video.mp4"
        b.long_video.touch()
        return b, cp

    def test_not_done_generates_thumbnail(self, tmp_path):
        b, cp = self._prep(tmp_path, cp_done=False)
        fake_thumb = tmp_path / "thumbnail_bold.jpg"
        fake_thumb.touch()

        with ExitStack() as stack:
            mock_gen = stack.enter_context(
                patch("src.media_builder.generate_thumbnail_variants", return_value=[fake_thumb])
            )
            stack.enter_context(patch("src.media_builder.pick_thumbnail", return_value=fake_thumb))
            stack.enter_context(patch("src.media_builder.settings"))
            b.build_thumbnail()

        mock_gen.assert_called_once()
        cp.mark_done.assert_called_once()
        b._observer.step_done.assert_called_once()
        assert b.thumbnail == fake_thumb

    def test_not_done_sets_thumbnail_style_in_summary(self, tmp_path):
        b, cp = self._prep(tmp_path, cp_done=False)
        fake_thumb = tmp_path / "thumbnail_minimal.jpg"
        fake_thumb.touch()

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.media_builder.generate_thumbnail_variants", return_value=[fake_thumb])
            )
            stack.enter_context(patch("src.media_builder.pick_thumbnail", return_value=fake_thumb))
            stack.enter_context(patch("src.media_builder.settings"))
            b.build_thumbnail()

        assert b._summary["thumbnail_style"] == "minimal"

    def test_done_with_existing_file_calls_step_skip(self, tmp_path):
        b, cp = self._prep(tmp_path, cp_done=True, style="bold")
        (tmp_path / "thumbnail_bold.jpg").touch()

        with ExitStack() as stack:
            mock_gen = stack.enter_context(
                patch("src.media_builder.generate_thumbnail_variants", return_value=[])
            )
            stack.enter_context(patch("src.media_builder.pick_thumbnail"))
            stack.enter_context(patch("src.media_builder.settings"))
            b.build_thumbnail()

        mock_gen.assert_not_called()
        b._observer.step_skip.assert_called_once()

    def test_done_but_missing_file_regenerates(self, tmp_path):
        b, cp = self._prep(tmp_path, cp_done=True, style="bold")
        fake_thumb = tmp_path / "thumbnail_bold.jpg"
        # file intentionally NOT created

        with ExitStack() as stack:
            mock_gen = stack.enter_context(
                patch("src.media_builder.generate_thumbnail_variants", return_value=[fake_thumb])
            )
            stack.enter_context(patch("src.media_builder.pick_thumbnail", return_value=fake_thumb))
            stack.enter_context(patch("src.media_builder.settings"))
            b.build_thumbnail()

        mock_gen.assert_called_once()

    def test_done_sets_thumbnail_and_style_from_checkpoint_metadata(self, tmp_path):
        b, cp = self._prep(tmp_path, cp_done=True, style="cinematic")
        (tmp_path / "thumbnail_cinematic.jpg").touch()

        with ExitStack() as stack:
            stack.enter_context(patch("src.media_builder.generate_thumbnail_variants"))
            stack.enter_context(patch("src.media_builder.pick_thumbnail"))
            stack.enter_context(patch("src.media_builder.settings"))
            b.build_thumbnail()

        assert b._summary["thumbnail_style"] == "cinematic"
        assert b.thumbnail == tmp_path / "thumbnail_cinematic.jpg"

    def test_fresh_marks_checkpoint_done_with_style(self, tmp_path):
        b, cp = self._prep(tmp_path, cp_done=False)
        fake_thumb = tmp_path / "thumbnail_dark.jpg"
        fake_thumb.touch()

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.media_builder.generate_thumbnail_variants", return_value=[fake_thumb])
            )
            stack.enter_context(patch("src.media_builder.pick_thumbnail", return_value=fake_thumb))
            stack.enter_context(patch("src.media_builder.settings"))
            b.build_thumbnail()

        cp.mark_done.assert_called_once_with("thumbnail", {"style": "dark"})
