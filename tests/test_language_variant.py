"""Tests for src/language_variant.py — _run_language_variant helper."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub google/elevenlabs deps loaded transitively
for _m in (
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
    "elevenlabs",
    "elevenlabs.client",
):
    sys.modules.setdefault(_m, MagicMock())

from src.language_variant import _run_language_variant
from src.script_generator import Scene, VideoScript


def _script() -> VideoScript:
    return VideoScript(
        title="Test RU",
        description="desc",
        tags=[],
        hook="hook",
        scenes=[Scene(idx=0, heading="S0", narration="text", visual_prompt="img")],
    )


def _all_mocks():
    """Return a list of patches that suppress all external I/O."""
    return [
        patch("src.language_variant._load_cached_script", return_value=None),
        patch("src.language_variant.translate_script", return_value=_script()),
        patch("src.language_variant.synthesize_script"),
        patch("src.language_variant.generate_subtitles", return_value=None),
        patch("src.language_variant._needs_video_rebuild", return_value=False),
        patch("src.language_variant.generate_thumbnail", return_value=Path("/tmp/t.jpg")),
        patch("src.language_variant.publish_episode", return_value={"video_id": "vid123"}),
        patch("src.language_variant._load_audio_durations"),
        patch("src.language_variant._get_intro_duration", return_value=0.0),
        patch("src.shorts_pipeline.build_short"),
    ]


def _call(**kwargs):
    """Call _run_language_variant with sensible defaults."""
    defaults = dict(
        english_script=_script(),
        run_dir=Path("/tmp/fake_run"),
        lang_code="ru",
        lang_name="Russian",
        voice_id="voice-ru",
        voice_model="model-ru",
        client_secrets=Path("/tmp/s.json"),
        token_file=Path("/tmp/t.json"),
    )
    defaults.update(kwargs)
    return _run_language_variant(**defaults)


class TestRunLanguageVariantHappyPath:
    def test_returns_published_status(self, tmp_path):
        mocks = _all_mocks()
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6], mocks[7], mocks[8], mocks[9]:
            result = _call(run_dir=tmp_path)
        assert result["status"] == "published"
        assert result["video_id"] == "vid123"

    def test_skip_upload_returns_built_not_uploaded(self, tmp_path):
        mocks = _all_mocks()
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6], mocks[7], mocks[8], mocks[9]:
            result = _call(run_dir=tmp_path, skip_upload=True)
        assert result["status"] == "built_not_uploaded"

    def test_reuses_cached_script_when_present(self, tmp_path):
        cached = _script()
        with (patch("src.language_variant._load_cached_script", return_value=cached),
              patch("src.language_variant.translate_script") as mock_translate,
              patch("src.language_variant.synthesize_script"),
              patch("src.language_variant.generate_subtitles", return_value=None),
              patch("src.language_variant._needs_video_rebuild", return_value=False),
              patch("src.language_variant.generate_thumbnail", return_value=Path("/tmp/t.jpg")),
              patch("src.language_variant.publish_episode", return_value={}),
              patch("src.language_variant._load_audio_durations"),
              patch("src.language_variant._get_intro_duration", return_value=0.0),
              patch("src.shorts_pipeline.build_short")):
            _call(run_dir=tmp_path)
        mock_translate.assert_not_called()

    def test_reuses_cached_audio_when_mp3s_present(self, tmp_path):
        audio_dir = tmp_path / "audio_ru"
        audio_dir.mkdir()
        (audio_dir / "scene_00.mp3").write_bytes(b"fake")
        with (patch("src.language_variant._load_cached_script", return_value=None),
              patch("src.language_variant.translate_script", return_value=_script()),
              patch("src.language_variant.synthesize_script") as mock_synth,
              patch("src.language_variant.generate_subtitles", return_value=None),
              patch("src.language_variant._needs_video_rebuild", return_value=False),
              patch("src.language_variant.generate_thumbnail", return_value=Path("/tmp/t.jpg")),
              patch("src.language_variant.publish_episode", return_value={}),
              patch("src.language_variant._load_audio_durations"),
              patch("src.language_variant._get_intro_duration", return_value=0.0),
              patch("src.shorts_pipeline.build_short")):
            _call(run_dir=tmp_path)
        mock_synth.assert_not_called()

    def test_subtitle_failure_is_non_fatal(self, tmp_path):
        with (patch("src.language_variant._load_cached_script", return_value=None),
              patch("src.language_variant.translate_script", return_value=_script()),
              patch("src.language_variant.synthesize_script"),
              patch("src.language_variant.generate_subtitles", side_effect=RuntimeError("fail")),
              patch("src.language_variant._needs_video_rebuild", return_value=False),
              patch("src.language_variant.generate_thumbnail", return_value=Path("/tmp/t.jpg")),
              patch("src.language_variant.publish_episode", return_value={}),
              patch("src.language_variant._load_audio_durations"),
              patch("src.language_variant._get_intro_duration", return_value=0.0),
              patch("src.shorts_pipeline.build_short")):
            result = _call(run_dir=tmp_path)
        assert result is not None

    def test_include_short_false_skips_short(self, tmp_path):
        mocks = _all_mocks()
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6], mocks[7], mocks[8], mocks[9]:
            result = _call(run_dir=tmp_path, include_short=False)
        assert result is not None

    def test_reuses_cached_short_when_exists(self, tmp_path):
        short = tmp_path / "shorts_ru.mp4"
        short.write_bytes(b"fake")
        mocks = _all_mocks()
        with mocks[0], mocks[1], mocks[2], mocks[3], mocks[4], mocks[5], mocks[6], mocks[7], mocks[8], mocks[9]:
            result = _call(run_dir=tmp_path, include_short=True)
        assert result is not None

    def test_assemble_called_when_video_needs_rebuild(self, tmp_path):
        with (patch("src.language_variant._load_cached_script", return_value=None),
              patch("src.language_variant.translate_script", return_value=_script()),
              patch("src.language_variant.synthesize_script"),
              patch("src.language_variant.generate_subtitles", return_value=None),
              patch("src.language_variant._needs_video_rebuild", return_value=True),
              patch("src.language_variant.assemble_video") as mock_assemble,
              patch("src.language_variant.generate_thumbnail", return_value=Path("/tmp/t.jpg")),
              patch("src.language_variant.publish_episode", return_value={}),
              patch("src.language_variant._load_audio_durations"),
              patch("src.language_variant._get_intro_duration", return_value=0.0),
              patch("src.shorts_pipeline.build_short")):
            _call(run_dir=tmp_path)
        mock_assemble.assert_called_once()


class TestRunLanguageVariantConcurrency:
    """Verify that the parallel steps both execute and interact correctly."""

    def test_subtitles_and_video_both_run(self, tmp_path):
        """generate_subtitles and assemble_video are both called when video needs rebuild."""
        with (patch("src.language_variant._load_cached_script", return_value=None),
              patch("src.language_variant.translate_script", return_value=_script()),
              patch("src.language_variant.synthesize_script"),
              patch("src.language_variant.generate_subtitles",
                    return_value=Path("/tmp/subs.srt")) as mock_subs,
              patch("src.language_variant._needs_video_rebuild", return_value=True),
              patch("src.language_variant.assemble_video") as mock_assemble,
              patch("src.language_variant.generate_thumbnail", return_value=Path("/tmp/t.jpg")),
              patch("src.language_variant.publish_episode", return_value={}),
              patch("src.language_variant._load_audio_durations"),
              patch("src.language_variant._get_intro_duration", return_value=0.0),
              patch("src.shorts_pipeline.build_short")):
            _call(run_dir=tmp_path)

        mock_subs.assert_called_once()
        mock_assemble.assert_called_once()

    def test_subtitle_failure_does_not_block_video(self, tmp_path):
        """Video assembly completes even when subtitle generation raises."""
        with (patch("src.language_variant._load_cached_script", return_value=None),
              patch("src.language_variant.translate_script", return_value=_script()),
              patch("src.language_variant.synthesize_script"),
              patch("src.language_variant.generate_subtitles",
                    side_effect=RuntimeError("whisper down")),
              patch("src.language_variant._needs_video_rebuild", return_value=True),
              patch("src.language_variant.assemble_video") as mock_assemble,
              patch("src.language_variant.generate_thumbnail", return_value=Path("/tmp/t.jpg")),
              patch("src.language_variant.publish_episode", return_value={}),
              patch("src.language_variant._load_audio_durations"),
              patch("src.language_variant._get_intro_duration", return_value=0.0),
              patch("src.shorts_pipeline.build_short")):
            result = _call(run_dir=tmp_path)

        assert result["status"] == "published"
        mock_assemble.assert_called_once()

    def test_short_and_thumbnail_both_called_when_include_short(self, tmp_path):
        """When include_short=True, both build_short and generate_thumbnail are called."""
        with (patch("src.language_variant._load_cached_script", return_value=None),
              patch("src.language_variant.translate_script", return_value=_script()),
              patch("src.language_variant.synthesize_script"),
              patch("src.language_variant.generate_subtitles", return_value=None),
              patch("src.language_variant._needs_video_rebuild", return_value=False),
              patch("src.language_variant.generate_thumbnail",
                    return_value=Path("/tmp/t.jpg")) as mock_thumb,
              patch("src.language_variant.publish_episode", return_value={}),
              patch("src.language_variant._load_audio_durations"),
              patch("src.language_variant._get_intro_duration", return_value=0.0),
              patch("src.shorts_pipeline.build_short") as mock_build_short):
            _call(run_dir=tmp_path, include_short=True)

        mock_thumb.assert_called_once()
        mock_build_short.assert_called_once()

    def test_thumbnail_called_even_when_include_short_false(self, tmp_path):
        """generate_thumbnail always runs regardless of include_short."""
        with (patch("src.language_variant._load_cached_script", return_value=None),
              patch("src.language_variant.translate_script", return_value=_script()),
              patch("src.language_variant.synthesize_script"),
              patch("src.language_variant.generate_subtitles", return_value=None),
              patch("src.language_variant._needs_video_rebuild", return_value=False),
              patch("src.language_variant.generate_thumbnail",
                    return_value=Path("/tmp/t.jpg")) as mock_thumb,
              patch("src.language_variant.publish_episode", return_value={}),
              patch("src.language_variant._load_audio_durations"),
              patch("src.language_variant._get_intro_duration", return_value=0.0),
              patch("src.shorts_pipeline.build_short")):
            result = _call(run_dir=tmp_path, include_short=False)

        mock_thumb.assert_called_once()
        assert result is not None

    def test_subtitle_path_passed_to_upload(self, tmp_path):
        """subtitle_path returned by generate_subtitles is forwarded to publish_episode."""
        srt = Path("/tmp/sub.srt")
        with (patch("src.language_variant._load_cached_script", return_value=None),
              patch("src.language_variant.translate_script", return_value=_script()),
              patch("src.language_variant.synthesize_script"),
              patch("src.language_variant.generate_subtitles", return_value=srt),
              patch("src.language_variant._needs_video_rebuild", return_value=False),
              patch("src.language_variant.generate_thumbnail", return_value=Path("/tmp/t.jpg")),
              patch("src.language_variant.publish_episode",
                    return_value={"video_id": "v1"}) as mock_pub,
              patch("src.language_variant._load_audio_durations"),
              patch("src.language_variant._get_intro_duration", return_value=0.0),
              patch("src.shorts_pipeline.build_short")):
            _call(run_dir=tmp_path)

        call_kwargs = mock_pub.call_args.kwargs
        assert call_kwargs["subtitle_path"] == srt
