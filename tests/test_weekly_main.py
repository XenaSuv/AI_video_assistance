"""Tests for src/weekly_main.py — run_weekly_pipeline function."""
from __future__ import annotations

import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch, call

# ── Stub heavy deps before any project import ─────────────────────────────────
for _m in (
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
    "elevenlabs",
    "elevenlabs.client",
    "aiohttp",
):
    sys.modules.setdefault(_m, MagicMock())

import pytest
from src.script_generator import Scene, VideoScript
import src.weekly_main as weekly_main
from src.weekly_main import run_weekly_pipeline


# ── Factories ─────────────────────────────────────────────────────────────────

def _make_scene(idx: int = 0, short_narration: str | None = "Short narration") -> Scene:
    return Scene(
        idx=idx,
        heading=f"Scene {idx}",
        narration="word " * 60,
        visual_prompt="some visual",
        duration_sec=30,
        short_narration=short_narration,
    )


def _make_script(n_scenes: int = 3) -> VideoScript:
    scenes = [_make_scene(i) for i in range(n_scenes)]
    return VideoScript(
        title="Test Tutorial: Claude Basics",
        description="A tutorial description.",
        tags=["ai", "claude", "tutorial"],
        hook="Learn Claude in 5 minutes.",
        hook_variants=["Hook A", "Hook B"],
        scenes=scenes,
    )


def _make_settings(tmp_path: Path) -> MagicMock:
    ms = MagicMock()
    ms.output_dir = tmp_path
    ms.data_dir = tmp_path
    ms.ru_enabled = False
    ms.ru_elevenlabs_voice_id = "ru-voice-id"
    ms.ru_elevenlabs_model = "ru-model"
    ms.ru_youtube_client_secrets = tmp_path / "client_secrets_ru.json"
    ms.ru_youtube_token_file = tmp_path / "token_ru.pickle"
    return ms


def _run_weekly(
    tmp_path: Path,
    *,
    tool: str = "claude",
    topic_override: str | None = None,
    skip_upload: bool = False,
    **patch_overrides,
):
    """Build a fully-patched call to run_weekly_pipeline.

    Returns (result_dict, mocks_dict).
    patch_overrides keys match patch target names (without the 'src.weekly_main.' prefix
    for most, or exact dotted path for others).
    """
    ms = _make_settings(tmp_path)
    script = _make_script()

    # Sensible mock for get_tool
    mock_tool = MagicMock()
    mock_tool.name = tool.capitalize()

    # thumbnail path that exists
    thumb = tmp_path / "thumbnail_bold.jpg"
    thumb.touch()

    # final_video path — needed for _needs_video_rebuild logic
    # We'll patch _needs_video_rebuild to return True by default

    defaults = {
        "settings": ms,
        "get_tool": MagicMock(return_value=mock_tool),
        "pick_topic": MagicMock(return_value="How to use Claude effectively"),
        "_setup_logging": MagicMock(),
        "_load_cached_script": MagicMock(return_value=None),
        "generate_tutorial_script": MagicMock(return_value=script),
        "synthesize_script": MagicMock(),
        "_load_audio_durations": MagicMock(),
        "generate_subtitles": MagicMock(return_value=tmp_path / "subtitles.srt"),
        "_needs_video_rebuild": MagicMock(return_value=True),
        "build_video": MagicMock(),
        "_get_shared_outro": MagicMock(return_value=None),
        "save_used_topic": MagicMock(),
        "build_tutorial_shorts": MagicMock(return_value=[]),
        "build_short": MagicMock(),
        "generate_thumbnail_variants": MagicMock(return_value=[thumb]),
        "pick_thumbnail": MagicMock(return_value=thumb),
        "publish_episode": MagicMock(return_value={"video_id": "vid_abc"}),
        "record_usage": MagicMock(),
        "record_thumbnail_usage": MagicMock(),
        "upload_video": MagicMock(return_value="short_vid_001"),
        "_run_language_variant": MagicMock(return_value={"status": "published"}),
        "notify_success": MagicMock(),
        "notify_failure": MagicMock(),
    }
    # Apply overrides
    defaults.update(patch_overrides)

    mocks: dict = {}

    with ExitStack() as stack:
        for name, mock_val in defaults.items():
            target = f"src.weekly_main.{name}"
            if isinstance(mock_val, MagicMock) and not callable(mock_val):
                m = stack.enter_context(patch(target, mock_val))
            else:
                # patch with new= to replace the name
                m = stack.enter_context(patch(target, new=mock_val))
            mocks[name] = mock_val

        result = run_weekly_pipeline(
            tool_key=tool,
            topic_override=topic_override,
            skip_upload=skip_upload,
        )

    return result, mocks


# ─────────────────────────────────────────────────────────────────────────────
# Validation tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWeeklyPipelineValidation:
    def test_invalid_tool_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="tool must be one of"):
            run_weekly_pipeline(tool_key="invalid_tool")

    def test_invalid_tool_mentions_valid_options(self, tmp_path):
        with pytest.raises(ValueError, match="claude"):
            run_weekly_pipeline(tool_key="bard")

    def test_valid_tool_claude_does_not_raise(self, tmp_path):
        result, _ = _run_weekly(tmp_path, tool="claude")
        assert result["tool"] == "claude"

    def test_valid_tool_chatgpt_does_not_raise(self, tmp_path):
        result, _ = _run_weekly(tmp_path, tool="chatgpt")
        assert result["tool"] == "chatgpt"

    def test_valid_tool_gemini_does_not_raise(self, tmp_path):
        result, _ = _run_weekly(tmp_path, tool="gemini")
        assert result["tool"] == "gemini"


# ─────────────────────────────────────────────────────────────────────────────
# Topic selection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWeeklyPipelineTopic:
    def test_topic_override_used_directly(self, tmp_path):
        result, mocks = _run_weekly(
            tmp_path,
            topic_override="Custom tutorial topic",
        )
        assert result["topic"] == "Custom tutorial topic"

    def test_topic_override_written_to_file(self, tmp_path):
        ms = _make_settings(tmp_path)
        # run_dir will be output_dir/weekly/claude/<date>
        # We need to capture the written file
        result, mocks = _run_weekly(
            tmp_path,
            topic_override="My Override Topic",
            settings=ms,
        )
        # topic_file should have been written to run_dir
        import datetime as dt
        date_str = dt.date.today().isoformat()
        run_dir = tmp_path / "weekly" / "claude" / date_str
        assert (run_dir / "topic.txt").read_text() == "My Override Topic"

    def test_topic_override_skips_pick_topic(self, tmp_path):
        mock_pick = MagicMock(return_value="Unused")
        result, mocks = _run_weekly(
            tmp_path,
            topic_override="Override Topic",
            pick_topic=mock_pick,
        )
        mock_pick.assert_not_called()

    def test_cached_topic_file_read_when_no_override(self, tmp_path):
        ms = _make_settings(tmp_path)
        import datetime as dt
        date_str = dt.date.today().isoformat()
        run_dir = tmp_path / "weekly" / "claude" / date_str
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "topic.txt").write_text("Cached topic from file")

        result, _ = _run_weekly(tmp_path, settings=ms)
        assert result["topic"] == "Cached topic from file"

    def test_cached_topic_skips_pick_topic(self, tmp_path):
        ms = _make_settings(tmp_path)
        import datetime as dt
        date_str = dt.date.today().isoformat()
        run_dir = tmp_path / "weekly" / "claude" / date_str
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "topic.txt").write_text("Cached topic")

        mock_pick = MagicMock(return_value="Should not be called")
        result, _ = _run_weekly(tmp_path, settings=ms, pick_topic=mock_pick)
        mock_pick.assert_not_called()

    def test_pick_topic_called_when_no_file_no_override(self, tmp_path):
        mock_pick = MagicMock(return_value="Picked topic")
        result, _ = _run_weekly(tmp_path, pick_topic=mock_pick)
        mock_pick.assert_called_once()
        assert result["topic"] == "Picked topic"

    def test_pick_topic_receives_tool_key(self, tmp_path):
        mock_pick = MagicMock(return_value="Topic for chatgpt")
        _run_weekly(tmp_path, tool="chatgpt", pick_topic=mock_pick)
        call_args = mock_pick.call_args
        assert "chatgpt" in call_args[0] or "chatgpt" in str(call_args)


# ─────────────────────────────────────────────────────────────────────────────
# Script tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWeeklyPipelineScript:
    def test_cached_script_reused(self, tmp_path):
        cached_script = _make_script(2)
        mock_load = MagicMock(return_value=cached_script)
        mock_generate = MagicMock(return_value=_make_script())

        result, mocks = _run_weekly(
            tmp_path,
            _load_cached_script=mock_load,
            generate_tutorial_script=mock_generate,
        )
        mock_generate.assert_not_called()
        assert result["title"] == cached_script.title

    def test_script_generated_when_cache_miss(self, tmp_path):
        fresh_script = _make_script(4)
        mock_load = MagicMock(return_value=None)
        mock_generate = MagicMock(return_value=fresh_script)

        result, _ = _run_weekly(
            tmp_path,
            _load_cached_script=mock_load,
            generate_tutorial_script=mock_generate,
        )
        mock_generate.assert_called_once()
        assert result["title"] == fresh_script.title
        assert result["num_scenes"] == 4

    def test_generate_tutorial_script_receives_tool_key(self, tmp_path):
        mock_generate = MagicMock(return_value=_make_script())
        _run_weekly(
            tmp_path,
            tool="gemini",
            _load_cached_script=MagicMock(return_value=None),
            generate_tutorial_script=mock_generate,
        )
        call_args = mock_generate.call_args
        assert "gemini" in call_args[0] or "gemini" in str(call_args)

    def test_script_saved_after_generation(self, tmp_path):
        fresh_script = MagicMock(spec=VideoScript)
        fresh_script.title = "Generated"
        fresh_script.scenes = [_make_scene(i) for i in range(3)]
        fresh_script.hook = "hook"
        fresh_script.tags = ["ai"]

        _run_weekly(
            tmp_path,
            _load_cached_script=MagicMock(return_value=None),
            generate_tutorial_script=MagicMock(return_value=fresh_script),
        )
        # save is called at least once after generation (also after synthesize)
        assert fresh_script.save.call_count >= 1

    def test_summary_contains_num_scenes(self, tmp_path):
        script = _make_script(5)
        result, _ = _run_weekly(
            tmp_path,
            _load_cached_script=MagicMock(return_value=script),
        )
        assert result["num_scenes"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# Voice tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWeeklyPipelineVoice:
    def test_synthesize_called_when_no_audio_dir(self, tmp_path):
        mock_synth = MagicMock()
        _run_weekly(tmp_path, synthesize_script=mock_synth)
        mock_synth.assert_called_once()

    def test_synthesize_called_when_audio_dir_empty(self, tmp_path):
        ms = _make_settings(tmp_path)
        import datetime as dt
        date_str = dt.date.today().isoformat()
        run_dir = tmp_path / "weekly" / "claude" / date_str
        run_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = run_dir / "audio"
        audio_dir.mkdir()
        # No mp3 files in audio dir

        mock_synth = MagicMock()
        _run_weekly(tmp_path, settings=ms, synthesize_script=mock_synth)
        mock_synth.assert_called_once()

    def test_cached_audio_used_when_enough_mp3s(self, tmp_path):
        script = _make_script(2)
        ms = _make_settings(tmp_path)

        import datetime as dt
        date_str = dt.date.today().isoformat()
        run_dir = tmp_path / "weekly" / "claude" / date_str
        run_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = run_dir / "audio"
        audio_dir.mkdir()
        # Create enough mp3 files (>= n_scenes)
        for i in range(2):
            (audio_dir / f"scene_{i:02d}.mp3").touch()

        mock_synth = MagicMock()
        mock_load_dur = MagicMock()
        _run_weekly(
            tmp_path,
            settings=ms,
            _load_cached_script=MagicMock(return_value=script),
            synthesize_script=mock_synth,
            _load_audio_durations=mock_load_dur,
        )
        mock_synth.assert_not_called()
        mock_load_dur.assert_called_once()

    def test_total_duration_in_summary(self, tmp_path):
        script = _make_script(3)
        for s in script.scenes:
            s.duration_sec = 10
        result, _ = _run_weekly(
            tmp_path,
            _load_cached_script=MagicMock(return_value=script),
        )
        assert result["total_duration_sec"] == 30


# ─────────────────────────────────────────────────────────────────────────────
# Subtitles tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWeeklyPipelineSubtitles:
    def test_subtitle_exception_swallowed(self, tmp_path):
        mock_gen = MagicMock(side_effect=RuntimeError("Subtitle boom"))
        # Should not raise
        result, _ = _run_weekly(tmp_path, generate_subtitles=mock_gen)
        # Pipeline completes without error
        assert "date" in result

    def test_subtitle_exception_does_not_set_failed_status(self, tmp_path):
        mock_gen = MagicMock(side_effect=RuntimeError("Subtitle boom"))
        result, _ = _run_weekly(
            tmp_path,
            generate_subtitles=mock_gen,
            skip_upload=True,
        )
        assert result.get("status") != "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Video tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWeeklyPipelineVideo:
    def test_build_video_called_when_rebuild_needed(self, tmp_path):
        mock_build = MagicMock()
        _run_weekly(
            tmp_path,
            _needs_video_rebuild=MagicMock(return_value=True),
            build_video=mock_build,
        )
        mock_build.assert_called_once()

    def test_build_video_not_called_when_cache_fresh(self, tmp_path):
        mock_build = MagicMock()
        _run_weekly(
            tmp_path,
            _needs_video_rebuild=MagicMock(return_value=False),
            build_video=mock_build,
        )
        mock_build.assert_not_called()

    def test_save_used_topic_called_after_build(self, tmp_path):
        mock_save = MagicMock()
        _run_weekly(
            tmp_path,
            _needs_video_rebuild=MagicMock(return_value=True),
            save_used_topic=mock_save,
        )
        mock_save.assert_called_once()

    def test_save_used_topic_not_called_when_cached(self, tmp_path):
        mock_save = MagicMock()
        _run_weekly(
            tmp_path,
            _needs_video_rebuild=MagicMock(return_value=False),
            save_used_topic=mock_save,
        )
        mock_save.assert_not_called()

    def test_build_video_receives_tool_key(self, tmp_path):
        mock_build = MagicMock()
        _run_weekly(
            tmp_path,
            tool="chatgpt",
            _needs_video_rebuild=MagicMock(return_value=True),
            build_video=mock_build,
        )
        call_kwargs = mock_build.call_args[1]
        assert call_kwargs.get("tool") == "chatgpt"

    def test_tutorial_shorts_count_in_summary(self, tmp_path):
        shorts_paths = [tmp_path / f"short_{i}.mp4" for i in range(3)]
        result, _ = _run_weekly(
            tmp_path,
            build_tutorial_shorts=MagicMock(return_value=shorts_paths),
        )
        assert result["tutorial_shorts_count"] == 3

    def test_digest_short_built_when_missing(self, tmp_path):
        mock_build_short = MagicMock()
        _run_weekly(tmp_path, build_short=mock_build_short)
        mock_build_short.assert_called_once()

    def test_digest_short_not_rebuilt_when_cached(self, tmp_path):
        ms = _make_settings(tmp_path)
        import datetime as dt
        date_str = dt.date.today().isoformat()
        run_dir = tmp_path / "weekly" / "claude" / date_str
        run_dir.mkdir(parents=True, exist_ok=True)
        # Create the cached shorts.mp4
        (run_dir / "shorts.mp4").touch()

        mock_build_short = MagicMock()
        _run_weekly(tmp_path, settings=ms, build_short=mock_build_short)
        mock_build_short.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Upload tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWeeklyPipelineUpload:
    def test_skip_upload_sets_status_built_not_uploaded(self, tmp_path):
        result, _ = _run_weekly(tmp_path, skip_upload=True)
        assert result["status"] == "built_not_uploaded"

    def test_skip_upload_does_not_call_publish_episode(self, tmp_path):
        mock_pub = MagicMock(return_value={"video_id": "vid_x"})
        _run_weekly(tmp_path, skip_upload=True, publish_episode=mock_pub)
        mock_pub.assert_not_called()

    def test_fresh_upload_calls_publish_episode(self, tmp_path):
        mock_pub = MagicMock(return_value={"video_id": "vid_fresh"})
        _run_weekly(tmp_path, skip_upload=False, publish_episode=mock_pub)
        mock_pub.assert_called_once()

    def test_fresh_upload_sets_status_published(self, tmp_path):
        result, _ = _run_weekly(
            tmp_path,
            skip_upload=False,
            publish_episode=MagicMock(return_value={"video_id": "vid_abc"}),
        )
        assert result["status"] == "published"

    def test_record_usage_called_with_video_id(self, tmp_path):
        mock_rec = MagicMock()
        _run_weekly(
            tmp_path,
            skip_upload=False,
            publish_episode=MagicMock(return_value={"video_id": "vid_xyz"}),
            record_usage=mock_rec,
        )
        mock_rec.assert_called_once()
        args = mock_rec.call_args[0]
        assert "vid_xyz" in args

    def test_record_thumbnail_usage_called(self, tmp_path):
        mock_rec = MagicMock()
        _run_weekly(
            tmp_path,
            skip_upload=False,
            publish_episode=MagicMock(return_value={"video_id": "vid_abc"}),
            record_thumbnail_usage=mock_rec,
        )
        mock_rec.assert_called_once()

    def test_tutorial_shorts_uploaded_via_upload_video(self, tmp_path):
        short_path = tmp_path / "short_0.mp4"
        short_path.touch()
        # Scene 0 has idx 0, so short_path.stem "short_0" → split("_")[1] = "0"
        mock_upload = MagicMock(return_value="short_vid_001")
        _run_weekly(
            tmp_path,
            skip_upload=False,
            build_tutorial_shorts=MagicMock(return_value=[short_path]),
            upload_video=mock_upload,
        )
        mock_upload.assert_called_once()

    def test_tutorial_short_ids_in_summary(self, tmp_path):
        short_path = tmp_path / "short_0.mp4"
        short_path.touch()
        result, _ = _run_weekly(
            tmp_path,
            skip_upload=False,
            build_tutorial_shorts=MagicMock(return_value=[short_path]),
            upload_video=MagicMock(return_value="short_id_001"),
        )
        assert result["tutorial_short_ids"] == ["short_id_001"]

    def test_no_shorts_uploaded_when_skip_upload(self, tmp_path):
        short_path = tmp_path / "short_0.mp4"
        short_path.touch()
        mock_upload = MagicMock(return_value="should_not_be_called")
        _run_weekly(
            tmp_path,
            skip_upload=True,
            build_tutorial_shorts=MagicMock(return_value=[short_path]),
            upload_video=mock_upload,
        )
        mock_upload.assert_not_called()

    def test_publish_episode_ids_merged_into_summary(self, tmp_path):
        result, _ = _run_weekly(
            tmp_path,
            skip_upload=False,
            publish_episode=MagicMock(return_value={"video_id": "v1", "playlist_id": "PL1"}),
        )
        assert result["video_id"] == "v1"
        assert result["playlist_id"] == "PL1"


# ─────────────────────────────────────────────────────────────────────────────
# Russian variant tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWeeklyPipelineRu:
    def test_ru_disabled_skips_language_variant(self, tmp_path):
        ms = _make_settings(tmp_path)
        ms.ru_enabled = False
        mock_variant = MagicMock(return_value={"status": "published"})
        _run_weekly(tmp_path, settings=ms, _run_language_variant=mock_variant)
        mock_variant.assert_not_called()

    def test_ru_disabled_no_ru_in_summary(self, tmp_path):
        ms = _make_settings(tmp_path)
        ms.ru_enabled = False
        result, _ = _run_weekly(tmp_path, settings=ms)
        assert "ru" not in result

    def test_ru_enabled_calls_language_variant(self, tmp_path):
        ms = _make_settings(tmp_path)
        ms.ru_enabled = True
        mock_variant = MagicMock(return_value={"status": "published"})
        _run_weekly(tmp_path, settings=ms, _run_language_variant=mock_variant)
        mock_variant.assert_called_once()

    def test_ru_enabled_result_in_summary(self, tmp_path):
        ms = _make_settings(tmp_path)
        ms.ru_enabled = True
        result, _ = _run_weekly(
            tmp_path,
            settings=ms,
            _run_language_variant=MagicMock(return_value={"status": "ru_published"}),
        )
        assert result["ru"] == {"status": "ru_published"}

    def test_ru_enabled_passes_skip_upload(self, tmp_path):
        ms = _make_settings(tmp_path)
        ms.ru_enabled = True
        mock_variant = MagicMock(return_value={"status": "built"})
        _run_weekly(
            tmp_path,
            skip_upload=True,
            settings=ms,
            _run_language_variant=mock_variant,
        )
        call_kwargs = mock_variant.call_args[1]
        assert call_kwargs.get("skip_upload") is True


# ─────────────────────────────────────────────────────────────────────────────
# Error handling tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWeeklyPipelineErrors:
    def test_exception_sets_status_failed(self, tmp_path):
        with pytest.raises(RuntimeError):
            _run_weekly(
                tmp_path,
                generate_tutorial_script=MagicMock(side_effect=RuntimeError("generate fail")),
                _load_cached_script=MagicMock(return_value=None),
            )

    def test_exception_calls_notify_failure(self, tmp_path):
        mock_notify = MagicMock()
        with pytest.raises(RuntimeError):
            _run_weekly(
                tmp_path,
                generate_tutorial_script=MagicMock(side_effect=RuntimeError("fail")),
                _load_cached_script=MagicMock(return_value=None),
                notify_failure=mock_notify,
            )
        mock_notify.assert_called_once()

    def test_exception_re_raised(self, tmp_path):
        with pytest.raises(RuntimeError, match="propagate this"):
            _run_weekly(
                tmp_path,
                build_video=MagicMock(side_effect=RuntimeError("propagate this")),
                _needs_video_rebuild=MagicMock(return_value=True),
            )

    def test_exception_error_in_summary(self, tmp_path):
        mock_notify = MagicMock()

        def capture_and_raise(*args, **kwargs):
            raise ValueError("something broke")

        try:
            _run_weekly(
                tmp_path,
                generate_tutorial_script=capture_and_raise,
                _load_cached_script=MagicMock(return_value=None),
                notify_failure=mock_notify,
            )
        except ValueError:
            pass

        # notify_failure receives the summary dict with 'error' key
        call_args = mock_notify.call_args[0]
        # third positional arg is the summary dict
        summary_arg = call_args[2]
        assert summary_arg.get("status") == "failed"
        assert "something broke" in summary_arg.get("error", "")

    def test_notify_failure_not_called_on_success(self, tmp_path):
        mock_notify = MagicMock()
        _run_weekly(tmp_path, notify_failure=mock_notify)
        mock_notify.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Summary / notify_success tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWeeklyPipelineSummary:
    def test_summary_contains_date(self, tmp_path):
        import datetime as dt
        result, _ = _run_weekly(tmp_path)
        assert result["date"] == dt.date.today().isoformat()

    def test_summary_contains_tool(self, tmp_path):
        result, _ = _run_weekly(tmp_path, tool="gemini")
        assert result["tool"] == "gemini"

    def test_summary_contains_run_dir(self, tmp_path):
        result, _ = _run_weekly(tmp_path)
        assert "run_dir" in result

    def test_summary_contains_topic(self, tmp_path):
        result, _ = _run_weekly(
            tmp_path,
            topic_override="My Topic",
        )
        assert result["topic"] == "My Topic"

    def test_summary_contains_title(self, tmp_path):
        result, _ = _run_weekly(tmp_path)
        assert "title" in result

    def test_summary_contains_thumbnail_style(self, tmp_path):
        result, _ = _run_weekly(tmp_path)
        assert "thumbnail_style" in result

    def test_notify_success_called_on_happy_path(self, tmp_path):
        mock_notify = MagicMock()
        _run_weekly(tmp_path, notify_success=mock_notify)
        mock_notify.assert_called_once()

    def test_notify_success_called_with_weekly_tag(self, tmp_path):
        mock_notify = MagicMock()
        _run_weekly(tmp_path, notify_success=mock_notify)
        call_args = mock_notify.call_args[0]
        assert "weekly" in call_args

    def test_summary_contains_tutorial_shorts_count(self, tmp_path):
        result, _ = _run_weekly(
            tmp_path,
            build_tutorial_shorts=MagicMock(return_value=[]),
        )
        assert "tutorial_shorts_count" in result

    def test_notify_success_not_called_on_exception(self, tmp_path):
        mock_notify = MagicMock()
        with pytest.raises(RuntimeError):
            _run_weekly(
                tmp_path,
                build_video=MagicMock(side_effect=RuntimeError("boom")),
                _needs_video_rebuild=MagicMock(return_value=True),
                notify_success=mock_notify,
            )
        mock_notify.assert_not_called()
