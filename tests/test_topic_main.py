"""Tests for src/topic_main.py — run_topic_pipeline function and CLI."""
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
import src.topic_main as topic_main
from src.topic_main import run_topic_pipeline, main, _parse_args


# ── Factories ─────────────────────────────────────────────────────────────────

def _make_scene(idx: int = 0) -> Scene:
    return Scene(
        idx=idx,
        heading=f"Scene {idx}",
        narration="word " * 60,
        visual_prompt="some visual",
        duration_sec=30,
    )


def _make_script(n_scenes: int = 3) -> VideoScript:
    scenes = [_make_scene(i) for i in range(n_scenes)]
    return VideoScript(
        title="Test Topic: GPT-5 Review",
        description="A topic description.",
        tags=["ai", "gpt5", "review"],
        hook="This is the hook for GPT-5.",
        hook_variants=["Hook A", "Hook B"],
        scenes=scenes,
    )


def _make_idea() -> MagicMock:
    idea = MagicMock()
    idea.format = "honest_review"
    idea.subject = "GPT-5 released"
    idea.suggested_title = "I tried GPT-5 — honest take"
    idea.angle = "Real-world testing with surprising results"
    idea.to_dict.return_value = {
        "format": "honest_review",
        "subject": "GPT-5 released",
        "suggested_title": "I tried GPT-5 — honest take",
        "angle": "Real-world testing with surprising results",
    }
    return idea


def _make_settings(tmp_path: Path) -> MagicMock:
    ms = MagicMock()
    ms.output_dir = tmp_path
    ms.data_dir = tmp_path
    return ms


def _make_ol_result(script: VideoScript, modified: bool = False) -> MagicMock:
    r = MagicMock()
    r.script = script
    r.modified = modified
    r.loops = []
    return r


def _make_cm_result(script: VideoScript, modified: bool = False) -> MagicMock:
    r = MagicMock()
    r.script = script
    r.modified = modified
    r.question = "What do you think about GPT-5?"
    r.inserted_at_scene = 2
    return r


def _make_pa_result(script: VideoScript, modified: bool = False) -> MagicMock:
    r = MagicMock()
    r.script = script
    r.modified = modified
    r.flat_zones = []
    r.scores = []
    return r


def _run_topic(
    tmp_path: Path,
    *,
    idea_override=None,
    format_filter=None,
    skip_upload: bool = False,
    **overrides,
):
    """Build a fully-patched call to run_topic_pipeline.

    Returns (result_dict, mocks_dict).
    """
    ms = _make_settings(tmp_path)
    idea = _make_idea()
    script = _make_script()

    thumb = tmp_path / "thumbnail_bold.jpg"
    thumb.touch()

    ol_result = _make_ol_result(script)
    cm_result = _make_cm_result(script)
    pa_result = _make_pa_result(script)

    defaults = {
        "settings": ms,
        "_setup_logging": MagicMock(),
        "pick_next_topic": MagicMock(return_value=idea),
        "_load_cached_script": MagicMock(return_value=None),
        "generate_topic_script": MagicMock(return_value=script),
        "apply_open_loops": MagicMock(return_value=ol_result),
        "apply_comment_magnet": MagicMock(return_value=cm_result),
        "audit_pacing": MagicMock(return_value=pa_result),
        "synthesize_script": MagicMock(),
        "_load_audio_durations": MagicMock(),
        "generate_subtitles": MagicMock(return_value=tmp_path / "subtitles.srt"),
        "_needs_video_rebuild": MagicMock(return_value=True),
        "build_video": MagicMock(),
        "_get_shared_outro": MagicMock(return_value=None),
        "build_short": MagicMock(),
        "generate_thumbnail_variants": MagicMock(return_value=[thumb]),
        "pick_thumbnail": MagicMock(return_value=thumb),
        "publish_episode": MagicMock(return_value={"video_id": "vid_abc"}),
        "record_usage": MagicMock(),
        "record_thumbnail_usage": MagicMock(),
        "notify_success": MagicMock(),
        "notify_failure": MagicMock(),
    }
    defaults.update(overrides)

    mocks: dict = {}

    with ExitStack() as stack:
        for name, mock_val in defaults.items():
            target = f"src.topic_main.{name}"
            m = stack.enter_context(patch(target, new=mock_val))
            mocks[name] = mock_val

        result = run_topic_pipeline(
            format_filter=format_filter,
            idea_override=idea_override,
            skip_upload=skip_upload,
        )

    return result, mocks


# ─────────────────────────────────────────────────────────────────────────────
# Idea selection tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunTopicPipelineIdeaSelection:
    def test_idea_override_used_directly(self, tmp_path):
        idea = _make_idea()
        result, mocks = _run_topic(tmp_path, idea_override=idea)
        mocks["pick_next_topic"].assert_not_called()
        assert result["subject"] == idea.subject

    def test_pick_next_topic_called_when_no_override(self, tmp_path):
        mock_pick = MagicMock(return_value=_make_idea())
        result, mocks = _run_topic(tmp_path, pick_next_topic=mock_pick)
        mock_pick.assert_called_once()

    def test_format_filter_passed_to_pick_next_topic(self, tmp_path):
        mock_pick = MagicMock(return_value=_make_idea())
        _run_topic(tmp_path, format_filter="honest_review", pick_next_topic=mock_pick)
        call_args = mock_pick.call_args
        assert "honest_review" in call_args[0] or "honest_review" in str(call_args)

    def test_summary_contains_format_and_subject(self, tmp_path):
        idea = _make_idea()
        result, _ = _run_topic(tmp_path, idea_override=idea)
        assert result["format"] == "honest_review"
        assert result["subject"] == "GPT-5 released"

    def test_idea_json_written_to_run_dir(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        _run_topic(tmp_path, idea_override=idea, settings=ms)
        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        run_dir = tmp_path / "topic" / date_str / slug
        idea_file = run_dir / "idea.json"
        assert idea_file.exists()

    def test_run_dir_uses_format_and_subject_slug(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        result, _ = _run_topic(tmp_path, idea_override=idea, settings=ms)
        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        expected = str(tmp_path / "topic" / date_str / slug)
        assert result["run_dir"] == expected


# ─────────────────────────────────────────────────────────────────────────────
# Script tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunTopicPipelineScript:
    def test_cached_script_reused_no_generate(self, tmp_path):
        cached = _make_script(2)
        mock_load = MagicMock(return_value=cached)
        mock_generate = MagicMock()
        result, _ = _run_topic(
            tmp_path,
            _load_cached_script=mock_load,
            generate_topic_script=mock_generate,
        )
        mock_generate.assert_not_called()
        assert result["title"] == cached.title

    def test_cached_script_skips_agents(self, tmp_path):
        cached = _make_script(2)
        mock_ol = MagicMock()
        mock_cm = MagicMock()
        mock_pa = MagicMock()
        _run_topic(
            tmp_path,
            _load_cached_script=MagicMock(return_value=cached),
            apply_open_loops=mock_ol,
            apply_comment_magnet=mock_cm,
            audit_pacing=mock_pa,
        )
        mock_ol.assert_not_called()
        mock_cm.assert_not_called()
        mock_pa.assert_not_called()

    def test_fresh_script_triggers_generate(self, tmp_path):
        fresh = _make_script(4)
        mock_generate = MagicMock(return_value=fresh)
        result, _ = _run_topic(
            tmp_path,
            _load_cached_script=MagicMock(return_value=None),
            generate_topic_script=mock_generate,
        )
        mock_generate.assert_called_once()
        assert result["title"] == fresh.title

    def test_open_loops_applied_on_fresh_script(self, tmp_path):
        script = _make_script()
        ol_result = _make_ol_result(script, modified=False)
        mock_ol = MagicMock(return_value=ol_result)
        _run_topic(
            tmp_path,
            _load_cached_script=MagicMock(return_value=None),
            apply_open_loops=mock_ol,
        )
        mock_ol.assert_called_once()

    def test_open_loops_json_written_when_modified(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        script = _make_script()

        loop = MagicMock()
        loop.teaser = "Coming up..."
        loop.closing = "And now the answer"
        loop.target_scene_idx = 1

        ol_result = MagicMock()
        ol_result.script = script
        ol_result.modified = True
        ol_result.loops = [loop]

        _run_topic(
            tmp_path,
            idea_override=idea,
            settings=ms,
            _load_cached_script=MagicMock(return_value=None),
            apply_open_loops=MagicMock(return_value=ol_result),
        )
        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        run_dir = tmp_path / "topic" / date_str / slug
        assert (run_dir / "open_loops.json").exists()

    def test_open_loops_json_not_written_when_not_modified(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        script = _make_script()
        ol_result = _make_ol_result(script, modified=False)
        _run_topic(
            tmp_path,
            idea_override=idea,
            settings=ms,
            _load_cached_script=MagicMock(return_value=None),
            apply_open_loops=MagicMock(return_value=ol_result),
        )
        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        run_dir = tmp_path / "topic" / date_str / slug
        assert not (run_dir / "open_loops.json").exists()

    def test_comment_magnet_applied_on_fresh_script(self, tmp_path):
        script = _make_script()
        cm_result = _make_cm_result(script, modified=False)
        mock_cm = MagicMock(return_value=cm_result)
        _run_topic(
            tmp_path,
            _load_cached_script=MagicMock(return_value=None),
            apply_comment_magnet=mock_cm,
        )
        mock_cm.assert_called_once()

    def test_comment_magnet_json_written_when_modified(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        script = _make_script()
        cm_result = _make_cm_result(script, modified=True)
        _run_topic(
            tmp_path,
            idea_override=idea,
            settings=ms,
            _load_cached_script=MagicMock(return_value=None),
            apply_comment_magnet=MagicMock(return_value=cm_result),
        )
        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        run_dir = tmp_path / "topic" / date_str / slug
        assert (run_dir / "comment_magnet.json").exists()

    def test_comment_magnet_json_not_written_when_not_modified(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        script = _make_script()
        cm_result = _make_cm_result(script, modified=False)
        _run_topic(
            tmp_path,
            idea_override=idea,
            settings=ms,
            _load_cached_script=MagicMock(return_value=None),
            apply_comment_magnet=MagicMock(return_value=cm_result),
        )
        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        run_dir = tmp_path / "topic" / date_str / slug
        assert not (run_dir / "comment_magnet.json").exists()

    def test_pacing_audit_applied_on_fresh_script(self, tmp_path):
        script = _make_script()
        pa_result = _make_pa_result(script, modified=False)
        mock_pa = MagicMock(return_value=pa_result)
        _run_topic(
            tmp_path,
            _load_cached_script=MagicMock(return_value=None),
            audit_pacing=mock_pa,
        )
        mock_pa.assert_called_once()

    def test_pacing_audit_json_written_when_modified(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        script = _make_script()

        zone = MagicMock()
        zone.scene_indices = [1, 2]
        zone.interrupt_at = 1
        zone.interrupt_text = "Let me show you something"

        score = MagicMock()
        score.idx = 1
        score.score = 0.4
        score.signals = ["low_energy"]

        pa_result = MagicMock()
        pa_result.script = script
        pa_result.modified = True
        pa_result.flat_zones = [zone]
        pa_result.scores = [score]

        _run_topic(
            tmp_path,
            idea_override=idea,
            settings=ms,
            _load_cached_script=MagicMock(return_value=None),
            audit_pacing=MagicMock(return_value=pa_result),
        )
        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        run_dir = tmp_path / "topic" / date_str / slug
        assert (run_dir / "pacing_audit.json").exists()

    def test_pacing_audit_json_not_written_when_not_modified(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        script = _make_script()
        pa_result = _make_pa_result(script, modified=False)
        _run_topic(
            tmp_path,
            idea_override=idea,
            settings=ms,
            _load_cached_script=MagicMock(return_value=None),
            audit_pacing=MagicMock(return_value=pa_result),
        )
        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        run_dir = tmp_path / "topic" / date_str / slug
        assert not (run_dir / "pacing_audit.json").exists()

    def test_script_saved_after_fresh_generation(self, tmp_path):
        fresh = MagicMock(spec=VideoScript)
        fresh.title = "Generated Topic"
        fresh.scenes = [_make_scene(i) for i in range(3)]
        fresh.hook = "hook text"
        fresh.tags = ["ai"]
        ol_result = _make_ol_result(fresh)
        cm_result = _make_cm_result(fresh)
        pa_result = _make_pa_result(fresh)
        _run_topic(
            tmp_path,
            _load_cached_script=MagicMock(return_value=None),
            generate_topic_script=MagicMock(return_value=fresh),
            apply_open_loops=MagicMock(return_value=ol_result),
            apply_comment_magnet=MagicMock(return_value=cm_result),
            audit_pacing=MagicMock(return_value=pa_result),
        )
        assert fresh.save.call_count >= 1

    def test_summary_scenes_count(self, tmp_path):
        script = _make_script(5)
        result, _ = _run_topic(
            tmp_path,
            _load_cached_script=MagicMock(return_value=script),
        )
        assert result["scenes"] == 5


# ─────────────────────────────────────────────────────────────────────────────
# Voice tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunTopicPipelineVoice:
    def test_synthesize_called_when_no_audio_dir(self, tmp_path):
        mock_synth = MagicMock()
        _run_topic(tmp_path, synthesize_script=mock_synth)
        mock_synth.assert_called_once()

    def test_synthesize_called_when_audio_dir_has_fewer_mp3s(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        script = _make_script(3)

        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        run_dir = tmp_path / "topic" / date_str / slug
        run_dir.mkdir(parents=True)
        audio_dir = run_dir / "audio"
        audio_dir.mkdir()
        # Only 1 mp3 for 3-scene script — should re-synthesize
        (audio_dir / "scene_00.mp3").touch()

        mock_synth = MagicMock()
        _run_topic(
            tmp_path,
            idea_override=idea,
            settings=ms,
            _load_cached_script=MagicMock(return_value=script),
            synthesize_script=mock_synth,
        )
        mock_synth.assert_called_once()

    def test_cached_audio_reused_when_enough_mp3s(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        script = _make_script(2)

        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        run_dir = tmp_path / "topic" / date_str / slug
        run_dir.mkdir(parents=True)
        audio_dir = run_dir / "audio"
        audio_dir.mkdir()
        for i in range(2):
            (audio_dir / f"scene_{i:02d}.mp3").touch()

        mock_synth = MagicMock()
        mock_load_dur = MagicMock()
        _run_topic(
            tmp_path,
            idea_override=idea,
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
            s.duration_sec = 15
        result, _ = _run_topic(
            tmp_path,
            _load_cached_script=MagicMock(return_value=script),
        )
        assert result["total_duration_sec"] == 45


# ─────────────────────────────────────────────────────────────────────────────
# Subtitle tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunTopicPipelineSubtitles:
    def test_subtitle_exception_swallowed(self, tmp_path):
        mock_gen = MagicMock(side_effect=RuntimeError("Subtitle boom"))
        # Should not raise
        result, _ = _run_topic(tmp_path, generate_subtitles=mock_gen)
        assert "date" in result

    def test_subtitle_exception_does_not_set_status_failed(self, tmp_path):
        mock_gen = MagicMock(side_effect=RuntimeError("Subtitle boom"))
        result, _ = _run_topic(
            tmp_path,
            skip_upload=True,
            generate_subtitles=mock_gen,
        )
        assert result.get("status") != "failed"


# ─────────────────────────────────────────────────────────────────────────────
# Video tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunTopicPipelineVideo:
    def test_build_video_called_when_rebuild_needed(self, tmp_path):
        mock_build = MagicMock()
        _run_topic(
            tmp_path,
            _needs_video_rebuild=MagicMock(return_value=True),
            build_video=mock_build,
        )
        mock_build.assert_called_once()

    def test_build_video_not_called_when_cache_fresh(self, tmp_path):
        mock_build = MagicMock()
        _run_topic(
            tmp_path,
            _needs_video_rebuild=MagicMock(return_value=False),
            build_video=mock_build,
        )
        mock_build.assert_not_called()

    def test_build_video_receives_format_as_tool(self, tmp_path):
        mock_build = MagicMock()
        idea = _make_idea()
        idea.format = "hidden_gems"
        _run_topic(
            tmp_path,
            idea_override=idea,
            _needs_video_rebuild=MagicMock(return_value=True),
            build_video=mock_build,
        )
        call_kwargs = mock_build.call_args[1]
        assert call_kwargs.get("tool") == "hidden_gems"

    def test_digest_short_built_when_missing(self, tmp_path):
        mock_build_short = MagicMock()
        _run_topic(tmp_path, build_short=mock_build_short)
        mock_build_short.assert_called_once()

    def test_digest_short_not_rebuilt_when_cached(self, tmp_path):
        import datetime as dt
        ms = _make_settings(tmp_path)
        idea = _make_idea()
        date_str = dt.date.today().isoformat()
        slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
        run_dir = tmp_path / "topic" / date_str / slug
        run_dir.mkdir(parents=True)
        (run_dir / "shorts.mp4").touch()

        mock_build_short = MagicMock()
        _run_topic(
            tmp_path,
            idea_override=idea,
            settings=ms,
            build_short=mock_build_short,
        )
        mock_build_short.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Upload tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunTopicPipelineUpload:
    def test_skip_upload_sets_status_built_not_uploaded(self, tmp_path):
        result, _ = _run_topic(tmp_path, skip_upload=True)
        assert result["status"] == "built_not_uploaded"

    def test_skip_upload_does_not_call_publish_episode(self, tmp_path):
        mock_pub = MagicMock(return_value={"video_id": "x"})
        _run_topic(tmp_path, skip_upload=True, publish_episode=mock_pub)
        mock_pub.assert_not_called()

    def test_fresh_upload_calls_publish_episode(self, tmp_path):
        mock_pub = MagicMock(return_value={"video_id": "vid_fresh"})
        _run_topic(tmp_path, skip_upload=False, publish_episode=mock_pub)
        mock_pub.assert_called_once()

    def test_fresh_upload_sets_status_published(self, tmp_path):
        result, _ = _run_topic(
            tmp_path,
            skip_upload=False,
            publish_episode=MagicMock(return_value={"video_id": "vid_abc"}),
        )
        assert result["status"] == "published"

    def test_record_usage_called_with_video_id(self, tmp_path):
        mock_rec = MagicMock()
        _run_topic(
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
        _run_topic(
            tmp_path,
            skip_upload=False,
            publish_episode=MagicMock(return_value={"video_id": "vid_abc"}),
            record_thumbnail_usage=mock_rec,
        )
        mock_rec.assert_called_once()

    def test_publish_episode_ids_merged_into_summary(self, tmp_path):
        result, _ = _run_topic(
            tmp_path,
            skip_upload=False,
            publish_episode=MagicMock(return_value={"video_id": "v1", "playlist_id": "PL1"}),
        )
        assert result["video_id"] == "v1"
        assert result["playlist_id"] == "PL1"

    def test_record_usage_not_called_when_no_video_id(self, tmp_path):
        mock_rec = MagicMock()
        _run_topic(
            tmp_path,
            skip_upload=False,
            publish_episode=MagicMock(return_value={}),
            record_usage=mock_rec,
        )
        mock_rec.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Error handling tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunTopicPipelineErrors:
    def test_exception_does_not_reraise(self, tmp_path):
        # topic_main catches all exceptions and returns normally
        result, _ = _run_topic(
            tmp_path,
            generate_topic_script=MagicMock(side_effect=RuntimeError("generate fail")),
            _load_cached_script=MagicMock(return_value=None),
        )
        # Should not raise; check we got a dict back
        assert isinstance(result, dict)

    def test_exception_sets_status_failed(self, tmp_path):
        result, _ = _run_topic(
            tmp_path,
            generate_topic_script=MagicMock(side_effect=RuntimeError("generate fail")),
            _load_cached_script=MagicMock(return_value=None),
        )
        assert result["status"] == "failed"

    def test_exception_calls_notify_failure(self, tmp_path):
        mock_notify = MagicMock()
        _run_topic(
            tmp_path,
            generate_topic_script=MagicMock(side_effect=RuntimeError("fail")),
            _load_cached_script=MagicMock(return_value=None),
            notify_failure=mock_notify,
        )
        mock_notify.assert_called_once()

    def test_notify_failure_not_called_on_success(self, tmp_path):
        mock_notify = MagicMock()
        _run_topic(tmp_path, notify_failure=mock_notify)
        mock_notify.assert_not_called()

    def test_exception_does_not_call_notify_success(self, tmp_path):
        mock_notify = MagicMock()
        _run_topic(
            tmp_path,
            generate_topic_script=MagicMock(side_effect=RuntimeError("boom")),
            _load_cached_script=MagicMock(return_value=None),
            notify_success=mock_notify,
        )
        mock_notify.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Summary / notify_success tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunTopicPipelineSummary:
    def test_summary_contains_date(self, tmp_path):
        import datetime as dt
        result, _ = _run_topic(tmp_path)
        assert result["date"] == dt.date.today().isoformat()

    def test_summary_contains_format(self, tmp_path):
        result, _ = _run_topic(tmp_path)
        assert result["format"] == "honest_review"

    def test_summary_contains_subject(self, tmp_path):
        result, _ = _run_topic(tmp_path)
        assert result["subject"] == "GPT-5 released"

    def test_summary_contains_run_dir(self, tmp_path):
        result, _ = _run_topic(tmp_path)
        assert "run_dir" in result

    def test_summary_contains_title(self, tmp_path):
        result, _ = _run_topic(tmp_path)
        assert "title" in result

    def test_summary_contains_thumbnail_style(self, tmp_path):
        thumb = tmp_path / "thumbnail_bold.jpg"
        thumb.touch()
        result, _ = _run_topic(
            tmp_path,
            generate_thumbnail_variants=MagicMock(return_value=[thumb]),
            pick_thumbnail=MagicMock(return_value=thumb),
        )
        assert "thumbnail_style" in result

    def test_notify_success_called_on_happy_path(self, tmp_path):
        mock_notify = MagicMock()
        _run_topic(tmp_path, notify_success=mock_notify)
        mock_notify.assert_called_once()

    def test_notify_success_called_with_topic_tag(self, tmp_path):
        mock_notify = MagicMock()
        _run_topic(tmp_path, notify_success=mock_notify)
        call_args = mock_notify.call_args[0]
        assert "topic" in call_args

    def test_notify_success_not_called_on_exception(self, tmp_path):
        mock_notify = MagicMock()
        _run_topic(
            tmp_path,
            generate_topic_script=MagicMock(side_effect=RuntimeError("boom")),
            _load_cached_script=MagicMock(return_value=None),
            notify_success=mock_notify,
        )
        mock_notify.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# CLI tests
# ─────────────────────────────────────────────────────────────────────────────

class TestTopicCLI:
    def test_list_ideas_prints_queue(self, tmp_path, capsys):
        idea1 = _make_idea()
        idea2 = _make_idea()
        idea2.format = "hidden_gems"
        idea2.subject = "5 hidden tools"
        idea2.suggested_title = "5 AI tools you've never heard of"
        idea2.angle = "Underground picks"

        with ExitStack() as stack:
            stack.enter_context(patch("src.topic_main._load_ideas", return_value=[idea1, idea2]))
            stack.enter_context(patch("sys.argv", ["topic_main.py", "--list-ideas"]))
            main()

        captured = capsys.readouterr()
        assert "honest_review" in captured.out
        assert "GPT-5 released" in captured.out

    def test_list_ideas_empty_queue_prints_message(self, tmp_path, capsys):
        with ExitStack() as stack:
            stack.enter_context(patch("src.topic_main._load_ideas", return_value=[]))
            stack.enter_context(patch("sys.argv", ["topic_main.py", "--list-ideas"]))
            main()

        captured = capsys.readouterr()
        assert "empty" in captured.out.lower() or "Queue" in captured.out

    def test_refresh_ideas_calls_generator(self, tmp_path, capsys):
        new_idea = _make_idea()
        mock_gen = MagicMock(return_value=[new_idea])

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.topic_main.generate_next_topic_ideas", new=mock_gen)
            )
            stack.enter_context(patch("sys.argv", ["topic_main.py", "--refresh-ideas"]))
            main()

        mock_gen.assert_called_once()

    def test_refresh_ideas_uses_count_argument(self, tmp_path, capsys):
        mock_gen = MagicMock(return_value=[])

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.topic_main.generate_next_topic_ideas", new=mock_gen)
            )
            stack.enter_context(
                patch("sys.argv", ["topic_main.py", "--refresh-ideas", "--count", "5"])
            )
            main()

        call_kwargs = mock_gen.call_args
        assert call_kwargs[1].get("count") == 5 or call_kwargs[0][0] == 5

    def test_refresh_ideas_prints_generated_ideas(self, tmp_path, capsys):
        idea = _make_idea()
        with ExitStack() as stack:
            stack.enter_context(
                patch("src.topic_main.generate_next_topic_ideas", return_value=[idea])
            )
            stack.enter_context(patch("sys.argv", ["topic_main.py", "--refresh-ideas"]))
            main()

        captured = capsys.readouterr()
        assert "honest_review" in captured.out or "GPT-5" in captured.out

    def test_default_calls_run_topic_pipeline(self, tmp_path):
        mock_run = MagicMock(return_value={"status": "published"})

        with ExitStack() as stack:
            stack.enter_context(patch("src.topic_main.run_topic_pipeline", new=mock_run))
            stack.enter_context(patch("sys.argv", ["topic_main.py"]))
            main()

        mock_run.assert_called_once()

    def test_default_passes_format_filter(self, tmp_path):
        mock_run = MagicMock(return_value={"status": "published"})

        with ExitStack() as stack:
            stack.enter_context(patch("src.topic_main.run_topic_pipeline", new=mock_run))
            stack.enter_context(
                patch("sys.argv", ["topic_main.py", "--format", "honest_review"])
            )
            main()

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("format_filter") == "honest_review"

    def test_default_passes_skip_upload(self, tmp_path):
        mock_run = MagicMock(return_value={"status": "built_not_uploaded"})

        with ExitStack() as stack:
            stack.enter_context(patch("src.topic_main.run_topic_pipeline", new=mock_run))
            stack.enter_context(patch("sys.argv", ["topic_main.py", "--skip-upload"]))
            main()

        call_kwargs = mock_run.call_args[1]
        assert call_kwargs.get("skip_upload") is True
