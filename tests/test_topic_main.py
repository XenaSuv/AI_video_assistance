"""Tests for src/topic_main.py — run_topic_pipeline and CLI."""
from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, call, patch

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


# ── Factories ─────────────────────────────────────────────────────────────────

def _make_script(n_scenes: int = 3) -> MagicMock:
    scenes = []
    for i in range(n_scenes):
        sc = MagicMock()
        sc.idx = i
        sc.duration_sec = 30
        scenes.append(sc)
    script = MagicMock()
    script.title = "Test Topic Title"
    script.hook = "This is the hook text"
    script.scenes = scenes
    script.save = MagicMock()
    return script


def _make_idea() -> MagicMock:
    idea = MagicMock()
    idea.format = "honest_review"
    idea.subject = "GPT-5 released"
    idea.suggested_title = "GPT-5: My Honest Review"
    idea.angle = "tool review angle"
    idea.to_dict = MagicMock(return_value={
        "format": "honest_review",
        "subject": "GPT-5 released",
        "angle": "tool review angle",
        "suggested_title": "GPT-5: My Honest Review",
        "created": "2026-05-19",
    })
    return idea


def _make_settings(tmp_path: Path) -> MagicMock:
    ms = MagicMock()
    ms.output_dir = tmp_path / "output"
    ms.data_dir = tmp_path / "data"
    ms.source_dir = tmp_path / "source"
    ms.output_dir.mkdir(parents=True, exist_ok=True)
    ms.data_dir.mkdir(parents=True, exist_ok=True)
    return ms


def _make_ol_result(script: MagicMock, modified: bool = False) -> MagicMock:
    result = MagicMock()
    result.script = script
    result.modified = modified
    loop = MagicMock()
    loop.teaser = "teaser text"
    loop.closing = "closing text"
    loop.target_scene_idx = 1
    result.loops = [loop] if modified else []
    return result


def _make_cm_result(script: MagicMock, modified: bool = False) -> MagicMock:
    result = MagicMock()
    result.script = script
    result.modified = modified
    result.question = "Q?"
    result.inserted_at_scene = 2
    return result


def _make_pa_result(script: MagicMock, modified: bool = False) -> MagicMock:
    result = MagicMock()
    result.script = script
    result.modified = modified
    flat_zone = MagicMock()
    flat_zone.scene_indices = [1, 2]
    flat_zone.interrupt_at = 1
    flat_zone.interrupt_text = "interrupt text"
    score_item = MagicMock()
    score_item.idx = 0
    score_item.score = 0.7
    score_item.signals = ["signal_a"]
    result.flat_zones = [flat_zone] if modified else []
    result.scores = [score_item] if modified else []
    return result


def _run_topic(
    tmp_path: Path,
    *,
    idea_override=None,
    format_filter=None,
    skip_upload: bool = False,
    script: MagicMock | None = None,
    cached_script: MagicMock | None = None,
    ol_modified: bool = False,
    cm_modified: bool = False,
    pa_modified: bool = False,
    publish_return: dict | None = None,
    audio_dir_exists: bool = False,
    audio_mp3_count: int = 0,
    video_needs_rebuild: bool = True,
    short_exists: bool = False,
    subtitle_raises: Exception | None = None,
    generate_script_raises: Exception | None = None,
):
    """Run run_topic_pipeline with all external dependencies patched.

    Returns (result_dict, mocks_dict).
    """
    if script is None:
        script = _make_script()
    if publish_return is None:
        publish_return = {"video_id": "vid_abc123"}

    ms = _make_settings(tmp_path)

    # Determine idea for slug so we can set up paths properly
    import datetime as dt
    date_str = dt.date.today().isoformat()
    idea = idea_override if idea_override is not None else _make_idea()
    slug = idea.format + "_" + idea.subject[:30].replace(" ", "_").lower()
    run_dir = ms.output_dir / "topic" / date_str / slug
    run_dir.mkdir(parents=True, exist_ok=True)

    if audio_dir_exists:
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for i in range(audio_mp3_count):
            (audio_dir / f"scene_{i:02d}.mp3").touch()

    if short_exists:
        (run_dir / "shorts.mp4").touch()

    # Build agent result mocks
    ol_result = _make_ol_result(script, modified=ol_modified)
    cm_result = _make_cm_result(script, modified=cm_modified)
    pa_result = _make_pa_result(script, modified=pa_modified)

    thumb_path = run_dir / "thumbnail_bold.jpg"
    thumb_variants = [thumb_path]

    mocks = {}

    with ExitStack() as stack:
        # settings
        stack.enter_context(patch("src.topic_main.settings", ms))

        # pick_next_topic
        mock_pick = stack.enter_context(
            patch("src.topic_main.pick_next_topic", return_value=_make_idea())
        )
        mocks["pick_next_topic"] = mock_pick

        # _setup_logging
        mock_setup_logging = stack.enter_context(
            patch("src.topic_main._setup_logging")
        )
        mocks["_setup_logging"] = mock_setup_logging

        # _load_cached_script
        mock_load_cached = stack.enter_context(
            patch(
                "src.topic_main._load_cached_script",
                return_value=cached_script,
            )
        )
        mocks["_load_cached_script"] = mock_load_cached

        # generate_topic_script
        if generate_script_raises:
            mock_gen_script = stack.enter_context(
                patch(
                    "src.topic_main.generate_topic_script",
                    side_effect=generate_script_raises,
                )
            )
        else:
            mock_gen_script = stack.enter_context(
                patch("src.topic_main.generate_topic_script", return_value=script)
            )
        mocks["generate_topic_script"] = mock_gen_script

        # apply_open_loops
        mock_ol = stack.enter_context(
            patch("src.topic_main.apply_open_loops", return_value=ol_result)
        )
        mocks["apply_open_loops"] = mock_ol

        # apply_comment_magnet
        mock_cm = stack.enter_context(
            patch("src.topic_main.apply_comment_magnet", return_value=cm_result)
        )
        mocks["apply_comment_magnet"] = mock_cm

        # audit_pacing
        mock_pa = stack.enter_context(
            patch("src.topic_main.audit_pacing", return_value=pa_result)
        )
        mocks["audit_pacing"] = mock_pa

        # synthesize_script
        mock_synth = stack.enter_context(
            patch("src.topic_main.synthesize_script")
        )
        mocks["synthesize_script"] = mock_synth

        # _load_audio_durations
        mock_load_dur = stack.enter_context(
            patch("src.topic_main._load_audio_durations")
        )
        mocks["_load_audio_durations"] = mock_load_dur

        # generate_subtitles
        if subtitle_raises:
            mock_subtitles = stack.enter_context(
                patch(
                    "src.topic_main.generate_subtitles",
                    side_effect=subtitle_raises,
                )
            )
        else:
            mock_subtitles = stack.enter_context(
                patch(
                    "src.topic_main.generate_subtitles",
                    return_value=run_dir / "subtitles.srt",
                )
            )
        mocks["generate_subtitles"] = mock_subtitles

        # _needs_video_rebuild
        mock_needs_rebuild = stack.enter_context(
            patch("src.topic_main._needs_video_rebuild", return_value=video_needs_rebuild)
        )
        mocks["_needs_video_rebuild"] = mock_needs_rebuild

        # build_video
        mock_build_video = stack.enter_context(
            patch("src.topic_main.build_video")
        )
        mocks["build_video"] = mock_build_video

        # _get_shared_outro
        mock_outro = stack.enter_context(
            patch("src.topic_main._get_shared_outro", return_value=None)
        )
        mocks["_get_shared_outro"] = mock_outro

        # build_short
        mock_build_short = stack.enter_context(
            patch("src.topic_main.build_short")
        )
        mocks["build_short"] = mock_build_short

        # generate_thumbnail_variants
        mock_thumb_variants = stack.enter_context(
            patch(
                "src.topic_main.generate_thumbnail_variants",
                return_value=thumb_variants,
            )
        )
        mocks["generate_thumbnail_variants"] = mock_thumb_variants

        # pick_thumbnail
        mock_pick_thumb = stack.enter_context(
            patch("src.topic_main.pick_thumbnail", return_value=thumb_path)
        )
        mocks["pick_thumbnail"] = mock_pick_thumb

        # publish_episode
        mock_publish = stack.enter_context(
            patch("src.topic_main.publish_episode", return_value=publish_return)
        )
        mocks["publish_episode"] = mock_publish

        # record_usage
        mock_record_usage = stack.enter_context(
            patch("src.topic_main.record_usage")
        )
        mocks["record_usage"] = mock_record_usage

        # record_thumbnail_usage
        mock_record_thumb = stack.enter_context(
            patch("src.topic_main.record_thumbnail_usage")
        )
        mocks["record_thumbnail_usage"] = mock_record_thumb

        # notify_success
        mock_notify_success = stack.enter_context(
            patch("src.topic_main.notify_success")
        )
        mocks["notify_success"] = mock_notify_success

        # notify_failure
        mock_notify_failure = stack.enter_context(
            patch("src.topic_main.notify_failure")
        )
        mocks["notify_failure"] = mock_notify_failure

        from src.topic_main import run_topic_pipeline

        result = run_topic_pipeline(
            format_filter=format_filter,
            idea_override=idea_override,
            skip_upload=skip_upload,
        )

    return result, mocks


# ── Idea selection ─────────────────────────────────────────────────────────────

class TestRunTopicPipelineIdeaSelection:
    def test_idea_override_used_directly(self, tmp_path):
        """When idea_override is given, pick_next_topic is NOT called."""
        idea = _make_idea()
        result, mocks = _run_topic(tmp_path, idea_override=idea)
        mocks["pick_next_topic"].assert_not_called()

    def test_pick_next_topic_called_when_no_override(self, tmp_path):
        """When no idea_override, pick_next_topic() is called."""
        result, mocks = _run_topic(tmp_path, idea_override=None)
        mocks["pick_next_topic"].assert_called_once()

    def test_format_filter_passed_to_pick_next_topic(self, tmp_path):
        """format_filter value is forwarded to pick_next_topic."""
        result, mocks = _run_topic(tmp_path, format_filter="honest_review")
        mocks["pick_next_topic"].assert_called_once_with("honest_review")

    def test_summary_contains_format_and_subject(self, tmp_path):
        """Summary dict has format and subject keys from the idea."""
        idea = _make_idea()
        result, _ = _run_topic(tmp_path, idea_override=idea)
        assert result["format"] == "honest_review"
        assert result["subject"] == "GPT-5 released"

    def test_idea_json_written_to_run_dir(self, tmp_path):
        """idea.json is written into the run directory."""
        idea = _make_idea()
        result, _ = _run_topic(tmp_path, idea_override=idea)
        run_dir = Path(result["run_dir"])
        idea_path = run_dir / "idea.json"
        assert idea_path.exists()
        loaded = json.loads(idea_path.read_text())
        assert loaded["format"] == "honest_review"


# ── Script caching ────────────────────────────────────────────────────────────

class TestRunTopicPipelineScript:
    def test_cached_script_reused_no_generate(self, tmp_path):
        """When _load_cached_script returns a script, generate is not called."""
        cached = _make_script()
        result, mocks = _run_topic(tmp_path, cached_script=cached)
        mocks["generate_topic_script"].assert_not_called()

    def test_cached_script_no_open_loops_applied(self, tmp_path):
        """When using cached script, open-loop agent is NOT called."""
        cached = _make_script()
        result, mocks = _run_topic(tmp_path, cached_script=cached)
        mocks["apply_open_loops"].assert_not_called()

    def test_fresh_script_calls_generate(self, tmp_path):
        """Without cache, generate_topic_script is called once."""
        result, mocks = _run_topic(tmp_path, cached_script=None)
        mocks["generate_topic_script"].assert_called_once()

    def test_fresh_script_open_loops_called(self, tmp_path):
        """apply_open_loops is called for fresh scripts."""
        result, mocks = _run_topic(tmp_path, cached_script=None)
        mocks["apply_open_loops"].assert_called_once()

    def test_fresh_script_comment_magnet_called(self, tmp_path):
        """apply_comment_magnet is called for fresh scripts."""
        result, mocks = _run_topic(tmp_path, cached_script=None)
        mocks["apply_comment_magnet"].assert_called_once()

    def test_fresh_script_pacing_audit_called(self, tmp_path):
        """audit_pacing is called for fresh scripts."""
        result, mocks = _run_topic(tmp_path, cached_script=None)
        mocks["audit_pacing"].assert_called_once()

    def test_open_loops_json_written_when_modified(self, tmp_path):
        """When ol_result.modified=True, open_loops.json is written."""
        result, mocks = _run_topic(tmp_path, cached_script=None, ol_modified=True)
        run_dir = Path(result["run_dir"])
        assert (run_dir / "open_loops.json").exists()

    def test_open_loops_json_not_written_when_not_modified(self, tmp_path):
        """When ol_result.modified=False, open_loops.json is NOT written."""
        result, mocks = _run_topic(tmp_path, cached_script=None, ol_modified=False)
        run_dir = Path(result["run_dir"])
        assert not (run_dir / "open_loops.json").exists()

    def test_comment_magnet_json_written_when_modified(self, tmp_path):
        """When cm_result.modified=True, comment_magnet.json is written."""
        result, mocks = _run_topic(tmp_path, cached_script=None, cm_modified=True)
        run_dir = Path(result["run_dir"])
        assert (run_dir / "comment_magnet.json").exists()

    def test_comment_magnet_json_not_written_when_not_modified(self, tmp_path):
        """When cm_result.modified=False, comment_magnet.json is NOT written."""
        result, mocks = _run_topic(tmp_path, cached_script=None, cm_modified=False)
        run_dir = Path(result["run_dir"])
        assert not (run_dir / "comment_magnet.json").exists()

    def test_pacing_audit_json_written_when_modified(self, tmp_path):
        """When pa_result.modified=True, pacing_audit.json is written."""
        result, mocks = _run_topic(tmp_path, cached_script=None, pa_modified=True)
        run_dir = Path(result["run_dir"])
        assert (run_dir / "pacing_audit.json").exists()

    def test_pacing_audit_json_not_written_when_not_modified(self, tmp_path):
        """When pa_result.modified=False, pacing_audit.json is NOT written."""
        result, mocks = _run_topic(tmp_path, cached_script=None, pa_modified=False)
        run_dir = Path(result["run_dir"])
        assert not (run_dir / "pacing_audit.json").exists()

    def test_script_saved_after_fresh_generation(self, tmp_path):
        """script.save() is called after the fresh generation + agent passes."""
        script = _make_script()
        result, mocks = _run_topic(tmp_path, cached_script=None, script=script)
        script.save.assert_called()

    def test_summary_contains_title_and_scene_count(self, tmp_path):
        """Summary has title and scenes keys after script step."""
        script = _make_script(n_scenes=5)
        result, _ = _run_topic(tmp_path, cached_script=None, script=script)
        assert result["title"] == "Test Topic Title"
        assert result["scenes"] == 5


# ── Voice synthesis ───────────────────────────────────────────────────────────

class TestRunTopicPipelineVoice:
    def test_synthesize_called_when_no_audio_dir(self, tmp_path):
        """synthesize_script is called when audio/ dir doesn't exist."""
        result, mocks = _run_topic(tmp_path, audio_dir_exists=False)
        mocks["synthesize_script"].assert_called_once()

    def test_synthesize_called_when_insufficient_mp3s(self, tmp_path):
        """synthesize_script is called when mp3 count < scene count."""
        # 3-scene script but only 1 mp3
        result, mocks = _run_topic(
            tmp_path,
            audio_dir_exists=True,
            audio_mp3_count=1,
        )
        mocks["synthesize_script"].assert_called_once()

    def test_load_durations_called_when_audio_cached(self, tmp_path):
        """_load_audio_durations is called when all audio is already present."""
        script = _make_script(n_scenes=2)
        result, mocks = _run_topic(
            tmp_path,
            script=script,
            audio_dir_exists=True,
            audio_mp3_count=2,
        )
        mocks["_load_audio_durations"].assert_called_once()
        mocks["synthesize_script"].assert_not_called()

    def test_total_duration_in_summary(self, tmp_path):
        """summary contains total_duration_sec from summing scene durations."""
        script = _make_script(n_scenes=3)  # each scene has duration_sec=30
        result, _ = _run_topic(tmp_path, script=script)
        assert result["total_duration_sec"] == 90


# ── Subtitles ─────────────────────────────────────────────────────────────────

class TestRunTopicPipelineSubtitles:
    def test_subtitle_exception_swallowed(self, tmp_path):
        """A subtitle generation failure is non-fatal; pipeline completes normally."""
        result, mocks = _run_topic(
            tmp_path,
            subtitle_raises=RuntimeError("whisper down"),
        )
        # Pipeline must have completed without re-raising
        assert "status" in result
        assert result["status"] != "failed"

    def test_subtitle_generate_called(self, tmp_path):
        """generate_subtitles is invoked during the pipeline."""
        result, mocks = _run_topic(tmp_path)
        mocks["generate_subtitles"].assert_called_once()


# ── Video ─────────────────────────────────────────────────────────────────────

class TestRunTopicPipelineVideo:
    def test_build_video_called_when_rebuild_needed(self, tmp_path):
        """build_video() is called when _needs_video_rebuild returns True."""
        result, mocks = _run_topic(tmp_path, video_needs_rebuild=True)
        mocks["build_video"].assert_called_once()

    def test_build_video_skipped_when_no_rebuild(self, tmp_path):
        """build_video() is NOT called when _needs_video_rebuild returns False."""
        result, mocks = _run_topic(tmp_path, video_needs_rebuild=False)
        mocks["build_video"].assert_not_called()

    def test_build_short_called_when_no_short_exists(self, tmp_path):
        """build_short() is called when shorts.mp4 doesn't exist."""
        result, mocks = _run_topic(tmp_path, short_exists=False)
        mocks["build_short"].assert_called_once()

    def test_build_short_skipped_when_short_exists(self, tmp_path):
        """build_short() is NOT called when shorts.mp4 already exists."""
        result, mocks = _run_topic(tmp_path, short_exists=True)
        mocks["build_short"].assert_not_called()


# ── Upload ────────────────────────────────────────────────────────────────────

class TestRunTopicPipelineUpload:
    def test_skip_upload_sets_status_built_not_uploaded(self, tmp_path):
        """skip_upload=True → status is 'built_not_uploaded'."""
        result, mocks = _run_topic(tmp_path, skip_upload=True)
        assert result["status"] == "built_not_uploaded"

    def test_skip_upload_does_not_call_publish_episode(self, tmp_path):
        """skip_upload=True → publish_episode is NOT called."""
        result, mocks = _run_topic(tmp_path, skip_upload=True)
        mocks["publish_episode"].assert_not_called()

    def test_fresh_upload_sets_status_published(self, tmp_path):
        """Normal upload path → status is 'published'."""
        result, mocks = _run_topic(
            tmp_path,
            skip_upload=False,
            publish_return={"video_id": "vid_xyz"},
        )
        assert result["status"] == "published"

    def test_fresh_upload_calls_publish_episode(self, tmp_path):
        """Normal upload path → publish_episode is called once."""
        result, mocks = _run_topic(tmp_path, skip_upload=False)
        mocks["publish_episode"].assert_called_once()

    def test_record_usage_called_with_video_id(self, tmp_path):
        """record_usage is called and receives the video_id."""
        result, mocks = _run_topic(
            tmp_path,
            skip_upload=False,
            publish_return={"video_id": "vid_rec_test"},
        )
        mocks["record_usage"].assert_called_once()
        call_args = mocks["record_usage"].call_args[0]
        assert "vid_rec_test" in call_args

    def test_record_thumbnail_usage_called(self, tmp_path):
        """record_thumbnail_usage is called after upload."""
        result, mocks = _run_topic(tmp_path, skip_upload=False)
        mocks["record_thumbnail_usage"].assert_called_once()

    def test_record_usage_not_called_when_no_video_id(self, tmp_path):
        """record_usage is NOT called when publish_episode returns no video_id."""
        result, mocks = _run_topic(
            tmp_path,
            skip_upload=False,
            publish_return={"playlist_id": "PL1"},  # no video_id key
        )
        mocks["record_usage"].assert_not_called()

    def test_publish_ids_merged_into_summary(self, tmp_path):
        """IDs returned by publish_episode are merged into summary."""
        result, mocks = _run_topic(
            tmp_path,
            skip_upload=False,
            publish_return={"video_id": "v1", "playlist_id": "PL99"},
        )
        assert result["video_id"] == "v1"
        assert result["playlist_id"] == "PL99"


# ── Error handling ─────────────────────────────────────────────────────────────

class TestRunTopicPipelineErrors:
    def test_exception_does_not_reraise(self, tmp_path):
        """A fatal exception inside the pipeline does NOT propagate to caller."""
        result, mocks = _run_topic(
            tmp_path,
            generate_script_raises=RuntimeError("script boom"),
        )
        # If we reach here, the exception was swallowed — that's the assertion
        assert result is not None

    def test_exception_sets_status_failed(self, tmp_path):
        """When pipeline raises, returned dict has status='failed'."""
        result, mocks = _run_topic(
            tmp_path,
            generate_script_raises=RuntimeError("script boom"),
        )
        assert result["status"] == "failed"

    def test_notify_failure_called_on_exception(self, tmp_path):
        """notify_failure() is called when the pipeline catches an exception."""
        result, mocks = _run_topic(
            tmp_path,
            generate_script_raises=RuntimeError("script boom"),
        )
        mocks["notify_failure"].assert_called_once()

    def test_notify_success_not_called_on_failure(self, tmp_path):
        """notify_success() is NOT called when the pipeline fails."""
        result, mocks = _run_topic(
            tmp_path,
            generate_script_raises=RuntimeError("script boom"),
        )
        mocks["notify_success"].assert_not_called()

    def test_summary_preserves_date_on_failure(self, tmp_path):
        """Even on failure the summary has the 'date' key."""
        result, mocks = _run_topic(
            tmp_path,
            generate_script_raises=RuntimeError("boom"),
        )
        assert "date" in result


# ── Happy path summary / notify ───────────────────────────────────────────────

class TestRunTopicPipelineSummary:
    def test_notify_success_called_on_happy_path(self, tmp_path):
        """notify_success() is called when everything succeeds."""
        result, mocks = _run_topic(tmp_path, skip_upload=True)
        mocks["notify_success"].assert_called_once()

    def test_notify_success_receives_topic_pipeline_name(self, tmp_path):
        """notify_success is called with 'topic' as pipeline name."""
        result, mocks = _run_topic(tmp_path, skip_upload=True)
        args = mocks["notify_success"].call_args[0]
        assert "topic" in args

    def test_summary_has_run_dir_key(self, tmp_path):
        """Summary always contains run_dir."""
        result, _ = _run_topic(tmp_path, skip_upload=True)
        assert "run_dir" in result

    def test_thumbnail_style_in_summary(self, tmp_path):
        """summary['thumbnail_style'] is set from picked thumbnail stem."""
        result, _ = _run_topic(tmp_path, skip_upload=True)
        assert "thumbnail_style" in result

    def test_summary_has_date_key(self, tmp_path):
        """Summary contains today's ISO date."""
        import datetime as dt
        result, _ = _run_topic(tmp_path, skip_upload=True)
        assert result["date"] == dt.date.today().isoformat()


# ── CLI ───────────────────────────────────────────────────────────────────────

class TestTopicCLI:
    def test_list_ideas_prints_queue(self, tmp_path, capsys):
        """--list-ideas calls _load_ideas and prints each idea."""
        idea1 = _make_idea()
        idea2 = _make_idea()
        idea2.suggested_title = "Claude 4: Worth It?"
        idea2.subject = "Claude 4"
        idea2.format = "hidden_gems"

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.topic_main._load_ideas", return_value=[idea1, idea2])
            )
            stack.enter_context(patch("sys.argv", ["topic_main.py", "--list-ideas"]))
            from src.topic_main import main
            main()

        captured = capsys.readouterr()
        assert "GPT-5: My Honest Review" in captured.out
        assert "Claude 4: Worth It?" in captured.out

    def test_list_ideas_empty_queue_prints_message(self, tmp_path, capsys):
        """--list-ideas with empty queue prints a helpful message."""
        with ExitStack() as stack:
            stack.enter_context(
                patch("src.topic_main._load_ideas", return_value=[])
            )
            stack.enter_context(patch("sys.argv", ["topic_main.py", "--list-ideas"]))
            from src.topic_main import main
            main()

        captured = capsys.readouterr()
        assert "empty" in captured.out.lower() or "refresh" in captured.out.lower()

    def test_refresh_ideas_calls_generator(self, tmp_path, capsys):
        """--refresh-ideas calls generate_next_topic_ideas."""
        new_idea = _make_idea()
        with ExitStack() as stack:
            mock_gen = stack.enter_context(
                patch(
                    "src.topic_main.generate_next_topic_ideas",
                    return_value=[new_idea],
                )
            )
            stack.enter_context(patch("sys.argv", ["topic_main.py", "--refresh-ideas"]))
            from src.topic_main import main
            main()

        mock_gen.assert_called_once_with(count=8)

    def test_refresh_ideas_custom_count(self, tmp_path, capsys):
        """--refresh-ideas --count N passes N to generator."""
        new_idea = _make_idea()
        with ExitStack() as stack:
            mock_gen = stack.enter_context(
                patch(
                    "src.topic_main.generate_next_topic_ideas",
                    return_value=[new_idea],
                )
            )
            stack.enter_context(
                patch("sys.argv", ["topic_main.py", "--refresh-ideas", "--count", "15"])
            )
            from src.topic_main import main
            main()

        mock_gen.assert_called_once_with(count=15)

    def test_main_default_calls_run_topic_pipeline(self, tmp_path):
        """Default invocation calls run_topic_pipeline."""
        with ExitStack() as stack:
            mock_run = stack.enter_context(
                patch("src.topic_main.run_topic_pipeline", return_value={})
            )
            stack.enter_context(patch("sys.argv", ["topic_main.py"]))
            from src.topic_main import main
            main()

        mock_run.assert_called_once()

    def test_main_skip_upload_flag_passed(self, tmp_path):
        """--skip-upload flag is forwarded to run_topic_pipeline."""
        with ExitStack() as stack:
            mock_run = stack.enter_context(
                patch("src.topic_main.run_topic_pipeline", return_value={})
            )
            stack.enter_context(
                patch("sys.argv", ["topic_main.py", "--skip-upload"])
            )
            from src.topic_main import main
            main()

        call_kwargs = mock_run.call_args.kwargs if mock_run.call_args.kwargs else {}
        assert call_kwargs.get("skip_upload") is True

    def test_main_format_filter_passed(self, tmp_path):
        """--format flag is forwarded as format_filter to run_topic_pipeline."""
        with ExitStack() as stack:
            mock_run = stack.enter_context(
                patch("src.topic_main.run_topic_pipeline", return_value={})
            )
            stack.enter_context(
                patch("sys.argv", ["topic_main.py", "--format", "honest_review"])
            )
            from src.topic_main import main
            main()

        call_kwargs = mock_run.call_args.kwargs if mock_run.call_args.kwargs else {}
        assert call_kwargs.get("format_filter") == "honest_review"
