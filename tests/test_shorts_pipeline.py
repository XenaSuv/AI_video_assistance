"""Tests for src/shorts_pipeline.py — primary Shorts production engine.

Patching strategy
-----------------
All imports in shorts_pipeline.py are either module-level or lazy (inside a
function body).  Module-level imports can be patched via
``src.shorts_pipeline.<name>``; lazy imports must be patched at their source
module (e.g. ``src.avatar_engine.AvatarEngine``) because the ``from X import Y``
inside a function binds a local name that is looked up fresh on each call.

Module-level:  ShortsEngineV2, ShortsExperimentEngine, scrape_all,
               pick_viral_news, notify_success, notify_failure
Lazy:          ElevenLabs (elevenlabs), AvatarEngine (src.avatar_engine),
               ffmpeg_utils (src.ffmpeg_utils.*), fetch_broll_for_short
               (src.broll_fetcher), upload_short (src.youtube_uploader),
               HookPatternEngine (src.hook_pattern_engine),
               ShortsExperiment (src.shorts_experiment_engine)
"""
from __future__ import annotations

import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy deps that can't be installed in the test environment.
# Must be set before any project module is imported that lazily imports them.
sys.modules.setdefault("elevenlabs", MagicMock())
sys.modules.setdefault("src.youtube_uploader", MagicMock())

from src.shorts_engine_v2 import ShortResult, ShortScript
from src.shorts_pipeline import (
    ShortsPipeline,
    _save_summary,
    _story_to_editorial,
    _upload,
    render_short,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _script(
    short_id: str = "sid-001",
    topic: str = "GPT-5 launch",
    hook_type: str = "conflict",
    hook: str = "OpenAI just broke everything",
    core: str = "GPT-5 scores 95% on every benchmark.",
    twist: str = "But it costs $200 per month.",
    ending: str = "Is it worth it?",
    style: str = "fast_cut",
    persona: str = "direct",
) -> ShortScript:
    return ShortScript(
        short_id=short_id, topic=topic, hook_type=hook_type,
        hook=hook, core=core, twist=twist, ending=ending,
        style=style, persona=persona,
    )


def _mock_settings(**overrides):
    ms = MagicMock()
    ms.heygen_enabled = False
    ms.elevenlabs_api_key = "test-key"
    ms.elevenlabs_voice_id = "voice-id"
    ms.elevenlabs_model = "eleven_multilingual_v2"
    ms.youtube_client_secrets = Path("/fake/secrets.json")
    ms.youtube_token_file = Path("/fake/token.pickle")
    ms.data_dir = Path("/fake/data")
    ms.output_dir = Path("/fake/output")
    ms.background_music_path = ""
    ms.shorts_music_volume = 0.1
    for k, v in overrides.items():
        setattr(ms, k, v)
    return ms


# ── _story_to_editorial ───────────────────────────────────────────────────────

class TestStoryToEditorial:
    def test_topic_uses_first_four_words(self):
        story = {"title": "Word1 Word2 Word3 Word4 Word5 Word6", "summary": "s"}
        assert _story_to_editorial(story)["topic"] == "Word1 Word2 Word3 Word4"

    def test_topic_is_full_title_when_fewer_than_four_words(self):
        story = {"title": "AI wins", "summary": "s"}
        assert _story_to_editorial(story)["topic"] == "AI wins"

    def test_empty_title_falls_back_to_ai(self):
        assert _story_to_editorial({"title": "", "summary": "s"})["topic"] == "AI"

    def test_missing_title_falls_back_to_ai(self):
        assert _story_to_editorial({"summary": "s"})["topic"] == "AI"

    def test_summary_becomes_core_fact(self):
        story = {"title": "T", "summary": "The model scored 95%."}
        assert _story_to_editorial(story)["core_fact"] == "The model scored 95%."

    def test_description_used_when_no_summary(self):
        story = {"title": "T", "description": "Desc text"}
        assert _story_to_editorial(story)["core_fact"] == "Desc text"

    def test_title_used_as_fallback_core_fact(self):
        story = {"title": "Only title here"}
        assert _story_to_editorial(story)["core_fact"] == "Only title here"

    def test_angle_is_full_title(self):
        story = {"title": "Full title string", "summary": "s"}
        assert _story_to_editorial(story)["angle"] == "Full title string"


# ── _synthesize_short_audio ───────────────────────────────────────────────────

class TestSynthesizeShortAudio:
    """ElevenLabs is a lazy import — patch at elevenlabs.ElevenLabs."""

    def _call(self, script, tmp_path, *, chunks=None, api_error=None):
        from src.shorts_pipeline import _synthesize_short_audio
        mock_client = MagicMock()
        if api_error:
            mock_client.text_to_speech.convert.side_effect = api_error
        else:
            mock_client.text_to_speech.convert.return_value = chunks or [b"audio"]
        mock_cls = MagicMock(return_value=mock_client)
        with (
            patch("src.shorts_pipeline.settings", _mock_settings()),
            patch("elevenlabs.ElevenLabs", mock_cls),
        ):
            return _synthesize_short_audio(script, tmp_path), mock_client

    def test_returns_cached_file_without_calling_api(self, tmp_path):
        s = _script()
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        cached = audio_dir / f"{s.short_id}.mp3"
        cached.write_bytes(b"audio")
        result, mock_client = self._call(s, tmp_path)
        assert result == cached
        mock_client.text_to_speech.convert.assert_not_called()

    def test_creates_audio_dir(self, tmp_path):
        self._call(_script(), tmp_path, chunks=[b"x"])
        assert (tmp_path / "audio").is_dir()

    def test_writes_chunks_to_mp3(self, tmp_path):
        result, _ = self._call(_script(), tmp_path, chunks=[b"aa", b"bb", b""])
        assert result is not None
        assert result.read_bytes() == b"aabb"

    def test_returns_none_on_api_failure(self, tmp_path):
        result, _ = self._call(_script(), tmp_path, api_error=RuntimeError("quota"))
        assert result is None


# ── _render_avatar_short ──────────────────────────────────────────────────────

class TestRenderAvatarShort:
    """AvatarEngine is a lazy import — patch at src.avatar_engine.AvatarEngine."""

    def _call(self, script, audio_path, tmp_path, *, heygen_enabled=False, clip=None, error=None):
        from src.shorts_pipeline import _render_avatar_short
        mock_engine = MagicMock()
        if error:
            mock_engine.render.side_effect = error
        else:
            mock_engine.render.return_value = clip
        with (
            patch("src.shorts_pipeline.settings", _mock_settings(heygen_enabled=heygen_enabled)),
            patch("src.avatar_engine.AvatarEngine", return_value=mock_engine),
        ):
            return _render_avatar_short(script, audio_path, tmp_path)

    def test_returns_none_when_heygen_disabled(self, tmp_path):
        assert self._call(_script(), None, tmp_path, heygen_enabled=False) is None

    def test_calls_avatar_engine_when_enabled(self, tmp_path):
        fake_clip = tmp_path / "avatar.mp4"
        fake_clip.touch()
        result = self._call(_script(), None, tmp_path, heygen_enabled=True, clip=fake_clip)
        assert result == fake_clip

    def test_returns_none_on_avatar_engine_error(self, tmp_path):
        result = self._call(
            _script(), None, tmp_path,
            heygen_enabled=True, error=RuntimeError("api down"),
        )
        assert result is None


# ── _render_text_short ────────────────────────────────────────────────────────

class TestRenderTextShort:
    """ffmpeg_utils and fetch_broll_for_short are lazy imports.
    Patch at src.ffmpeg_utils.* and src.broll_fetcher.fetch_broll_for_short.
    shutil.copy2 is also patched to avoid real filesystem copies.
    """

    def _ctx(self, tmp_path, *, broll=None, music_path="", audio_exists=False):
        """Return an ExitStack with all external I/O mocked."""
        ms = _mock_settings(background_music_path=music_path)
        stack = ExitStack()
        stack.enter_context(patch("src.shorts_pipeline.settings", ms))
        stack.enter_context(patch("src.ffmpeg_utils.duration", return_value=25.0))
        mocks = {}
        mocks["burn"] = stack.enter_context(
            patch("src.ffmpeg_utils.burn_captions",
                  side_effect=lambda base, text, out, **kw: out)
        )
        mocks["black"] = stack.enter_context(patch("src.ffmpeg_utils.black_clip"))
        mocks["av"] = stack.enter_context(
            patch("src.ffmpeg_utils.merge_av",
                  side_effect=lambda v, a, out: out)
        )
        mocks["music"] = stack.enter_context(
            patch("src.ffmpeg_utils.mix_music",
                  side_effect=lambda v, m, out, **kw: out)
        )
        stack.enter_context(
            patch("src.broll_fetcher.fetch_broll_for_short", return_value=broll)
        )
        stack.enter_context(patch("shutil.copy2"))
        return stack, mocks

    def test_returns_cached_file_without_rendering(self, tmp_path):
        from src.shorts_pipeline import _render_text_short
        s = _script()
        out = tmp_path / f"text_{s.short_id}.mp4"
        out.touch()
        stack, mocks = self._ctx(tmp_path)
        with stack:
            result = _render_text_short(s, None, tmp_path)
        assert result == out
        mocks["burn"].assert_not_called()

    def test_uses_dark_background_when_no_broll(self, tmp_path):
        from src.shorts_pipeline import _render_text_short
        s = _script()
        stack, mocks = self._ctx(tmp_path, broll=None)
        with stack:
            _render_text_short(s, None, tmp_path)
        mocks["black"].assert_called_once()

    def test_skips_black_clip_when_broll_available(self, tmp_path):
        from src.shorts_pipeline import _render_text_short
        s = _script()
        fake_broll = tmp_path / "broll.mp4"
        fake_broll.touch()
        stack, mocks = self._ctx(tmp_path, broll=fake_broll)
        with stack:
            _render_text_short(s, None, tmp_path)
        mocks["black"].assert_not_called()

    def test_merges_audio_when_path_exists(self, tmp_path):
        from src.shorts_pipeline import _render_text_short
        s = _script()
        audio = tmp_path / "narration.mp3"
        audio.write_bytes(b"audio")
        stack, mocks = self._ctx(tmp_path, broll=None)
        with stack:
            _render_text_short(s, audio, tmp_path)
        mocks["av"].assert_called_once()

    def test_skips_audio_merge_when_no_audio(self, tmp_path):
        from src.shorts_pipeline import _render_text_short
        s = _script()
        stack, mocks = self._ctx(tmp_path, broll=None)
        with stack:
            _render_text_short(s, None, tmp_path)
        mocks["av"].assert_not_called()

    def test_skips_music_when_path_not_set(self, tmp_path):
        from src.shorts_pipeline import _render_text_short
        s = _script()
        stack, mocks = self._ctx(tmp_path, broll=None, music_path="")
        with stack:
            _render_text_short(s, None, tmp_path)
        mocks["music"].assert_not_called()

    def test_mixes_music_when_path_exists(self, tmp_path):
        from src.shorts_pipeline import _render_text_short
        s = _script()
        music = tmp_path / "bg.mp3"
        music.write_bytes(b"music")
        ms = _mock_settings(
            background_music_path=str(music),
            source_dir=tmp_path,
        )
        stack = ExitStack()
        stack.enter_context(patch("src.shorts_pipeline.settings", ms))
        stack.enter_context(patch("src.ffmpeg_utils.duration", return_value=25.0))
        stack.enter_context(patch("src.ffmpeg_utils.burn_captions",
                                  side_effect=lambda b, t, out, **kw: out))
        mock_music = stack.enter_context(patch("src.ffmpeg_utils.mix_music",
                                               side_effect=lambda v, m, out, **kw: out))
        stack.enter_context(patch("src.ffmpeg_utils.black_clip"))
        stack.enter_context(patch("src.ffmpeg_utils.merge_av",
                                  side_effect=lambda v, a, out: out))
        stack.enter_context(patch("src.broll_fetcher.fetch_broll_for_short", return_value=None))
        stack.enter_context(patch("shutil.copy2"))
        with stack:
            _render_text_short(s, None, tmp_path)
        mock_music.assert_called_once()

    def test_returns_out_path(self, tmp_path):
        from src.shorts_pipeline import _render_text_short
        s = _script()
        stack, _ = self._ctx(tmp_path, broll=None)
        with stack:
            result = _render_text_short(s, None, tmp_path)
        assert result == tmp_path / f"text_{s.short_id}.mp4"


# ── render_short ──────────────────────────────────────────────────────────────

class TestRenderShort:
    def test_returns_avatar_when_available(self, tmp_path):
        fake_avatar = tmp_path / "avatar.mp4"
        with (
            patch("src.shorts_pipeline._synthesize_short_audio", return_value=None),
            patch("src.shorts_pipeline._render_avatar_short", return_value=fake_avatar),
            patch("src.shorts_pipeline._render_text_short") as mock_text,
        ):
            result = render_short(_script(), tmp_path)
        assert result == fake_avatar
        mock_text.assert_not_called()

    def test_falls_back_to_text_short_when_no_avatar(self, tmp_path):
        fake_text = tmp_path / "text.mp4"
        with (
            patch("src.shorts_pipeline._synthesize_short_audio", return_value=None),
            patch("src.shorts_pipeline._render_avatar_short", return_value=None),
            patch("src.shorts_pipeline._render_text_short", return_value=fake_text),
        ):
            result = render_short(_script(), tmp_path)
        assert result == fake_text

    def test_returns_none_on_total_render_failure(self, tmp_path):
        with (
            patch("src.shorts_pipeline._synthesize_short_audio", return_value=None),
            patch("src.shorts_pipeline._render_avatar_short", return_value=None),
            patch("src.shorts_pipeline._render_text_short",
                  side_effect=RuntimeError("ffmpeg died")),
        ):
            result = render_short(_script(), tmp_path)
        assert result is None

    def test_audio_path_forwarded_to_text_renderer(self, tmp_path):
        fake_audio = tmp_path / "narr.mp3"
        fake_text = tmp_path / "text.mp4"
        with (
            patch("src.shorts_pipeline._synthesize_short_audio", return_value=fake_audio),
            patch("src.shorts_pipeline._render_avatar_short", return_value=None),
            patch("src.shorts_pipeline._render_text_short", return_value=fake_text) as mock_text,
        ):
            render_short(_script(), tmp_path)
        _, call_audio, _ = mock_text.call_args.args
        assert call_audio == fake_audio


# ── _upload ───────────────────────────────────────────────────────────────────

class TestUpload:
    """upload_short is a lazy import — patch at src.youtube_uploader.upload_short."""

    def _call(self, script, video_path, *, video_id="yt-abc123", error=None):
        mock_fn = MagicMock(return_value=video_id)
        if error:
            mock_fn.side_effect = error
        with (
            patch("src.shorts_pipeline.settings", _mock_settings()),
            patch("src.youtube_uploader.upload_short", mock_fn),
        ):
            return _upload(script, video_path), mock_fn

    def test_returns_video_id_on_success(self, tmp_path):
        video = tmp_path / "short.mp4"
        video.touch()
        result, _ = self._call(_script(), video)
        assert result == "yt-abc123"

    def test_title_truncated_to_95_chars_with_ellipsis(self, tmp_path):
        s = _script(hook="A" * 100)
        video = tmp_path / "short.mp4"
        video.touch()
        _, mock_fn = self._call(s, video)
        title = mock_fn.call_args.kwargs["title"]
        assert len(title) <= 96
        assert title.endswith("…")

    def test_short_title_not_truncated(self, tmp_path):
        s = _script(hook="Short hook")
        video = tmp_path / "short.mp4"
        video.touch()
        _, mock_fn = self._call(s, video)
        assert mock_fn.call_args.kwargs["title"] == "Short hook"

    def test_returns_none_on_upload_failure(self, tmp_path):
        video = tmp_path / "short.mp4"
        video.touch()
        result, _ = self._call(_script(), video, error=RuntimeError("auth failed"))
        assert result is None

    def test_description_contains_hook_type_hashtag(self, tmp_path):
        s = _script(hook_type="curiosity")
        video = tmp_path / "short.mp4"
        video.touch()
        _, mock_fn = self._call(s, video)
        assert "#curiosity" in mock_fn.call_args.kwargs["description"]


# ── _save_summary ─────────────────────────────────────────────────────────────

class TestSaveSummary:
    def test_writes_json_file(self, tmp_path):
        _save_summary(tmp_path, {"status": "done", "uploaded": 3})
        written = json.loads((tmp_path / "shorts_summary.json").read_text())
        assert written["status"] == "done"
        assert written["uploaded"] == 3

    def test_non_fatal_on_write_error(self):
        _save_summary(Path("/nonexistent/path"), {"x": 1})  # must not raise


# ── ShortsPipeline._get_story ─────────────────────────────────────────────────

class TestGetStory:
    def test_override_returns_dict_directly(self):
        result = ShortsPipeline()._get_story("Custom story title")
        assert result == {"title": "Custom story title",
                          "source": "manual", "summary": "Custom story title"}

    def test_scrapes_and_returns_top_viral_story(self):
        item = MagicMock(title="Top AI story", source="OpenAI", summary="Details", description="")
        with (
            patch("src.shorts_pipeline.scrape_all", return_value=[item]),
            patch("src.shorts_pipeline.pick_viral_news", return_value=[item]),
        ):
            result = ShortsPipeline()._get_story(None)
        assert result["title"] == "Top AI story"

    def test_returns_first_news_when_viral_empty(self):
        item = MagicMock(title="Fallback", source="Src", summary="s", description="")
        with (
            patch("src.shorts_pipeline.scrape_all", return_value=[item]),
            patch("src.shorts_pipeline.pick_viral_news", return_value=[]),
        ):
            result = ShortsPipeline()._get_story(None)
        assert result["title"] == "Fallback"

    def test_returns_none_when_no_news(self):
        with (
            patch("src.shorts_pipeline.scrape_all", return_value=[]),
            patch("src.shorts_pipeline.pick_viral_news", return_value=[]),
        ):
            assert ShortsPipeline()._get_story(None) is None

    def test_returns_none_on_scrape_exception(self):
        with patch("src.shorts_pipeline.scrape_all", side_effect=ConnectionError("network")):
            assert ShortsPipeline()._get_story(None) is None


# ── ShortsPipeline.run ────────────────────────────────────────────────────────

class TestShortsPipelineRun:
    def _scripts(self, n: int = 2) -> list[ShortScript]:
        return [_script(short_id=f"sid-{i:03d}") for i in range(n)]

    def _run(self, tmp_path, *, scripts=None, dry_run=False,
             story_override="Top AI news", upload_id="yt-001",
             render_path=None, render_side_effect=None):
        if scripts is None:
            scripts = self._scripts()
        if render_path is None and render_side_effect is None:
            render_path = tmp_path / "short.mp4"
            render_path.touch()

        ms = _mock_settings(output_dir=tmp_path)
        mock_engine = MagicMock()
        mock_engine.generate.return_value = scripts
        mock_engine.get_best_combination.return_value = None
        mock_engine.load_scripts.return_value = []
        mock_exp = MagicMock()
        mock_exp.collect_analytics.return_value = 0
        mock_exp._store.load_results.return_value = []

        render_mock = MagicMock(
            side_effect=render_side_effect,
            return_value=render_path,
        )

        with ExitStack() as stack:
            stack.enter_context(patch("src.shorts_pipeline.settings", ms))
            stack.enter_context(patch("src.shorts_pipeline.ShortsEngineV2",
                                      return_value=mock_engine))
            stack.enter_context(patch("src.shorts_pipeline.ShortsExperimentEngine",
                                      return_value=mock_exp))
            stack.enter_context(patch("src.hook_pattern_engine.HookPatternEngine",
                                      return_value=MagicMock()))
            stack.enter_context(patch("src.shorts_experiment_engine.ShortsExperiment",
                                      return_value=MagicMock()))
            stack.enter_context(patch("src.shorts_pipeline._setup_logging"))
            stack.enter_context(patch("src.shorts_pipeline.notify_success"))
            stack.enter_context(patch("src.shorts_pipeline.notify_failure"))
            stack.enter_context(patch("src.shorts_pipeline.render_short", render_mock))
            stack.enter_context(patch("src.shorts_pipeline._upload",
                                      return_value=upload_id))
            return ShortsPipeline().run(
                dry_run=dry_run,
                story_override=story_override,
            ), mock_exp

    def test_dry_run_stops_after_script_generation(self, tmp_path):
        scripts = self._scripts(3)
        summary, _ = self._run(tmp_path, scripts=scripts, dry_run=True)
        assert summary["status"] == "dry_run"
        assert summary["generated"] == 3
        assert "scripts" in summary
        assert summary["uploaded"] == 0

    def test_no_story_returns_no_story_status(self, tmp_path):
        ms = _mock_settings(output_dir=tmp_path)
        with (
            patch("src.shorts_pipeline.settings", ms),
            patch("src.shorts_pipeline._setup_logging"),
            patch("src.shorts_pipeline.ShortsPipeline._get_story", return_value=None),
        ):
            summary = ShortsPipeline().run(story_override=None)
        assert summary["status"] == "no_story"

    def test_full_run_counts_rendered_and_uploaded(self, tmp_path):
        scripts = self._scripts(3)
        summary, _ = self._run(tmp_path, scripts=scripts)
        assert summary["rendered"] == 3
        assert summary["uploaded"] == 3
        assert summary["status"] == "done"

    def test_render_failure_does_not_abort_remaining(self, tmp_path):
        scripts = self._scripts(3)
        fake = tmp_path / "short.mp4"
        fake.touch()
        # First fails, next two succeed
        summary, _ = self._run(tmp_path, scripts=scripts,
                                render_side_effect=[None, fake, fake])
        assert summary["rendered"] == 2
        assert summary["uploaded"] == 2
        assert len(summary["results"]) == 3

    def test_failed_script_has_error_in_results(self, tmp_path):
        scripts = self._scripts(1)
        summary, _ = self._run(tmp_path, scripts=scripts,
                                render_side_effect=[None])
        assert summary["results"][0]["error"] == "render_failed"

    def test_summary_has_expected_keys(self, tmp_path):
        summary, _ = self._run(tmp_path)
        for key in ("date", "run_dir", "generated", "rendered", "uploaded",
                    "results", "status"):
            assert key in summary

    def test_unexpected_exception_sets_failed_status(self, tmp_path):
        ms = _mock_settings(output_dir=tmp_path)
        with (
            patch("src.shorts_pipeline.settings", ms),
            patch("src.shorts_pipeline._setup_logging"),
            patch("src.shorts_pipeline.notify_failure"),
            patch("src.shorts_pipeline.ShortsPipeline._get_story",
                  side_effect=RuntimeError("boom")),
        ):
            summary = ShortsPipeline().run(story_override=None)
        assert summary["status"] == "failed"
        assert "boom" in summary["error"]

    def test_experiment_registered_after_successful_upload(self, tmp_path):
        scripts = self._scripts(1)
        _, mock_exp = self._run(tmp_path, scripts=scripts)
        mock_exp._store.save_experiment.assert_called_once()
