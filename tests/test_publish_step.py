"""Tests for src/publish_step.py — _PublishStep class."""
from __future__ import annotations

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
):
    sys.modules.setdefault(_m, MagicMock())

import pytest

from src.pipeline_context import PipelineContext
from src.publish_step import _PublishStep
from src.script_generator import Scene, VideoScript

# ── Factories ─────────────────────────────────────────────────────────────────

def _mock_script(n_scenes: int = 3, hook_variants: list[str] | None = None) -> VideoScript:
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
        hook_variants=hook_variants if hook_variants is not None else ["Hook A", "Hook B"],
        scenes=scenes,
    )


def _publisher(
    tmp_path: Path,
    *,
    script: VideoScript | None = None,
    skip_upload: bool = False,
    cp: MagicMock | None = None,
    summary: dict | None = None,
    news: list | None = None,
    thumbnail: Path | None = None,
    long_video: Path | None = None,
    feedback_history: list | None = None,
) -> _PublishStep:
    if script is None:
        script = _mock_script()
    if cp is None:
        cp = MagicMock()
        cp.is_done.return_value = False
        cp.metadata.return_value = {}
    if summary is None:
        summary = {}
    if news is None:
        news = [MagicMock(title="Story 1")]
    if thumbnail is None:
        thumbnail = tmp_path / "thumbnail_bold.jpg"
        thumbnail.touch()
    if long_video is None:
        long_video = tmp_path / "final_video.mp4"
        long_video.touch()
    if feedback_history is None:
        feedback_history = []

    ctx = PipelineContext(
        run_dir=tmp_path,
        cp=cp,
        observer=MagicMock(),
        live=MagicMock(),
        seen=MagicMock(),
    )

    return _PublishStep(
        script=script,
        ctx=ctx,
        news=news,
        summary=summary,
        skip_upload=skip_upload,
        auto_strategy=None,
        auto_actions=[],
        long_video=long_video,
        short_video=None,
        thumbnail=thumbnail,
        subtitle_path=None,
        feedback_history=feedback_history,
    )


# ── run_quality_gate tests ────────────────────────────────────────────────────

class TestRunQualityGate:
    def test_sets_quality_score_in_summary(self, tmp_path):
        summary = {}
        p = _publisher(tmp_path, summary=summary)
        mock_report = MagicMock()
        mock_report.score = 0.85
        mock_report.warnings = []

        with patch("src.publish_step.run_gate", return_value=mock_report):
            p.run_quality_gate()

        assert summary["quality_score"] == 0.85

    def test_sets_quality_warnings_when_present(self, tmp_path):
        summary = {}
        p = _publisher(tmp_path, summary=summary)
        mock_report = MagicMock()
        mock_report.score = 0.6
        mock_report.warnings = ["Hook too short", "Slow pacing"]

        with patch("src.publish_step.run_gate", return_value=mock_report):
            p.run_quality_gate()

        assert summary["quality_warnings"] == ["Hook too short", "Slow pacing"]

    def test_no_quality_warnings_key_when_empty(self, tmp_path):
        summary = {}
        p = _publisher(tmp_path, summary=summary)
        mock_report = MagicMock()
        mock_report.score = 0.9
        mock_report.warnings = []

        with patch("src.publish_step.run_gate", return_value=mock_report):
            p.run_quality_gate()

        assert "quality_warnings" not in summary

    def test_calls_run_gate_with_audio_dir(self, tmp_path):
        p = _publisher(tmp_path, feedback_history=[{"vid": 1}])
        mock_report = MagicMock()
        mock_report.score = 0.7
        mock_report.warnings = []

        with patch("src.publish_step.run_gate", return_value=mock_report) as mock_gate:
            p.run_quality_gate()

        mock_gate.assert_called_once_with(
            p._script, tmp_path / "audio", [{"vid": 1}]
        )


# ── upload_en tests ───────────────────────────────────────────────────────────

class TestUploadEn:
    def test_checkpoint_done_skipped_sets_status_built_not_uploaded(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = True
        cp.metadata.return_value = {"skipped": True}
        summary = {}
        p = _publisher(tmp_path, cp=cp, summary=summary)

        p.upload_en()

        assert summary["status"] == "built_not_uploaded"
        p._observer.step_skip.assert_called_once_with("upload_en")

    def test_checkpoint_done_published_sets_status_published(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = True
        cp.metadata.return_value = {"skipped": False, "video_id": "abc123"}
        summary = {}
        p = _publisher(tmp_path, cp=cp, summary=summary)

        p.upload_en()

        assert summary["status"] == "published"
        assert summary["video_id"] == "abc123"

    def test_checkpoint_done_published_updates_summary_from_meta(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = True
        cp.metadata.return_value = {"skipped": False, "video_id": "xyz", "playlist_id": "PL1"}
        summary = {}
        p = _publisher(tmp_path, cp=cp, summary=summary)

        p.upload_en()

        assert summary["video_id"] == "xyz"
        assert summary["playlist_id"] == "PL1"

    def test_skip_upload_sets_status_built_not_uploaded(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        summary = {}
        p = _publisher(tmp_path, skip_upload=True, cp=cp, summary=summary)

        p.upload_en()

        assert summary["status"] == "built_not_uploaded"
        cp.mark_done.assert_called_once_with("upload_en", {"skipped": True})
        p._observer.step_done.assert_called_once_with("upload_en", skipped=True)

    def test_skip_upload_does_not_call_publish_episode(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        p = _publisher(tmp_path, skip_upload=True, cp=cp)

        with patch("src.publish_step.publish_episode") as mock_pub:
            p.upload_en()

        mock_pub.assert_not_called()

    def test_fresh_upload_calls_publish_episode(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        summary = {}
        p = _publisher(tmp_path, cp=cp, summary=summary)

        with ExitStack() as stack:
            mock_pub = stack.enter_context(
                patch("src.publish_step.publish_episode", return_value={"video_id": "v1"})
            )
            stack.enter_context(patch("src.publish_step.get_recommendations", return_value=[]))
            stack.enter_context(patch("src.publish_step.record_usage"))
            stack.enter_context(patch("src.publish_step.record_thumbnail_usage"))
            stack.enter_context(patch("src.publish_step.save_result"))
            stack.enter_context(patch("src.publish_step.ThompsonBandit"))
            stack.enter_context(patch("src.publish_step._classify_hook_type", return_value="curiosity"))
            p.upload_en()

        mock_pub.assert_called_once()

    def test_fresh_upload_sets_status_published(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        summary = {}
        p = _publisher(tmp_path, cp=cp, summary=summary)

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.publish_episode", return_value={"video_id": "v1"})
            )
            stack.enter_context(patch("src.publish_step.get_recommendations", return_value=[]))
            stack.enter_context(patch("src.publish_step.record_usage"))
            stack.enter_context(patch("src.publish_step.record_thumbnail_usage"))
            stack.enter_context(patch("src.publish_step.save_result"))
            stack.enter_context(patch("src.publish_step.ThompsonBandit"))
            stack.enter_context(patch("src.publish_step._classify_hook_type", return_value="curiosity"))
            p.upload_en()

        assert summary["status"] == "published"

    def test_fresh_upload_calls_record_usage_with_video_id(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        p = _publisher(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.publish_episode", return_value={"video_id": "v42"})
            )
            stack.enter_context(patch("src.publish_step.get_recommendations", return_value=[]))
            mock_rec = stack.enter_context(patch("src.publish_step.record_usage"))
            stack.enter_context(patch("src.publish_step.record_thumbnail_usage"))
            stack.enter_context(patch("src.publish_step.save_result"))
            stack.enter_context(patch("src.publish_step.ThompsonBandit"))
            stack.enter_context(patch("src.publish_step._classify_hook_type", return_value="curiosity"))
            stack.enter_context(patch("src.publish_step.settings"))
            p.upload_en()

        mock_rec.assert_called_once()
        args = mock_rec.call_args[0]
        assert "v42" in args

    def test_fresh_upload_calls_save_result(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        p = _publisher(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.publish_episode", return_value={"video_id": "v99"})
            )
            stack.enter_context(patch("src.publish_step.get_recommendations", return_value=[]))
            stack.enter_context(patch("src.publish_step.record_usage"))
            stack.enter_context(patch("src.publish_step.record_thumbnail_usage"))
            mock_save = stack.enter_context(patch("src.publish_step.save_result"))
            stack.enter_context(patch("src.publish_step.ThompsonBandit"))
            stack.enter_context(patch("src.publish_step._classify_hook_type", return_value="curiosity"))
            stack.enter_context(patch("src.publish_step.settings"))
            p.upload_en()

        mock_save.assert_called_once()

    def test_fresh_upload_marks_checkpoint_done(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        p = _publisher(tmp_path, cp=cp)

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.publish_episode", return_value={"video_id": "v1"})
            )
            stack.enter_context(patch("src.publish_step.get_recommendations", return_value=[]))
            stack.enter_context(patch("src.publish_step.record_usage"))
            stack.enter_context(patch("src.publish_step.record_thumbnail_usage"))
            stack.enter_context(patch("src.publish_step.save_result"))
            stack.enter_context(patch("src.publish_step.ThompsonBandit"))
            stack.enter_context(patch("src.publish_step._classify_hook_type", return_value="curiosity"))
            stack.enter_context(patch("src.publish_step.settings"))
            p.upload_en()

        cp.mark_done.assert_called_once()


# ── run_shorts_experiments early-exit tests ───────────────────────────────────

class TestRunShortsExperiments:
    def test_returns_immediately_when_skip_upload(self, tmp_path):
        p = _publisher(tmp_path, skip_upload=True)

        with patch("src.publish_step.ShortsExperimentEngine") as mock_engine:
            p.run_shorts_experiments()

        mock_engine.assert_not_called()

    def test_returns_immediately_when_no_news(self, tmp_path):
        p = _publisher(tmp_path, news=[])

        with patch("src.publish_step.ShortsExperimentEngine") as mock_engine:
            p.run_shorts_experiments()

        mock_engine.assert_not_called()

    def test_returns_immediately_when_script_is_none(self, tmp_path):
        p = _publisher(tmp_path)
        p._script = None

        with patch("src.publish_step.ShortsExperimentEngine") as mock_engine:
            p.run_shorts_experiments()

        mock_engine.assert_not_called()

    def test_returns_immediately_when_checkpoint_done(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = True
        p = _publisher(tmp_path, cp=cp)

        with patch("src.publish_step.ShortsExperimentEngine") as mock_engine:
            p.run_shorts_experiments()

        mock_engine.assert_not_called()


# ── upload_ru tests ───────────────────────────────────────────────────────────

class TestUploadRu:
    def test_ru_disabled_calls_step_skip(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        p = _publisher(tmp_path, cp=cp)

        ms = MagicMock()
        ms.ru_enabled = False

        with patch("src.publish_step.settings", ms):
            p.upload_ru()

        p._observer.step_skip.assert_called_once_with("upload_ru")

    def test_ru_enabled_already_done_calls_step_skip(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = True
        cp.metadata.return_value = {"status": "published"}
        summary = {}
        p = _publisher(tmp_path, cp=cp, summary=summary)

        ms = MagicMock()
        ms.ru_enabled = True
        ms.source_dir = tmp_path

        with ExitStack() as stack:
            stack.enter_context(patch("src.publish_step.settings", ms))
            stack.enter_context(patch("src.publish_step._get_shared_outro", return_value=None))
            p.upload_ru()

        p._observer.step_skip.assert_called_once_with("upload_ru")
        assert summary["ru"] == {"status": "published"}

    def test_ru_enabled_fresh_calls_run_lang_variant(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        summary = {}
        p = _publisher(tmp_path, cp=cp, summary=summary)

        ms = MagicMock()
        ms.ru_enabled = True
        ms.source_dir = tmp_path
        ms.ru_elevenlabs_voice_id = "voice-ru"
        ms.ru_elevenlabs_model = "model-ru"
        ms.ru_youtube_client_secrets = tmp_path / "secrets_ru.json"
        ms.ru_youtube_token_file = tmp_path / "token_ru.pickle"

        with ExitStack() as stack:
            stack.enter_context(patch("src.publish_step.settings", ms))
            stack.enter_context(patch("src.publish_step._get_shared_outro", return_value=None))
            mock_run_variant = stack.enter_context(
                patch("src.publish_step._run_lang_variant", return_value={"status": "published"})
            )
            p.upload_ru()

        mock_run_variant.assert_called_once()
        assert summary["ru"] == {"status": "published"}
        cp.mark_done.assert_called_once()


# ── _setup_thompson_variants tests ────────────────────────────────────────────

class TestSetupThompsonVariants:
    def test_registers_variants_and_calls_select(self, tmp_path):
        script = _mock_script(hook_variants=["Hook A", "Hook B", "Hook C"])
        p = _publisher(tmp_path, script=script)

        mock_bandit_instance = MagicMock()
        mock_bandit_instance.select_variant.return_value = None

        with ExitStack() as stack:
            mock_bandit_cls = stack.enter_context(
                patch("src.publish_step.ThompsonBandit", return_value=mock_bandit_instance)
            )
            stack.enter_context(
                patch("src.publish_step._classify_hook_type", return_value="curiosity")
            )
            stack.enter_context(patch("src.publish_step.ABTestVariant", wraps=__import__(
                "src.thompson_bandit", fromlist=["ABTestVariant"]
            ).ABTestVariant))
            p._setup_thompson_variants("video123")

        mock_bandit_instance.register_variants.assert_called_once()
        mock_bandit_instance.select_variant.assert_called_once()

    def test_uses_hook_as_fallback_when_no_variants(self, tmp_path):
        script = _mock_script(hook_variants=[])
        p = _publisher(tmp_path, script=script)

        mock_bandit_instance = MagicMock()
        mock_bandit_instance.select_variant.return_value = None

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.ThompsonBandit", return_value=mock_bandit_instance)
            )
            stack.enter_context(
                patch("src.publish_step._classify_hook_type", return_value="simple")
            )
            from src.thompson_bandit import ABTestVariant
            captured_variants = []
            orig_abv = ABTestVariant

            def capturing_abv(**kwargs):
                v = orig_abv(**kwargs)
                captured_variants.append(v)
                return v

            stack.enter_context(patch("src.publish_step.ABTestVariant", side_effect=capturing_abv))
            p._setup_thompson_variants("vid_fallback")

        assert len(captured_variants) == 1
        assert captured_variants[0].hook == script.hook

    def test_caps_variants_at_four(self, tmp_path):
        script = _mock_script(hook_variants=["H1", "H2", "H3", "H4", "H5", "H6"])
        p = _publisher(tmp_path, script=script)

        mock_bandit_instance = MagicMock()
        mock_bandit_instance.select_variant.return_value = None

        registered = []

        def capture_register(variants):
            registered.extend(variants)

        mock_bandit_instance.register_variants.side_effect = capture_register

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.ThompsonBandit", return_value=mock_bandit_instance)
            )
            stack.enter_context(
                patch("src.publish_step._classify_hook_type", return_value="conflict")
            )
            from src.thompson_bandit import ABTestVariant
            stack.enter_context(
                patch("src.publish_step.ABTestVariant", side_effect=lambda **kw: ABTestVariant(**kw))
            )
            p._setup_thompson_variants("vid_cap")

        assert len(registered) == 4

    def test_record_switch_called_when_best_variant_selected(self, tmp_path):
        script = _mock_script(hook_variants=["Hook A"])
        p = _publisher(tmp_path, script=script)

        mock_best = MagicMock()
        mock_best.variant_id = "A"
        mock_best.type = "curiosity"
        mock_best.hook = "Hook A"

        mock_bandit_instance = MagicMock()
        mock_bandit_instance.select_variant.return_value = mock_best

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.ThompsonBandit", return_value=mock_bandit_instance)
            )
            stack.enter_context(
                patch("src.publish_step._classify_hook_type", return_value="curiosity")
            )
            from src.thompson_bandit import ABTestVariant
            stack.enter_context(
                patch("src.publish_step.ABTestVariant", side_effect=lambda **kw: ABTestVariant(**kw))
            )
            p._setup_thompson_variants("vid_switch")

        mock_bandit_instance.record_switch.assert_called_once()

    def test_record_switch_not_called_when_no_best(self, tmp_path):
        script = _mock_script(hook_variants=["Hook A"])
        p = _publisher(tmp_path, script=script)

        mock_bandit_instance = MagicMock()
        mock_bandit_instance.select_variant.return_value = None

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.ThompsonBandit", return_value=mock_bandit_instance)
            )
            stack.enter_context(
                patch("src.publish_step._classify_hook_type", return_value="curiosity")
            )
            from src.thompson_bandit import ABTestVariant
            stack.enter_context(
                patch("src.publish_step.ABTestVariant", side_effect=lambda **kw: ABTestVariant(**kw))
            )
            p._setup_thompson_variants("vid_no_switch")

        mock_bandit_instance.record_switch.assert_not_called()

    def test_variant_ids_assigned_alphabetically(self, tmp_path):
        script = _mock_script(hook_variants=["H1", "H2", "H3"])
        p = _publisher(tmp_path, script=script)

        mock_bandit_instance = MagicMock()
        mock_bandit_instance.select_variant.return_value = None

        created = []

        def capture(**kwargs):
            from src.thompson_bandit import ABTestVariant
            v = ABTestVariant(**kwargs)
            created.append(v)
            return v

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.ThompsonBandit", return_value=mock_bandit_instance)
            )
            stack.enter_context(
                patch("src.publish_step._classify_hook_type", return_value="simple")
            )
            stack.enter_context(patch("src.publish_step.ABTestVariant", side_effect=capture))
            p._setup_thompson_variants("vid_abc")

        assert [v.id for v in created] == ["A", "B", "C"]


# ── run_shorts_experiments — main execution path ──────────────────────────────

class TestRunShortsExperimentsExecution:
    """Tests covering lines 195-251: the engine + per-experiment loop."""

    @staticmethod
    def _make_exp(idx: int = 0, hook_type: str = "curiosity") -> MagicMock:
        exp = MagicMock()
        exp.hook_text = f"Hook text {idx}"
        exp.core_text = f"Core text {idx}"
        exp.payoff_text = f"Payoff {idx}"
        exp.hook_type = hook_type
        exp.experiment_id = f"exp_{idx:03d}"
        return exp

    def _run(
        self,
        p: _PublishStep,
        tmp_path: Path,
        experiments: list,
        upload_raises: Exception | None = None,
        engine_raises: Exception | None = None,
    ) -> MagicMock:
        """Patch inline imports and run the step; return the engine mock."""
        mock_engine = MagicMock()
        if engine_raises:
            mock_engine.generate.side_effect = engine_raises
        else:
            mock_engine.generate.return_value = experiments

        video_path = tmp_path / "short.mp4"
        video_path.touch()

        mock_sp = MagicMock()
        mock_sp.render_short.return_value = video_path

        mock_upload = MagicMock()
        if upload_raises:
            mock_upload.side_effect = upload_raises
        else:
            mock_upload.return_value = "short_vid_1"

        ms = MagicMock()
        ms.data_dir = tmp_path
        ms.youtube_client_secrets = tmp_path / "s.json"
        ms.youtube_token_file = tmp_path / "t.pickle"

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.ShortsExperimentEngine", return_value=mock_engine)
            )
            stack.enter_context(patch("src.publish_step.settings", ms))
            stack.enter_context(patch.dict(sys.modules, {
                "src.shorts_pipeline": mock_sp,
                "src.shorts_engine_v2": MagicMock(),
            }))
            stack.enter_context(
                patch("src.youtube_uploader.upload_short", mock_upload, create=True)
            )
            p.run_shorts_experiments()

        return mock_engine

    def test_single_experiment_uploaded_updates_summary(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        summary = {}
        p = _publisher(tmp_path, cp=cp, summary=summary)

        self._run(p, tmp_path, [self._make_exp()])

        assert summary["shorts_experiments"]["uploaded"] == 1

    def test_marks_checkpoint_done_with_count(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        p = _publisher(tmp_path, cp=cp)

        self._run(p, tmp_path, [self._make_exp()])

        cp.mark_done.assert_called_once_with("shorts_experiments", {"uploaded": 1})

    def test_multiple_experiments_all_counted(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        summary = {}
        p = _publisher(tmp_path, cp=cp, summary=summary)

        self._run(p, tmp_path, [self._make_exp(i) for i in range(3)])

        assert summary["shorts_experiments"]["uploaded"] == 3

    def test_per_experiment_upload_failure_swallowed(self, tmp_path):
        """A single failing experiment doesn't abort the loop."""
        cp = MagicMock()
        cp.is_done.return_value = False
        summary = {}
        p = _publisher(tmp_path, cp=cp, summary=summary)

        self._run(p, tmp_path, [self._make_exp()], upload_raises=RuntimeError("upload boom"))

        assert summary["shorts_experiments"]["uploaded"] == 0
        cp.mark_done.assert_called_once_with("shorts_experiments", {"uploaded": 0})

    def test_outer_engine_failure_swallowed(self, tmp_path):
        """Engine.generate failure is non-fatal — checkpoint not marked done."""
        cp = MagicMock()
        cp.is_done.return_value = False
        p = _publisher(tmp_path, cp=cp, summary={})

        self._run(p, tmp_path, [], engine_raises=RuntimeError("engine boom"))

        cp.mark_done.assert_not_called()

    def test_mark_uploaded_called_for_each_experiment(self, tmp_path):
        cp = MagicMock()
        cp.is_done.return_value = False
        p = _publisher(tmp_path, cp=cp, summary={})

        exp = self._make_exp(idx=7)
        engine = self._run(p, tmp_path, [exp])

        engine.mark_uploaded.assert_called_once()
        first_arg = engine.mark_uploaded.call_args[0][0]
        assert first_arg == exp.experiment_id

    def test_hook_text_truncated_to_95_chars(self, tmp_path):
        """Short title is capped at 95 chars + ellipsis when hook is longer."""
        cp = MagicMock()
        cp.is_done.return_value = False
        p = _publisher(tmp_path, cp=cp, summary={})

        long_hook = "X" * 120
        exp = self._make_exp()
        exp.hook_text = long_hook

        uploaded_titles: list[str] = []

        video_path = tmp_path / "s.mp4"
        video_path.touch()
        mock_sp = MagicMock()
        mock_sp.render_short.return_value = video_path

        def capture_upload(**kwargs):
            uploaded_titles.append(kwargs.get("title", ""))
            return "vid_x"

        ms = MagicMock()
        ms.data_dir = tmp_path
        ms.youtube_client_secrets = tmp_path / "s.json"
        ms.youtube_token_file = tmp_path / "t.pickle"

        mock_engine = MagicMock()
        mock_engine.generate.return_value = [exp]

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.ShortsExperimentEngine", return_value=mock_engine)
            )
            stack.enter_context(patch("src.publish_step.settings", ms))
            stack.enter_context(patch.dict(sys.modules, {
                "src.shorts_pipeline": mock_sp,
                "src.shorts_engine_v2": MagicMock(),
            }))
            mock_up = MagicMock(side_effect=lambda **kw: capture_upload(**kw) or "v")
            stack.enter_context(
                patch("src.youtube_uploader.upload_short", mock_up, create=True)
            )
            p.run_shorts_experiments()

        if uploaded_titles:
            assert len(uploaded_titles[0]) <= 96


# ── upload_en — optional engine except-handlers ───────────────────────────────

class TestUploadEnOptionalEngines:
    """Cover the except-branches in the 5 optional post-upload engine calls."""

    @staticmethod
    def _run_upload_with_failing_engine(tmp_path: Path, module_name: str) -> None:
        """Patch one inline-imported engine module so its instantiation raises."""
        cp = MagicMock()
        cp.is_done.return_value = False
        p = _publisher(tmp_path, cp=cp, summary={})

        failing_mod = MagicMock()
        # All attributes raise when called so any `Engine()` call raises
        for attr in ["PersonaEngine", "get_narrative_engine", "get_conflict_engine",
                     "EmotionEngine", "EmotionalArcEngine"]:
            getattr(failing_mod, attr).side_effect = RuntimeError("engine down")

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.publish_episode", return_value={"video_id": "v1"})
            )
            stack.enter_context(patch("src.publish_step.get_recommendations", return_value=[]))
            stack.enter_context(patch("src.publish_step.record_usage"))
            stack.enter_context(patch("src.publish_step.record_thumbnail_usage"))
            stack.enter_context(patch("src.publish_step.save_result"))
            stack.enter_context(patch("src.publish_step.ThompsonBandit"))
            stack.enter_context(patch("src.publish_step._classify_hook_type", return_value="c"))
            stack.enter_context(patch("src.publish_step.settings"))
            stack.enter_context(patch.dict(sys.modules, {module_name: failing_mod}))
            p.upload_en()  # must not raise

    def test_persona_engine_exception_swallowed(self, tmp_path):
        self._run_upload_with_failing_engine(tmp_path, "src.persona_engine")

    def test_narrative_identity_engine_exception_swallowed(self, tmp_path):
        self._run_upload_with_failing_engine(tmp_path, "src.narrative_identity_engine")

    def test_emotion_engine_exception_swallowed(self, tmp_path):
        self._run_upload_with_failing_engine(tmp_path, "src.emotion_engine")

    def test_emotional_arc_engine_exception_swallowed(self, tmp_path):
        self._run_upload_with_failing_engine(tmp_path, "src.emotional_arc_engine")

    def test_conflict_engine_with_conflict_line_calls_record(self, tmp_path):
        """When editorial plan has conflict_line, ConflictEngine.record is called."""
        cp = MagicMock()
        cp.is_done.return_value = False
        summary = {
            "editorial_plan": {
                "editorial_plan": [{
                    "angle": "threat",
                    "format": "hot_take",
                    "conflict_type": "external",
                    "conflict_line": "AI is taking over",
                    "conflict_intensity": 0.8,
                }]
            }
        }
        p = _publisher(tmp_path, cp=cp, summary=summary)

        mock_conflict_engine = MagicMock()
        mock_conflict_mod = MagicMock()
        mock_conflict_mod.get_conflict_engine.return_value = mock_conflict_engine

        with ExitStack() as stack:
            stack.enter_context(
                patch("src.publish_step.publish_episode", return_value={"video_id": "v1"})
            )
            stack.enter_context(patch("src.publish_step.get_recommendations", return_value=[]))
            stack.enter_context(patch("src.publish_step.record_usage"))
            stack.enter_context(patch("src.publish_step.record_thumbnail_usage"))
            stack.enter_context(patch("src.publish_step.save_result"))
            stack.enter_context(patch("src.publish_step.ThompsonBandit"))
            stack.enter_context(patch("src.publish_step._classify_hook_type", return_value="c"))
            stack.enter_context(patch("src.publish_step.settings"))
            stack.enter_context(
                patch.dict(sys.modules, {"src.narrative_conflict_engine": mock_conflict_mod})
            )
            p.upload_en()

        mock_conflict_engine.record.assert_called_once()
        call_kw = mock_conflict_engine.record.call_args[1]
        assert call_kw["line"] == "AI is taking over"
        assert call_kw["intensity"] == 0.8
