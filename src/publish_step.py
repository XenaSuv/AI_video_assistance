"""Upload and publish steps."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.analytics import get_recommendations
from src.checkpoint import PipelineCheckpoint
from src.deduplicator import SeenStories
from src.hook_selector import record_usage
from src.language_variant import _run_language_variant as _run_lang_variant
from src.live_state import LiveState
from src.performance_tracker import save_result
from src.pipeline_helpers import _classify_hook_type, _get_shared_outro
from src.pipeline_observer import PipelineObserver
from src.quality_gate import QualityGateError, run_gate
from src.script_generator import VideoScript
from src.shorts_experiment_engine import ShortsExperimentEngine
from src.thompson_bandit import ABTestVariant, ThompsonBandit
from src.thumbnail_ab import record_thumbnail_usage
from src.youtube_uploader import publish_episode


class _PublishStep:
    def __init__(
        self,
        script: VideoScript | None,
        run_dir: Path,
        cp: PipelineCheckpoint,
        observer: PipelineObserver,
        live: LiveState,
        seen: SeenStories,
        news: list[Any],
        summary: dict[str, Any],
        skip_upload: bool,
        auto_strategy: dict[str, Any] | None,
        auto_actions: list[Any],
        long_video: Path | None,
        short_video: Path | None,
        thumbnail: Path | None,
        subtitle_path: Path | None,
        feedback_history: list[Any],
    ) -> None:
        self._script = script
        self._run_dir = run_dir
        self._cp = cp
        self._observer = observer
        self._live = live
        self._seen = seen
        self._news = news
        self._summary = summary
        self._skip_upload = skip_upload
        self._auto_strategy = auto_strategy
        self._auto_actions = auto_actions
        self._long_video = long_video
        self._short_video = short_video
        self._thumbnail = thumbnail
        self._subtitle_path = subtitle_path
        self._feedback_history = feedback_history

    def run_quality_gate(self) -> None:
        assert self._script is not None
        quality_report = run_gate(
            self._script, self._run_dir / "audio", self._feedback_history
        )
        self._summary["quality_score"] = quality_report.score
        if quality_report.warnings:
            self._summary["quality_warnings"] = quality_report.warnings

    def upload_en(self) -> None:
        assert self._script is not None
        assert self._long_video is not None
        assert self._thumbnail is not None
        self._observer.step_start("upload_en")
        if not self._cp.is_done("upload_en"):
            if self._skip_upload:
                logger.info("--skip-upload specified; leaving files on disk")
                self._summary["status"] = "built_not_uploaded"
                self._cp.mark_done("upload_en", {"skipped": True})
                self._observer.step_done("upload_en", skipped=True)
            else:
                ids = publish_episode(
                    self._script, self._long_video, self._short_video,
                    thumbnail=self._thumbnail,
                    subtitle_path=self._subtitle_path,
                )
                self._summary.update(ids)
                self._summary["status"] = "published"
                self._summary["analytics"] = get_recommendations()
                if video_id := ids.get("video_id"):
                    record_usage(self._script.hook, video_id, settings.data_dir, "daily")
                    record_thumbnail_usage(self._thumbnail, video_id, settings.data_dir, "daily")
                    _ep = self._summary.get("editorial_plan", {}).get("editorial_plan") or [{}]
                    save_result(
                        video_id, self._script.hook, self._script.title, self._script.description,
                        content_type="daily",
                        strategy=self._auto_strategy,
                        auto_actions=self._auto_actions,
                        angle=_ep[0].get("angle", ""),
                        format=_ep[0].get("format", ""),
                        platform="youtube",
                    )
                    self._setup_thompson_variants(video_id)
                    # Record persona memory so the character can reference this
                    # opinion in future videos ("I said this before...")
                    try:
                        from src.persona_engine import PersonaEngine
                        _fmt = _ep[0].get("format", "opinion")
                        PersonaEngine().record(
                            topic      = self._script.title,
                            opinion    = self._script.hook,
                            entry_type = _fmt if _fmt in ("hot_take", "prediction", "reaction", "breakdown") else "opinion",
                            video_id   = video_id,
                        )
                    except Exception as _mem_exc:
                        logger.debug(f"PersonaMemory record skipped: {_mem_exc}")
                    try:
                        from src.narrative_identity_engine import get_narrative_engine
                        _ep_plan = _ep[0] if _ep else {}
                        get_narrative_engine().record(
                            topic      = self._script.title,
                            content    = self._script.hook,
                            entry_type = _ep_plan.get("format", "opinion"),
                            theme      = _ep_plan.get("theme", ""),
                            arc_key    = _ep_plan.get("arc_key", ""),
                            video_id   = video_id,
                        )
                    except Exception as _narr_exc:
                        logger.debug(f"NarrativeMemory record skipped: {_narr_exc}")
                    try:
                        from src.narrative_conflict_engine import get_conflict_engine
                        _ep_plan = _ep[0] if _ep else {}
                        _ctype = _ep_plan.get("conflict_type", "external")
                        _cline = _ep_plan.get("conflict_line", "")
                        _cint  = _ep_plan.get("conflict_intensity", 0.5)
                        if _cline:
                            get_conflict_engine().record(
                                topic         = self._script.title,
                                conflict_type = _ctype,
                                intensity     = _cint,
                                line          = _cline,
                                video_id      = video_id,
                            )
                    except Exception as _conf_exc:
                        logger.debug(f"ConflictMemory record skipped: {_conf_exc}")
                    try:
                        from src.emotion_engine import EmotionEngine
                        EmotionEngine().log_stats(self._script, video_id)
                    except Exception as _emo_exc:
                        logger.debug(f"EmotionEngine log_stats skipped: {_emo_exc}")
                    try:
                        from src.emotional_arc_engine import EmotionalArcEngine
                        _arc_eng  = EmotionalArcEngine()
                        _arc_type = _arc_eng.select_arc_from_editorial(
                            self._editorial if hasattr(self, "_editorial") else {}
                        )
                        _arc_eng.log_stats(self._script, _arc_type, video_id)
                    except Exception as _arc_exc:
                        logger.debug(f"EmotionalArcEngine log_stats skipped: {_arc_exc}")
                    self._live.log_event(f"Uploaded to YouTube EN: {video_id}")
                    self._live.set_metrics({"video_id": video_id, "views": 0, "ctr": 0.0})
                self._cp.mark_done("upload_en", {k: v for k, v in ids.items()})
                self._observer.step_done(
                    "upload_en",
                    **{k: v for k, v in ids.items() if isinstance(v, (str, int, float, bool))},
                )
        else:
            logger.info("Step 7 (upload_en) already done")
            meta = self._cp.metadata("upload_en")
            self._summary["status"] = (
                "built_not_uploaded" if meta.get("skipped") else "published"
            )
            if not meta.get("skipped"):
                self._summary.update(meta)
            self._observer.step_skip("upload_en")

    def run_shorts_experiments(self) -> None:
        """Generate and upload hook-variant Shorts experiments for the top story.

        Each experiment is an independent Short rendered via the modern pipeline:
        ElevenLabs TTS audio + Pexels/Pixabay B-roll background + animated subtitles.
        Results are collected in the next run's deferred feedback loop via
        _collect_thompson_strategy() which calls ShortsExperimentEngine.collect_analytics().

        Non-fatal: failures are logged and the pipeline continues normally.
        """
        if self._skip_upload or not self._news or not self._script:
            return
        if self._cp.is_done("shorts_experiments"):
            logger.info("Shorts experiments already done for this run")
            return

        try:
            from src.shorts_engine_v2 import ShortScript
            from src.shorts_pipeline import render_short
            from src.youtube_uploader import upload_short

            shorts_engine = ShortsExperimentEngine(data_dir=settings.data_dir)
            top_story = {
                "title":  self._news[0].title,
                "source": getattr(self._news[0], "source", ""),
            }
            experiments = shorts_engine.generate(top_story)
            uploaded = 0

            for exp in experiments:
                try:
                    exp_dir = self._run_dir / "shorts_experiments" / exp.experiment_id
                    exp_dir.mkdir(parents=True, exist_ok=True)

                    script = ShortScript(
                        short_id=exp.experiment_id,
                        topic=exp.story_title,
                        hook_type=exp.hook_type,
                        hook=exp.hook_text,
                        core=exp.core_text,
                        twist=exp.payoff_text,
                        ending=exp.payoff_text,
                        style="fast_cut",
                        persona="direct",
                    )
                    video_path = render_short(script, exp_dir)
                    if video_path is None:
                        raise RuntimeError("render_short returned None")

                    title = script.hook[:95] + ("…" if len(script.hook) > 95 else "")
                    video_id = upload_short(
                        video_path=video_path,
                        title=title,
                        description=(
                            f"{exp.hook_text}\n\n"
                            f"{exp.core_text}\n\n"
                            f"#AINews #Shorts #AI"
                        ),
                        tags=["AI", "AINews", "Shorts", exp.hook_type],
                        client_secrets=settings.youtube_client_secrets,
                        token_file=settings.youtube_token_file,
                    )
                    shorts_engine.mark_uploaded(
                        exp.experiment_id,
                        video_id=video_id,
                        video_path=str(video_path),
                    )
                    uploaded += 1
                    logger.info(
                        f"ShortsExperiment uploaded: [{exp.hook_type}] {video_id}"
                    )
                except Exception as exc:
                    logger.warning(
                        f"ShortsExperiment: failed to generate/upload "
                        f"[{exp.hook_type}]: {exc}"
                    )

            self._cp.mark_done("shorts_experiments", {"uploaded": uploaded})
            self._summary["shorts_experiments"] = {"uploaded": uploaded}
            logger.info(
                f"ShortsExperimentEngine: {uploaded}/{len(experiments)} experiments uploaded"
            )
        except Exception as exc:
            logger.warning(f"ShortsExperimentEngine: step failed (non-fatal): {exc}")

    def upload_ru(self) -> None:
        assert self._script is not None
        self._observer.step_start("upload_ru")
        if settings.ru_enabled and not self._cp.is_done("upload_ru"):
            ru_intro = settings.source_dir / "ai-novosti-intro.mp4"
            ru = self._run_language_variant(
                english_script=self._script,
                lang_code="ru",
                lang_name="Russian",
                voice_id=settings.ru_elevenlabs_voice_id,
                voice_model=settings.ru_elevenlabs_model,
                client_secrets=settings.ru_youtube_client_secrets,
                token_file=settings.ru_youtube_token_file,
                intro_path=ru_intro if ru_intro.exists() else None,
                outro_path=_get_shared_outro(),
                include_short=False,
            )
            self._summary["ru"] = ru
            self._cp.mark_done("upload_ru", ru)
            self._observer.step_done("upload_ru", status=ru.get("status"))
        elif settings.ru_enabled:
            logger.info("Step 8 (upload_ru) already done")
            self._summary["ru"] = self._cp.metadata("upload_ru")
            self._observer.step_skip("upload_ru")
        else:
            self._observer.step_skip("upload_ru")

    def _setup_thompson_variants(self, video_id: str) -> None:
        """Register hook variants as Thompson arms for this video after upload.

        Creates one arm per hook_variant (max 4), classifies each by type, then
        calls select_variant() to log which arm the bandit currently favours.
        State is persisted to output/<date>/thompson_state.json so the next
        pipeline run can update it with real YouTube metrics.
        """
        hook_texts = self._script.hook_variants or [self._script.hook]  # type: ignore[union-attr]
        variants = [
            ABTestVariant(
                id=chr(ord("A") + i),
                type=_classify_hook_type(h),
                title=self._script.title,  # type: ignore[union-attr]
                thumbnail={},
                hook=h,
            )
            for i, h in enumerate(hook_texts[:4])
        ]

        bandit = ThompsonBandit(data_dir=self._run_dir)
        bandit.register_variants(variants)

        best = bandit.select_variant()
        if best:
            bandit.record_switch()
            logger.info(
                f"ThompsonBandit: {len(variants)} arm(s) registered for {video_id}; "
                f"initial selection → {best.variant_id!r} ({best.type}) "
                f"— {best.hook[:60]!r}"
            )

    def _run_language_variant(
        self,
        english_script: VideoScript,
        lang_code: str,
        lang_name: str,
        voice_id: str,
        voice_model: str,
        client_secrets: Path,
        token_file: Path,
        intro_path: Path | None = None,
        outro_path: Path | None = None,
        include_short: bool = True,
    ) -> dict[str, Any]:
        return _run_lang_variant(
            english_script=english_script,
            run_dir=self._run_dir,
            lang_code=lang_code,
            lang_name=lang_name,
            voice_id=voice_id,
            voice_model=voice_model,
            client_secrets=client_secrets,
            token_file=token_file,
            skip_upload=self._skip_upload,
            intro_path=intro_path,
            outro_path=outro_path,
            include_short=include_short,
        )
