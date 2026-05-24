"""Media artifact construction — voice, subtitles, video, thumbnail."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.image_generator import async_prefetch_scene_images
from src.pipeline_context import PipelineContext
from src.pipeline_helpers import (
    _get_intro_duration,
    _get_shared_outro,
    _load_audio_durations,
    _needs_video_rebuild,
)
from src.script_generator import VideoScript
from src.subtitle_generator import generate_subtitles
from src.thumbnail_ab import generate_thumbnail_variants, pick_thumbnail
from src.video_generator import build_video
from src.voice_generator import async_synthesize_script


class _MediaBuilder:
    def __init__(
        self,
        script: VideoScript | None,
        ctx: PipelineContext,
        news: list[Any],
        summary: dict[str, Any],
        thompson_preferred_type: str | None,
    ) -> None:
        self._script = script
        self._run_dir = ctx.run_dir
        self._cp = ctx.cp
        self._observer = ctx.observer
        self._live = ctx.live
        self._seen = ctx.seen
        self._news = news
        self._summary = summary
        self._thompson_preferred_type = thompson_preferred_type
        # Results (populated by build_* methods)
        self.subtitle_path: Path | None = None
        self.en_intro: Path | None = None
        self.en_outro: Path | None = None
        self.long_video: Path | None = None
        self.short_video: Path | None = None
        self.thumbnail: Path | None = None

    def build_voice(self) -> None:
        assert self._script is not None
        audio_dir = self._run_dir / "audio"
        self._observer.step_start("voice")
        if not self._cp.is_done("voice"):
            if not audio_dir.exists() or len(list(audio_dir.glob("*.mp3"))) < len(self._script.scenes):
                # Run TTS and image pre-fetch concurrently: DALL-E and ElevenLabs
                # calls overlap in time instead of running sequentially, saving
                # several minutes on a typical episode.
                _script = self._script  # narrowed by assert above; captured for closure
                async def _voice_and_images() -> None:
                    await asyncio.gather(
                        async_synthesize_script(_script, self._run_dir),
                        async_prefetch_scene_images(_script, self._run_dir),
                    )
                asyncio.run(_voice_and_images())
                self._script.save(self._run_dir / "script.json")
            else:
                logger.info("Reusing cached audio; measuring durations")
                _load_audio_durations(self._script, audio_dir)
            total_dur = sum(s.duration_sec for s in self._script.scenes)
            self._live.log_event(f"Audio synthesized: {total_dur:.0f}s total")
            self._cp.mark_done("voice", {"total_duration_sec": total_dur})
            self._observer.step_done("voice", total_duration_sec=total_dur)
        else:
            logger.info("Step 3 (voice) already done — measuring cached audio durations")
            _load_audio_durations(self._script, audio_dir)
            self._observer.step_skip(
                "voice",
                total_duration_sec=sum(s.duration_sec for s in self._script.scenes),
            )

        self._summary["total_duration_sec"] = sum(s.duration_sec for s in self._script.scenes)

    def build_subtitles(self) -> None:
        assert self._script is not None
        en_intro = settings.source_dir / "ai-news-intro.mp4"
        self.en_intro = en_intro if en_intro.exists() else None
        self.en_outro = _get_shared_outro()

        cached_srt = self._run_dir / "subtitles.srt"
        self.subtitle_path = cached_srt if cached_srt.exists() else None

        self._observer.step_start("subtitles")
        if not self._cp.is_done("subtitles"):
            try:
                self.subtitle_path = generate_subtitles(
                    self._script,
                    self._run_dir / "audio",
                    self._run_dir / "subtitles.srt",
                    intro_duration=_get_intro_duration(self.en_intro),
                )
                self._cp.mark_done("subtitles")
                self._observer.step_done("subtitles")
            except Exception as exc:
                logger.warning(f"Subtitle generation failed (non-fatal): {exc}")
                self._observer.step_fail("subtitles", exc)
        else:
            logger.info("Step 4 (subtitles) already done")
            self._observer.step_skip("subtitles")

    def build_video(self) -> None:
        assert self._script is not None
        self.long_video = self._run_dir / "final_video.mp4"
        self._observer.step_start("video")
        rebuild_video = not self._cp.is_done("video") or _needs_video_rebuild(self.long_video)
        if rebuild_video:
            build_video(
                self._script, self._run_dir,
                intro_path=self.en_intro,
                outro_path=self.en_outro,
                use_presenter=settings.presenter_enabled or settings.heygen_enabled,
            )
            self._seen.mark_featured(self._news)
            self._live.log_event(f"Video built: {self.long_video.name}")
            self._cp.mark_done("video", {"presenter": settings.presenter_enabled})
            self._observer.step_done(
                "video",
                file=self.long_video.name,
                presenter=settings.presenter_enabled,
            )
        else:
            logger.info(f"Step 5 (video) already done — reusing {self.long_video.name}")
            self._observer.step_skip("video")

        self.short_video = self._run_dir / "shorts.mp4"
        self.short_video = None

        from src.broll_fetcher import get_pexels_credits
        credits = get_pexels_credits(self._run_dir)
        if credits:
            self._script.description += (
                "\n\nVideo clips provided by Pexels:\n" + "\n".join(credits)
            )

    def build_thumbnail(self) -> None:
        assert self._script is not None
        assert self.long_video is not None
        self._observer.step_start("thumbnail")
        if not self._cp.is_done("thumbnail"):
            thumb_variants = generate_thumbnail_variants(
                self.long_video, self._script.title, self._run_dir
            )
            self.thumbnail = pick_thumbnail(thumb_variants, settings.data_dir, "daily")
            _style = self.thumbnail.stem.removeprefix("thumbnail_")
            self._cp.mark_done("thumbnail", {"style": _style})
            self._observer.step_done("thumbnail", style=_style, variants=len(thumb_variants))
        else:
            logger.info("Step 6 (thumbnail) already done")
            _style = self._cp.metadata("thumbnail").get("style", "default")
            self.thumbnail = self._run_dir / f"thumbnail_{_style}.jpg"
            if not self.thumbnail.exists():
                thumb_variants = generate_thumbnail_variants(
                    self.long_video, self._script.title, self._run_dir
                )
                self.thumbnail = pick_thumbnail(thumb_variants, settings.data_dir, "daily")
            self._observer.step_skip("thumbnail", style=_style)

        self._summary["thumbnail_style"] = self.thumbnail.stem.removeprefix("thumbnail_")
