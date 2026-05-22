"""Standalone Shorts Pipeline — Shorts as first-class growth engine.

STATUS: PRIMARY pipeline for Shorts production.
    Run directly (``python src/shorts_pipeline.py``) or call
    ``ShortsPipeline().run()`` programmatically.
    Uses ShortsEngineV2 for content generation and ShortsExperimentEngine
    for hypothesis tracking and learning.
    The clip-from-long-form helpers (build_short, create_short_video, etc.)
    that were previously in the deprecated ``src.shorts_generator`` module are
    now consolidated here for callers that generate derivative Shorts from an
    already-rendered long-form video.

This pipeline runs independently of the daily long-form pipeline.
It is NOT a "clip the video" pipeline. Every Short is:
  - built from scratch (no dependency on a long video)
  - a single idea in 25–30 seconds
  - a hypothesis test for hook effectiveness

Execution order:
    scrape top stories
        ↓
    ShortsEngineV2.generate() → 5 ShortScript objects
        ↓
    for each script:
        TTS via ElevenLabs  (real voice — same as long videos)
        render_short()      (fast-cut text + avatar if HeyGen enabled)
        upload_short()      → video_id
        register in ShortsExperimentEngine
        ↓
    (next pipeline run) collect_analytics() → learn from winners
        → feed best hook types back into daily pipeline

Run:
    python src/shorts_pipeline.py
    python src/shorts_pipeline.py --dry-run   # generate scripts only, no upload
    python src/shorts_pipeline.py --story "OpenAI releases GPT-5"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.ffmpeg_utils as ffmpeg_utils
from config import settings
from src.scraper import scrape_all
from src.script_generator import VideoScript
from src.shorts_engine_v2 import ShortResult, ShortScript, ShortsEngineV2
from src.shorts_experiment_engine import ShortsExperimentEngine
from src.slack_notifier import notify_failure, notify_success
from src.viral_selector import pick_viral_news

SHORT_W, SHORT_H   = 1080, 1920
SHORT_MAX_SECONDS  = 58   # YouTube Shorts hard-limit is 60


# ── Editorial helper ──────────────────────────────────────────────────────────

def _story_to_editorial(story: dict[str, Any]) -> dict[str, Any]:
    """Convert a scraped story dict to the editorial format ShortsEngineV2 expects."""
    title   = story.get("title", "AI")
    summary = story.get("summary", story.get("description", title))
    words   = title.split()
    topic   = " ".join(words[:4]) if words else "AI"
    return {
        "topic":     topic,
        "core_fact": summary,
        "angle":     title,
    }


# ── Video rendering ───────────────────────────────────────────────────────────

def _synthesize_short_audio(script: ShortScript, out_dir: Path) -> Path | None:
    """Generate TTS audio for the full narration of *script*.

    Uses ElevenLabs with the same voice as the daily pipeline.
    Returns the mp3 path or None on failure.
    """
    try:
        from elevenlabs import ElevenLabs
        client   = ElevenLabs(api_key=settings.elevenlabs_api_key)
        audio_dir = out_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / f"{script.short_id}.mp3"
        if audio_path.exists():
            return audio_path

        chunks = client.text_to_speech.convert(
            text=script.full_narration,
            voice_id=settings.elevenlabs_voice_id,
            model_id=settings.elevenlabs_model,
            output_format="mp3_44100_128",
        )
        with open(audio_path, "wb") as f:
            for chunk in chunks:
                if chunk:
                    f.write(chunk)
        logger.info(f"Short audio synthesized: {audio_path.name}")
        return audio_path
    except Exception as exc:
        logger.warning(f"Short TTS failed (non-fatal): {exc}")
        return None


def _render_avatar_short(
    script: ShortScript,
    audio_path: Path | None,
    out_dir: Path,
) -> Path | None:
    """Render a HeyGen avatar clip via AvatarEngine (intent-aware, with subtitles and zoom)."""
    if not settings.heygen_enabled:
        return None
    try:
        from src.avatar_engine import AvatarEngine
        engine   = AvatarEngine()
        out_path = out_dir / f"avatar_{script.short_id}.mp4"
        clip = engine.render(
            audio_path or Path("/dev/null"),
            out_path,
            intent       = script.hook_type,   # conflict/curiosity/mistake/warning/insider
            persona      = script.persona,      # direct/serious/curious
            is_shorts    = True,
            subtitle_text= script.full_narration,
            aspect       = "9:16",
        )
        if clip:
            logger.info(f"AvatarEngine: Short clip ready — {clip.name}")
        return clip
    except Exception as exc:
        logger.warning(f"Avatar render failed (non-fatal): {exc}")
        return None


def _render_text_short(
    script: ShortScript,
    audio_path: Path | None,
    out_dir: Path,
) -> Path:
    """Render a text-overlay Short with B-roll background and animated subtitles.

    Rendering layers (bottom → top):
      1. Background — Pexels/Pixabay B-roll at 9:16; falls back to dark gradient
      2. Hook text  — large centred caption at ~35% height for the full run
      3. Subtitles  — animated 5-word chunks of the full narration at ~72% height
      4. Audio      — TTS narration track
      5. Music      — optional background track at low volume
    """
    import shutil

    import src.ffmpeg_utils as ffmpeg_utils
    from src.broll_fetcher import fetch_broll_for_short

    assembled = out_dir / "assembled"
    assembled.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"text_{script.short_id}.mp4"
    if out_path.exists():
        return out_path

    duration = 28.0
    if audio_path and audio_path.exists():
        try:
            duration = min(ffmpeg_utils.duration(audio_path), 58.0)
        except Exception:
            pass

    # ── Layer 1: Background ───────────────────────────────────────────────────
    # Try B-roll from Pexels/Pixabay using the topic; fall back to dark clip.
    broll_path = assembled / f"broll_{script.short_id}.mp4"
    base = fetch_broll_for_short(
        script.topic, duration, broll_path, out_w=SHORT_W, out_h=SHORT_H,
    )
    if base is None:
        logger.info(f"Short [{script.hook_type}]: no B-roll available — using dark background")
        dark = assembled / f"base_{script.short_id}.mp4"
        ffmpeg_utils.black_clip(dark, width=SHORT_W, height=SHORT_H, duration_sec=duration)
        base = dark

    # ── Layer 2: Hook text (large, upper-centre) ──────────────────────────────
    hook_overlay = assembled / f"hook_{script.short_id}.mp4"
    hook_overlay = ffmpeg_utils.burn_captions(
        base,
        script.hook,
        hook_overlay,
        font_size=88,
        y_pct=0.35,
        color="white",
    )

    # ── Layer 3: Animated narration subtitles (bottom third) ─────────────────
    subtitle_text = " ".join([script.core, script.twist, script.ending])
    subtitled = assembled / f"subtitled_{script.short_id}.mp4"
    subtitled = ffmpeg_utils.burn_captions(
        hook_overlay,
        subtitle_text,
        subtitled,
        font_size=58,
        y_pct=0.72,
        color="yellow",
    )

    # ── Layer 4: Audio ────────────────────────────────────────────────────────
    if audio_path and audio_path.exists():
        with_audio = assembled / f"audio_{script.short_id}.mp4"
        ffmpeg_utils.merge_av(subtitled, audio_path, with_audio)
        final: Path = with_audio
    else:
        final = subtitled

    # ── Layer 5: Background music ─────────────────────────────────────────────
    music_raw = settings.background_music_path.strip()
    if music_raw:
        music_path = Path(music_raw) if Path(music_raw).is_absolute() else settings.source_dir / music_raw
        if music_path.exists():
            try:
                with_music = assembled / f"music_{script.short_id}.mp4"
                ffmpeg_utils.mix_music(
                    final, music_path, with_music,
                    volume=settings.shorts_music_volume,
                    fade_in=0.5, fade_out=1.0,
                )
                final = with_music
            except Exception as exc:
                logger.warning(f"Music mix failed (non-fatal): {exc}")

    shutil.copy2(str(final), str(out_path))
    logger.info(f"Text Short rendered: {out_path.name} ({duration:.0f}s)")
    return out_path


def render_short(
    script: ShortScript,
    out_dir: Path,
) -> Path | None:
    """Render a Short using avatar (preferred) or text fallback.

    Returns the final mp4 path or None on total failure.
    """
    audio_path = _synthesize_short_audio(script, out_dir)
    avatar     = _render_avatar_short(script, audio_path, out_dir)
    if avatar:
        return avatar
    try:
        return _render_text_short(script, audio_path, out_dir)
    except Exception as exc:
        logger.error(f"Short render failed completely: {exc}")
        return None


# ── Upload ────────────────────────────────────────────────────────────────────

def _upload(script: ShortScript, video_path: Path) -> str | None:
    """Upload *video_path* as a YouTube Short. Returns video_id or None."""
    try:
        from src.youtube_uploader import upload_short
        title = script.hook[:95] + ("…" if len(script.hook) > 95 else "")
        video_id = upload_short(
            video_path=video_path,
            title=title,
            description=(
                f"{script.hook}\n\n"
                f"{script.core}\n\n"
                f"#AINews #Shorts #AI #{script.hook_type}"
            ),
            tags=["AI", "AINews", "Shorts", script.hook_type, script.topic],
            client_secrets=settings.youtube_client_secrets,
            token_file=settings.youtube_token_file,
        )
        logger.info(f"Short uploaded: {video_id} [{script.hook_type}]")
        return video_id
    except Exception as exc:
        logger.warning(f"Short upload failed (non-fatal): {exc}")
        return None


# ── Pipeline ──────────────────────────────────────────────────────────────────

class ShortsPipeline:
    """Standalone Shorts pipeline — runs independently of the daily pipeline."""

    def run(
        self,
        dry_run: bool = False,
        story_override: str | None = None,
        max_shorts: int = 5,
    ) -> dict[str, Any]:
        date_str  = dt.date.today().isoformat()
        run_dir   = settings.output_dir / f"shorts_{date_str}"
        run_dir.mkdir(parents=True, exist_ok=True)

        _setup_logging(run_dir)
        logger.info(f"=== Shorts Pipeline: {date_str} ===")

        summary: dict[str, Any] = {
            "date":      date_str,
            "run_dir":   str(run_dir),
            "generated": 0,
            "rendered":  0,
            "uploaded":  0,
            "results":   [],
        }

        try:
            # 1. Get top story
            story = self._get_story(story_override)
            if not story:
                logger.error("Shorts Pipeline: no story available — aborting")
                summary["status"] = "no_story"
                return summary
            logger.info(f"Story: {story.get('title')}")

            # 2. Build editorial dict and generate Short scripts
            editorial = _story_to_editorial(story)
            engine    = ShortsEngineV2(data_dir=settings.data_dir)

            # Log best known combination to guide future generation strategy
            best_combo = engine.get_best_combination()
            if best_combo:
                logger.info(
                    f"ShortsV2 best combo so far: "
                    f"[{best_combo['hook_type']}|{best_combo['style']}|{best_combo['persona']}] "
                    f"avg_retention_3s={best_combo['avg_retention_3s']}"
                )

            scripts = engine.generate(editorial, max_shorts=max_shorts)
            summary["generated"] = len(scripts)
            logger.info(f"Generated {len(scripts)} Short scripts")

            if dry_run:
                logger.info("Dry run — stopping after script generation")
                summary["status"]  = "dry_run"
                summary["scripts"] = [s.to_dict() for s in scripts]
                return summary

            # 3. Collect deferred analytics from previous run's Shorts
            from src.hook_pattern_engine import HookPatternEngine
            exp_engine     = ShortsExperimentEngine(data_dir=settings.data_dir)
            pattern_engine = HookPatternEngine(data_dir=settings.data_dir)
            new_results = exp_engine.collect_analytics(min_age_hours=4.0)
            if new_results:
                analysis = exp_engine.analyze()
                logger.info(
                    f"ShortsExperiment: {new_results} new result(s), "
                    f"winners={analysis['winners']}, best={analysis['best_hook_type']!r}"
                )
                # Feed analytics into ShortsEngineV2's learning store and
                # HookPatternEngine so both levels of learning improve.
                prev_scripts = {s.short_id: s for s in engine.load_scripts()}
                for exp_result in exp_engine._store.load_results():
                    script = prev_scripts.get(exp_result.experiment_id)
                    if script:
                        metrics = {
                            "retention_3s": exp_result.retention_3s,
                            "avg_watch":    exp_result.avg_watch_pct,
                            "completion":   exp_result.completion_rate,
                        }
                        engine.learn(script, metrics)
                        # Feature-level learning: WHY did this hook work?
                        pattern_engine.record(
                            hook_text    = script.hook,
                            hook_type    = script.hook_type,
                            retention_3s = exp_result.retention_3s,
                            short_id     = script.short_id,
                        )

            # 4. Render + upload each Short
            results: list[dict[str, Any]] = []
            for script in scripts:
                result = ShortResult(short_id=script.short_id)
                try:
                    video_path = render_short(script, run_dir)
                    if video_path:
                        result.video_path = str(video_path)
                        summary["rendered"] += 1

                        video_id = _upload(script, video_path)
                        if video_id:
                            result.video_id  = video_id
                            result.uploaded  = True
                            summary["uploaded"] += 1

                            # Register in ShortsExperimentEngine so the next
                            # run's collect_analytics() can fetch real metrics.
                            # We create a minimal experiment record using the
                            # script's short_id as the experiment_id.
                            from src.shorts_experiment_engine import ShortsExperiment
                            exp = ShortsExperiment(
                                experiment_id = script.short_id,
                                story_title   = story.get("title", ""),
                                hook_type     = script.hook_type,
                                hook_text     = script.hook,
                                core_text     = script.core,
                                payoff_text   = script.ending,
                                video_id      = video_id,
                                video_path    = str(video_path),
                            )
                            exp_engine._store.save_experiment(exp)
                            logger.info(f"Experiment registered: {script.short_id}")
                    else:
                        result.error = "render_failed"
                except Exception as exc:
                    result.error = str(exc)
                    logger.warning(f"Short [{script.hook_type}] failed: {exc}")
                results.append(result.to_dict())

            summary["results"] = results
            summary["status"]  = "done"
            _save_summary(run_dir, summary)
            notify_success(summary, "shorts")
            logger.info(
                f"=== Shorts Pipeline done: "
                f"{summary['uploaded']}/{summary['generated']} uploaded ==="
            )

        except Exception as exc:
            logger.error(f"Shorts Pipeline failed: {exc}")
            logger.error(traceback.format_exc())
            summary["status"] = "failed"
            summary["error"]  = str(exc)
            notify_failure(exc, "shorts", summary, traceback.format_exc())

        return summary

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_story(self, override: str | None) -> dict[str, Any] | None:
        if override:
            return {"title": override, "source": "manual", "summary": override}
        try:
            news = scrape_all()
            viral = pick_viral_news(news, top_n=3)
            top   = (viral or news)
            if not top:
                return None
            item = top[0]
            return {
                "title":       item.title,
                "source":      getattr(item, "source", ""),
                "summary":     getattr(item, "summary", item.title),
                "description": getattr(item, "description", ""),
            }
        except Exception as exc:
            logger.warning(f"Story fetch failed: {exc}")
            return None


def _setup_logging(run_dir: Path) -> None:
    from src.log_filter import scrub
    logger.remove()
    logger.add(
        sys.stderr, level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<7}</level> | {message}",
        filter=scrub,
    )
    logger.add(run_dir / "shorts_run.log", level="DEBUG", rotation="10 MB", filter=scrub)


def _save_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    try:
        (run_dir / "shorts_summary.json").write_text(json.dumps(summary, indent=2))
    except Exception as exc:
        logger.warning(f"Could not save summary: {exc}")


# ── Clip-from-long-form helpers ───────────────────────────────────────────────
# These functions produce a derivative Short from an already-rendered long-form
# video.  They were previously in the deprecated src/shorts_generator.py; that
# module has been deleted.  New code should use ShortsPipeline for independent
# Shorts, but these helpers remain for callers that need clip-based generation
# (breaking_main, weekly_main, topic_main, language_variant).


def _saliency_crop_x(frame: np.ndarray, crop_w: int) -> int:  # type: ignore[type-arg]
    """Return x-offset of the most visually interesting *crop_w*-wide window.

    Uses horizontal edge density (column-wise gradient magnitude) as a proxy
    for visual saliency.  Pure numpy — no extra dependencies.
    Falls back to center if the frame is too small.
    """
    h, w = frame.shape[:2]
    max_offset = w - crop_w
    if max_offset <= 0:
        return 0
    gray   = frame.mean(axis=2).astype(np.float32)
    edges  = np.abs(np.diff(gray, axis=1))          # H × (W-1)
    col_sc = edges.sum(axis=0)                       # saliency per column
    wins   = np.convolve(col_sc, np.ones(crop_w, dtype=np.float32), mode="valid")
    best   = int(np.argmax(wins))
    # Blend toward center (weight 0.3) to avoid extreme off-center crops
    center = (w - crop_w) // 2
    return int(best * 0.7 + center * 0.3)


def _resolve_music_path() -> Path | None:
    """Return the background music Path, or None if not configured / file missing."""
    raw = settings.background_music_path.strip()
    if not raw:
        return None
    p = Path(raw)
    if not p.is_absolute():
        p = settings.source_dir / raw
    if not p.exists():
        logger.warning(f"Background music not found: {p} — skipping")
        return None
    return p


def build_short(
    script: VideoScript,
    main_video: Path,
    out_dir: Path,
    *,
    audio_subdir: str = "audio",
    out_name: str = "shorts.mp4",
) -> Path:
    """Produce a Shorts-ready mp4 by cropping and clipping an existing long-form video.

    *audio_subdir* lets language variants point at e.g. ``audio_ru/``.
    *out_name* lets variants write ``shorts_ru.mp4`` alongside ``shorts.mp4``.
    """
    assembled_dir = out_dir / "assembled"
    assembled_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / out_name

    # Audio is the master clock — video is trimmed to match, not the other way.
    audio_path = out_dir / audio_subdir / "scene_00.mp3"
    if audio_path.exists():
        target_duration = min(ffmpeg_utils.duration(audio_path), SHORT_MAX_SECONDS)
    else:
        audio_path      = None  # type: ignore[assignment]
        target_duration = SHORT_MAX_SECONDS

    logger.info(f"Short target duration: {target_duration:.1f}s")

    clip_dir     = out_dir / "clips"
    source_clips = sorted(clip_dir.glob("scene_00_clip_*.mp4")) \
                 + sorted(clip_dir.glob("scene_01_clip_*.mp4"))
    if not source_clips:
        logger.warning("No source clips found, falling back to main video head")
        source_clips = [main_video]

    # 1. Concat source clips
    if len(source_clips) == 1:
        base_path = source_clips[0]
    else:
        cat_path = assembled_dir / "short_source_cat.mp4"
        base_path = ffmpeg_utils.concat(source_clips, cat_path, video_only=True)

    # 2. Loop/trim to target duration
    looped_path = assembled_dir / "short_looped.mp4"
    looped_path = ffmpeg_utils.loop_and_trim(base_path, looped_path, target_sec=target_duration)

    # 3. Saliency crop_x
    crop_x: int | None
    try:
        mid_t     = target_duration / 2
        mid_frame = ffmpeg_utils.get_frame(looped_path, mid_t)
        src_w, src_h = ffmpeg_utils.video_size(looped_path)
        target_ratio = SHORT_W / SHORT_H
        crop_w = int(src_h * target_ratio)
        crop_x = _saliency_crop_x(mid_frame, crop_w)
    except Exception as exc:
        logger.warning(f"Saliency crop failed (non-fatal): {exc} — using centre crop")
        crop_x = None

    # 4. make_vertical
    vertical_path = assembled_dir / "short_vertical.mp4"
    vertical_path = ffmpeg_utils.make_vertical(
        looped_path, vertical_path, crop_x=crop_x, out_w=SHORT_W, out_h=SHORT_H
    )

    # 5. merge_av with audio
    if audio_path is not None:
        with_audio = assembled_dir / "short_with_audio.mp4"
        with_audio = ffmpeg_utils.merge_av(vertical_path, audio_path, with_audio)
    else:
        with_audio = vertical_path

    # 6. burn_captions
    with_captions = assembled_dir / "short_with_captions.mp4"
    with_captions = ffmpeg_utils.burn_captions(
        with_audio, script.hook, with_captions,
        font_size=80, y_pct=0.60, color="yellow",
    )

    # 7. mix_music (non-fatal if file missing)
    music_path = _resolve_music_path()
    if music_path:
        final_path = assembled_dir / "short_with_music.mp4"
        result = ffmpeg_utils.mix_music(
            with_captions, music_path, final_path,
            volume=settings.shorts_music_volume,
            fade_in=1.0, fade_out=1.5,
        )
        shutil.copy2(str(result), str(out))
    else:
        shutil.copy2(str(with_captions), str(out))

    logger.info(f"Short written: {out} ({target_duration:.0f}s)")
    return out


def add_big_captions(video: Path, text: str, output: Path) -> Path:
    """Burn large centered captions onto a short-friendly clip."""
    output.parent.mkdir(parents=True, exist_ok=True)
    return ffmpeg_utils.burn_captions(
        video, text, output,
        font_size=72, y_pct=0.45, color="white",
    )


def create_short_video(
    script_text: str,
    out_dir: Path,
    *,
    out_name: str = "shorts_mvp.mp4",
    duration_sec: float = 20.0,
    audio_path: Path | None = None,
) -> Path:
    """Create a static Shorts video with big burned-in captions.

    Used as a fallback when no source video is available (e.g. breaking news
    with no associated long-form clip).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    assembled_dir = out_dir / "assembled"
    assembled_dir.mkdir(parents=True, exist_ok=True)

    base_clip = assembled_dir / "shorts_mvp_base.mp4"
    captioned  = assembled_dir / "shorts_mvp_captioned.mp4"
    out_path   = out_dir / out_name

    ffmpeg_utils.black_clip(base_clip, width=1080, height=1920, duration_sec=duration_sec)
    add_big_captions(base_clip, script_text, captioned)
    if audio_path is not None and audio_path.exists():
        with_audio = assembled_dir / "shorts_mvp_with_audio.mp4"
        ffmpeg_utils.merge_av(captioned, audio_path, with_audio)
        shutil.copy2(str(with_audio), str(out_path))
    else:
        shutil.copy2(str(captioned), str(out_path))
    logger.info(f"Created short video: {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Shorts Pipeline")
    parser.add_argument("--dry-run",  action="store_true", help="Generate scripts only, no upload")
    parser.add_argument("--story",    default=None,        help="Override top story title")
    parser.add_argument("--max",      type=int, default=5, help="Max Shorts per run (default 5)")
    args = parser.parse_args()

    pipeline = ShortsPipeline()
    result   = pipeline.run(
        dry_run=args.dry_run,
        story_override=args.story,
        max_shorts=args.max,
    )
    sys.exit(0 if result.get("status") in ("done", "dry_run") else 1)


if __name__ == "__main__":
    main()
