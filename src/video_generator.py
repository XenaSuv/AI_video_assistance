"""Assemble the final 15-minute video by stitching DALL-E + Ken Burns clips,
overlaying chyrons, and mixing in narration audio.

B-roll generation is handled by image_generator.generate_scene_clip().
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import Scene, VideoScript
from src.image_generator import generate_scene_clip
from src.quote_card import render_quote_card_png
from src.end_card import render_end_card_png as render_end_card
from src.presenter import generate_presenter_clip
from src.voice_generator import synthesize_hook
import src.ffmpeg_utils as ffmpeg_utils

VIDEO_W, VIDEO_H = 1280, 720   # canonical output resolution


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


# --------------------- Per-scene clip generation ---------------------

def generate_clips_for_scene(
    scene: Scene,
    out_dir: Path,
    *,
    tool: str | None = None,
    run_dir: Path | None = None,
    is_breaking: bool = False,
) -> list[Path]:
    """Return a list containing the single clip for this scene.

    *tool*        — enables real-screenshot capture for weekly tutorials.
    *run_dir*     — passed to broll_fetcher so Pexels credits are recorded.
    *is_breaking* — activates Stable Diffusion instead of DALL-E.
    """
    return [generate_scene_clip(scene, out_dir, tool=tool, run_dir=run_dir, is_breaking=is_breaking)]


# --------------------- Assembly ---------------------

def assemble_video(
    script: VideoScript,
    clip_paths_by_scene: dict[int, list[Path]],
    audio_paths_by_scene: dict[int, Path],
    output_path: Path,
    *,
    intro_path: Path | None = None,
    outro_path: Path | None = None,
    presenter_clip: Path | None = None,
) -> Path:
    """Combine generated clips + narration into the final 16:9 video.

    When *intro_path* / *outro_path* are supplied and the files exist they are
    prepended / appended as-is (audio included). All intermediate files are
    cached in out_dir/assembled/ so re-runs skip already-built segments.
    """
    assembled_dir = output_path.parent / "assembled"
    assembled_dir.mkdir(parents=True, exist_ok=True)

    ordered_segments: list[Path] = []

    # Presenter hook clip — prepended before all scenes
    if presenter_clip and presenter_clip.exists():
        pw, ph = ffmpeg_utils.video_size(presenter_clip)
        if (pw, ph) != (VIDEO_W, VIDEO_H):
            logger.info(f"Presenter: resizing {pw}×{ph} → {VIDEO_W}×{VIDEO_H}")
            presenter_resized = assembled_dir / "presenter_resized.mp4"
            presenter_clip = ffmpeg_utils.resize_video(
                presenter_clip, presenter_resized,
                width=VIDEO_W, height=VIDEO_H,
                keep_audio=True,
            )
        ordered_segments.append(presenter_clip)
        logger.info(f"Presenter clip prepended: {presenter_clip.name}")

    for scene in script.scenes:
        audio_path = audio_paths_by_scene[scene.idx]
        target_dur = ffmpeg_utils.duration(audio_path)

        def _valid_video_clip(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                w, h = ffmpeg_utils.video_size(path)
                dur = ffmpeg_utils.duration(path)
                return w == VIDEO_W and h == VIDEO_H and dur > 0
            except Exception:
                return False

        # Merge video + audio (loop video if shorter than narration)
        clip_paths = clip_paths_by_scene[scene.idx]
        if len(clip_paths) == 1:
            raw_clip = clip_paths[0]
        else:
            cat_path = assembled_dir / f"scene_{scene.idx:02d}_cat.mp4"
            raw_clip = ffmpeg_utils.concat(clip_paths, cat_path)

        if not _valid_video_clip(raw_clip):
            logger.warning(
                f"Scene {scene.idx}: raw clip invalid or corrupt, using placeholder instead"
            )
            raw_clip = assembled_dir / f"scene_{scene.idx:02d}_placeholder_raw.mp4"
            raw_clip = ffmpeg_utils.black_clip(raw_clip, width=VIDEO_W, height=VIDEO_H, duration_sec=target_dur)

        # Quote card — prepended as first 4 s of visual (narration plays over it)
        has_quote = bool(scene.source_quote) and target_dur >= 8
        if has_quote:
            try:
                qc_png  = assembled_dir / f"quote_{scene.idx:02d}.png"
                qc_clip = assembled_dir / f"quote_{scene.idx:02d}.mp4"
                render_quote_card_png(scene, qc_png)
                ffmpeg_utils.quote_card_clip(qc_png, qc_clip, duration_sec=4.0)
                combined = assembled_dir / f"scene_{scene.idx:02d}_qc.mp4"
                raw_clip = ffmpeg_utils.concat([qc_clip, raw_clip], combined, video_only=True)
                logger.info(f"Scene {scene.idx}: quote card prepended")
            except Exception as exc:
                logger.warning(f"Scene {scene.idx}: quote card failed (non-fatal): {exc}")
                has_quote = False

        assembled_clip = assembled_dir / f"scene_{scene.idx:02d}.mp4"
        try:
            assembled_clip = ffmpeg_utils.merge_av(
                raw_clip, audio_path, assembled_clip, loop_video=True
            )
        except Exception as exc:
            logger.warning(
                f"Scene {scene.idx}: merge_av failed ({exc}), using placeholder video instead"
            )
            placeholder = assembled_dir / f"scene_{scene.idx:02d}_placeholder.mp4"
            placeholder = ffmpeg_utils.black_clip(
                placeholder,
                width=VIDEO_W,
                height=VIDEO_H,
                duration_sec=target_dur,
            )
            assembled_clip = ffmpeg_utils.merge_av(
                placeholder,
                audio_path,
                assembled_clip,
                loop_video=False,
            )

        # Chyron: skip for quote-card scenes (attribution line already identifies the scene)
        chyron_clip = assembled_dir / f"scene_{scene.idx:02d}_chyron.mp4"
        if not has_quote:
            chyron_clip = ffmpeg_utils.burn_chyron(assembled_clip, scene.heading, chyron_clip)
        else:
            chyron_clip = assembled_clip   # no chyron — quote card handles attribution

        # Title card before all scenes except the first
        if scene.idx > 0:
            card_path = assembled_dir / f"title_{scene.idx:02d}.mp4"
            card_path = ffmpeg_utils.title_card(scene.heading, card_path)
            ordered_segments.append(card_path)

        ordered_segments.append(chyron_clip)
        logger.info(
            f"Scene {scene.idx} assembled: {target_dur:.1f}s"
            + (" (+ title card)" if scene.idx > 0 else "")
        )

    # End card — appended after all scenes (background music covers it)
    if settings.end_card_enabled:
        try:
            ec_png = assembled_dir / "end_card.png"
            ec_mp4 = assembled_dir / "end_card.mp4"
            render_end_card(settings.channel_name, settings.channel_handle, ec_png,
                            cta=settings.channel_cta)
            ffmpeg_utils.end_card_clip(ec_png, ec_mp4, duration_sec=settings.end_card_duration)
            ordered_segments.append(ec_mp4)
            logger.info(f"End card appended ({settings.end_card_duration:.0f}s)")
        except Exception as exc:
            logger.warning(f"End card failed (non-fatal): {exc}")

    # Concatenate all scene segments into content.mp4
    content_path = assembled_dir / "content.mp4"
    content_path = ffmpeg_utils.concat(ordered_segments, content_path)

    # Mix background music (non-fatal)
    music_path = _resolve_music_path()
    if music_path:
        content_music = assembled_dir / "content_music.mp4"
        content_path = ffmpeg_utils.mix_music(
            content_path,
            music_path,
            content_music,
            volume=settings.background_music_volume,
        )

    # Wrap with intro / outro
    has_intro = intro_path and intro_path.exists()
    has_outro = outro_path and outro_path.exists()

    if has_intro or has_outro:
        parts: list[Path] = []
        if has_intro:
            # Resize intro if needed
            intro_w, intro_h = ffmpeg_utils.video_size(intro_path)
            if (intro_w, intro_h) != (VIDEO_W, VIDEO_H):
                logger.info(
                    f"Intro: resizing {intro_w}×{intro_h} → {VIDEO_W}×{VIDEO_H}"
                )
                intro_resized = assembled_dir / "intro_resized.mp4"
                intro_path = ffmpeg_utils.resize_video(
                    intro_path, intro_resized, width=VIDEO_W, height=VIDEO_H
                )
            parts.append(intro_path)
            logger.info(f"Intro: {intro_path.name}")

        parts.append(content_path)

        if has_outro:
            outro_w, outro_h = ffmpeg_utils.video_size(outro_path)
            if (outro_w, outro_h) != (VIDEO_W, VIDEO_H):
                logger.info(
                    f"Outro: resizing {outro_w}×{outro_h} → {VIDEO_W}×{VIDEO_H}"
                )
                outro_resized = assembled_dir / "outro_resized.mp4"
                outro_path = ffmpeg_utils.resize_video(
                    outro_path, outro_resized, width=VIDEO_W, height=VIDEO_H
                )
            parts.append(outro_path)
            logger.info(f"Outro: {outro_path.name}")

        final_path = ffmpeg_utils.concat(parts, output_path)
    else:
        # No intro/outro — just copy content to output_path
        import shutil
        shutil.copy2(str(content_path), str(output_path))
        final_path = output_path

    logger.info(f"Final video written: {final_path}")
    return final_path


def build_video(
    script: VideoScript,
    out_dir: Path,
    *,
    tool: str | None = None,
    intro_path: Path | None = None,
    outro_path: Path | None = None,
    is_breaking: bool = False,
    use_presenter: bool = False,
) -> Path:
    """End-to-end video creation.

    *tool*          – pass claude/chatgpt/gemini for weekly tutorials (enables
                      real-screenshot capture); None → DALL-E / B-roll / infographic.
    *is_breaking*   – activates Stable Diffusion as the image generator.
    *use_presenter* – generate a D-ID talking-head hook clip (weekly only).
    *intro_path / outro_path* – prepended / appended when the file exists.
    """
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_paths_by_scene = {
        s.idx: generate_clips_for_scene(s, clip_dir, tool=tool, run_dir=out_dir, is_breaking=is_breaking)
        for s in script.scenes
    }
    audio_paths_by_scene = {
        s.idx: out_dir / "audio" / f"scene_{s.idx:02d}.mp3" for s in script.scenes
    }

    # Presenter hook clip (D-ID) — weekly tutorials only
    presenter_clip: Path | None = None
    if use_presenter and settings.presenter_enabled:
        try:
            hook_audio = synthesize_hook(script, out_dir)
            avatar_path = Path(settings.presenter_avatar_path)
            if not avatar_path.is_absolute():
                avatar_path = settings.source_dir.parent / avatar_path
            presenter_clip = generate_presenter_clip(
                hook_audio,
                out_dir / "assembled" / "presenter_hook.mp4",
                api_key=settings.did_api_key,
                avatar_path=avatar_path,
            )
        except Exception as exc:
            logger.warning(f"Presenter generation failed (non-fatal): {exc}")

    output_path = out_dir / "final_video.mp4"
    return assemble_video(
        script, clip_paths_by_scene, audio_paths_by_scene, output_path,
        intro_path=intro_path,
        outro_path=outro_path,
        presenter_clip=presenter_clip,
    )
