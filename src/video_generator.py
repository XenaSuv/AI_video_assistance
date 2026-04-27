"""Assemble the final 15-minute video by stitching DALL-E + Ken Burns clips,
overlaying chyrons, and mixing in narration audio.

B-roll generation is handled by image_generator.generate_scene_clip().
"""
from __future__ import annotations

import textwrap
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    ColorClip,
)
from moviepy.video.fx.all import fadein, fadeout, loop, resize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.script_generator import Scene, VideoScript
from src.image_generator import generate_scene_clip

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
VIDEO_W, VIDEO_H = 1280, 720   # canonical output resolution


# --------------------- Per-scene clip generation ---------------------

def generate_clips_for_scene(scene: Scene, out_dir: Path, *, tool: str | None = None) -> list[Path]:
    """Return a list containing the single Ken Burns clip for this scene.

    *tool* enables real-screenshot capture for weekly tutorials when the scene
    has a screenshot_key. Daily news passes tool=None and always uses DALL-E.
    """
    return [generate_scene_clip(scene, out_dir, tool=tool)]


# --------------------- Assembly ---------------------

def _chyron(heading: str, duration: float) -> ImageClip:
    """Lower-third text graphic rendered with PIL — no ImageMagick required."""
    try:
        font = ImageFont.truetype(_FONT_PATH, 48)
    except OSError:
        font = ImageFont.load_default()

    # Draw heading onto a transparent 1280×720 canvas
    canvas = Image.new("RGBA", (1280, 720), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    bbox = font.getbbox(heading)
    text_w = bbox[2] - bbox[0]
    x = (1280 - text_w) // 2
    y = 640
    draw.text((x + 2, y + 2), heading, font=font, fill=(0, 0, 0, 200))   # shadow
    draw.text((x,     y    ), heading, font=font, fill=(255, 255, 255, 255))

    rgb   = np.array(canvas.convert("RGB"))
    alpha = np.array(canvas.split()[3]) / 255.0

    clip = (
        ImageClip(rgb, duration=duration)
        .set_mask(ImageClip(alpha, ismask=True, duration=duration))
    )
    return fadein(fadeout(clip, 0.5), 0.5)


_CARD_BG     = (13,  17,  23)   # dark navy
_CARD_ACCENT = (56, 189, 248)   # sky blue
_CARD_DUR    = 2.5              # seconds each title card is shown


def _title_card(heading: str) -> ImageClip:
    """Full-screen scene-divider card: dark background, accent bars, centered heading.

    Shown for _CARD_DUR seconds before each scene (except the opening hook) so
    viewers always know which topic is starting.
    """
    W, H = 1280, 720
    canvas = Image.new("RGB", (W, H), _CARD_BG)
    draw   = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype(_FONT_PATH, 56)
    except OSError:
        font = ImageFont.load_default()

    # Wrap long headings to at most 2 lines
    lines: list[str] = []
    for width in (34, 48):
        lines = textwrap.wrap(heading, width=width)
        if len(lines) <= 2:
            break

    line_h  = 56 + 14          # font size + leading
    total_h = len(lines) * line_h
    mid     = H // 2

    # Accent bar above text
    draw.rectangle(
        [(80, mid - total_h // 2 - 20), (W - 80, mid - total_h // 2 - 17)],
        fill=_CARD_ACCENT,
    )
    # Accent bar below text
    draw.rectangle(
        [(80, mid + total_h // 2 + 14), (W - 80, mid + total_h // 2 + 17)],
        fill=_CARD_ACCENT,
    )

    y_start = mid - total_h // 2
    for i, line in enumerate(lines):
        bbox  = font.getbbox(line)
        x     = (W - (bbox[2] - bbox[0])) // 2
        y     = y_start + i * line_h
        draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0))        # shadow
        draw.text((x,     y    ), line, font=font, fill=(255, 255, 255))   # text

    arr  = np.array(canvas)
    clip = ImageClip(arr, duration=_CARD_DUR)
    return fadein(fadeout(clip, 0.4), 0.4)


def _fit_clip_to_duration(video_path: Path, target_seconds: int) -> VideoFileClip:
    """Loop or trim a clip to exactly target_seconds."""
    clip = VideoFileClip(str(video_path)).without_audio()
    if clip.duration < target_seconds:
        clip = loop(clip, duration=target_seconds)
    return clip.subclip(0, target_seconds)


def assemble_video(
    script: VideoScript,
    clip_paths_by_scene: dict[int, list[Path]],
    audio_paths_by_scene: dict[int, Path],
    output_path: Path,
    *,
    intro_path: Path | None = None,
    outro_path: Path | None = None,
) -> Path:
    """Combine generated clips + narration into the final 16:9 video.

    When *intro_path* / *outro_path* are supplied and the files exist they are
    prepended / appended as-is (audio included). The global fade-in/out is only
    applied to the main content when NO intro/outro is present so that the
    branded clips handle their own transitions.
    """
    # ── main content ──────────────────────────────────────────────────────────
    segments = []
    for scene in script.scenes:
        if scene.idx > 0:
            segments.append(_title_card(scene.heading))

        narration  = AudioFileClip(str(audio_paths_by_scene[scene.idx]))
        target_dur = narration.duration

        scene_clips = [VideoFileClip(str(p)).without_audio()
                       for p in clip_paths_by_scene[scene.idx]]
        video_seg = concatenate_videoclips(scene_clips, method="compose")
        if video_seg.duration < target_dur:
            video_seg = loop(video_seg, duration=target_dur)
        video_seg = video_seg.subclip(0, target_dur)

        chyron    = _chyron(scene.heading, min(4.0, target_dur))
        composite = CompositeVideoClip([video_seg, chyron])
        composite = composite.set_audio(narration)

        segments.append(composite)
        logger.info(f"Scene {scene.idx} assembled: {target_dur:.1f}s"
                    + (" (+ title card)" if scene.idx > 0 else ""))

    content = concatenate_videoclips(segments, method="compose")

    # ── wrap with intro / outro ───────────────────────────────────────────────
    has_intro = intro_path and intro_path.exists()
    has_outro = outro_path and outro_path.exists()

    if has_intro or has_outro:
        # Branded clips handle their own fades; skip global fade on content.
        # Resize to the canonical resolution so concatenation never letterboxes.
        parts = []
        if has_intro:
            intro_clip = VideoFileClip(str(intro_path))
            if (intro_clip.w, intro_clip.h) != (VIDEO_W, VIDEO_H):
                logger.info(
                    f"Intro: resizing {intro_clip.w}×{intro_clip.h} → {VIDEO_W}×{VIDEO_H}"
                )
                intro_clip = resize(intro_clip, (VIDEO_W, VIDEO_H))
            parts.append(intro_clip)
            logger.info(f"Intro: {intro_path.name} ({intro_clip.duration:.1f}s)")
        parts.append(content)
        if has_outro:
            outro_clip = VideoFileClip(str(outro_path))
            if (outro_clip.w, outro_clip.h) != (VIDEO_W, VIDEO_H):
                logger.info(
                    f"Outro: resizing {outro_clip.w}×{outro_clip.h} → {VIDEO_W}×{VIDEO_H}"
                )
                outro_clip = resize(outro_clip, (VIDEO_W, VIDEO_H))
            parts.append(outro_clip)
            logger.info(f"Outro: {outro_path.name} ({outro_clip.duration:.1f}s)")
        final = concatenate_videoclips(parts, method="chain")
    else:
        final = fadein(fadeout(content, 1.0), 1.0)

    logger.info(f"Writing final video to {output_path} ({final.duration:.0f}s)")
    final.write_videofile(
        str(output_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        bitrate="6M",
        threads=4,
        preset="medium",
        logger=None,
    )
    final.close()
    return output_path


def build_video(
    script: VideoScript,
    out_dir: Path,
    *,
    tool: str | None = None,
    intro_path: Path | None = None,
    outro_path: Path | None = None,
) -> Path:
    """End-to-end video creation.

    *tool*        – pass claude/chatgpt/gemini for weekly tutorials (enables
                    real-screenshot capture); None always uses DALL-E.
    *intro_path*  – prepended as-is (with audio) when the file exists.
    *outro_path*  – appended as-is (with audio) when the file exists.
    """
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_paths_by_scene = {
        s.idx: generate_clips_for_scene(s, clip_dir, tool=tool) for s in script.scenes
    }
    audio_paths_by_scene = {
        s.idx: out_dir / "audio" / f"scene_{s.idx:02d}.mp3" for s in script.scenes
    }

    output_path = out_dir / "final_video.mp4"
    return assemble_video(
        script, clip_paths_by_scene, audio_paths_by_scene, output_path,
        intro_path=intro_path,
        outro_path=outro_path,
    )
