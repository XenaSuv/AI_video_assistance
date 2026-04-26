"""Assemble the final 15-minute video by stitching DALL-E + Ken Burns clips,
overlaying chyrons, and mixing in narration audio.

B-roll generation is handled by image_generator.generate_scene_clip().
"""
from __future__ import annotations

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
from moviepy.video.fx.all import fadein, fadeout, loop
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.script_generator import Scene, VideoScript
from src.image_generator import generate_scene_clip

_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# --------------------- Per-scene clip generation ---------------------

def generate_clips_for_scene(scene: Scene, out_dir: Path) -> list[Path]:
    """Return a list containing the single Ken Burns clip for this scene."""
    return [generate_scene_clip(scene, out_dir)]


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
) -> Path:
    """Combine generated clips + narration into the final 16:9 video."""
    segments = []

    for scene in script.scenes:
        # Audio is the master clock — trim video to exactly match it.
        narration   = AudioFileClip(str(audio_paths_by_scene[scene.idx]))
        target_dur  = narration.duration   # float, authoritative

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
        logger.info(f"Scene {scene.idx} assembled: {target_dur:.1f}s")

    final = concatenate_videoclips(segments, method="compose")
    final = fadein(fadeout(final, 1.0), 1.0)

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


def build_video(script: VideoScript, out_dir: Path) -> Path:
    """End-to-end video creation."""
    clip_dir = out_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_paths_by_scene = {
        s.idx: generate_clips_for_scene(s, clip_dir) for s in script.scenes
    }
    audio_paths_by_scene = {
        s.idx: out_dir / "audio" / f"scene_{s.idx:02d}.mp3" for s in script.scenes
    }

    output_path = out_dir / "final_video.mp4"
    return assemble_video(script, clip_paths_by_scene, audio_paths_by_scene, output_path)
