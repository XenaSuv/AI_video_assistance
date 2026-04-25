"""Generate b-roll clips via RunwayML Gen-3 Alpha Turbo, then assemble
the full 15-minute video by stitching clips, overlaying chyrons, and
mixing in the narration audio.

RunwayML API flow:
  POST /v1/image_to_video or /v1/text_to_video  →  {"id": "<task_id>"}
  GET  /v1/tasks/<task_id>                       →  poll until status == "SUCCEEDED"
"""
from __future__ import annotations

"""RunwayML video generation - updated for current SDK."""
from runwayml import RunwayML, TaskFailedError


# Replace the OLD constants:
# RUNWAY_BASE = "https://api.dev.runwayml.com/v1"
# Remove _runway_headers() entirely
# Keep RUNWAY_CLIP_SECONDS, POLL_INTERVAL, POLL_TIMEOUT

import sys
import time
from pathlib import Path

import requests
from loguru import logger
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    TextClip,
    VideoFileClip,
    concatenate_videoclips,
    ColorClip,
)
from moviepy.video.fx.all import fadein, fadeout, loop

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import Scene, VideoScript


RUNWAY_BASE = "https://api.dev.runwayml.com/v1"
RUNWAY_CLIP_SECONDS = 10  # Gen-3 Turbo produces 5s or 10s clips
POLL_INTERVAL = 8
POLL_TIMEOUT = 600  # 10 min per clip


# --------------------- RunwayML API ---------------------

def _is_retryable_runway(exc: BaseException) -> bool:
    # TaskFailedError means the prompt was rejected or the job itself failed — don't retry.
    return not isinstance(exc, TaskFailedError)


def _log_runway_retry(retry_state) -> None:
    logger.warning(
        f"RunwayML retry {retry_state.attempt_number}: {retry_state.outcome.exception()}"
    )


def _runway_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.runwayml_api_key}",
        "Content-Type": "application/json",
        "X-Runway-Version": "2024-11-06",
    }

@retry(
    retry=retry_if_exception(_is_retryable_runway),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(4),
    before_sleep=_log_runway_retry,
    reraise=True,
)
def generate_runway_clip(prompt: str, duration: int = 10, seed: int | None = None) -> bytes:
    """Submit a text-to-video job using the official SDK and return mp4 bytes."""
    client = RunwayML(api_key=settings.runwayml_api_key)
    
    logger.info(f"RunwayML: submitting prompt: {prompt[:80]}...")
    
    try:
        kwargs = {
            "model": "gen4.5",        # ← was "gen4_turbo"
            "prompt_text": prompt,
            "ratio": "1280:720",
            "duration": 10 if duration > 5 else 5,
        }
        if seed is not None:
            kwargs["seed"] = seed
        
        # text_to_video, NOT image_to_video
        task = client.text_to_video.create(**kwargs).wait_for_task_output()
        
        video_url = task.output[0]
        dl = requests.get(video_url, timeout=120)
        dl.raise_for_status()
        return dl.content
        
    except TaskFailedError as e:
        raise RuntimeError(f"Runway task failed: {e.task_details}")


def generate_clips_for_scene(scene: Scene, out_dir: Path) -> list[Path]:
    """Produce as many ~10s clips as needed to cover the scene's narration."""
    n_clips = max(1, (scene.duration_sec + RUNWAY_CLIP_SECONDS - 1) // RUNWAY_CLIP_SECONDS)
    paths: list[Path] = []

    for i in range(n_clips):
        out = out_dir / f"scene_{scene.idx:02d}_clip_{i}.mp4"
        if out.exists():
            logger.info(f"Reusing cached clip: {out.name}")
            paths.append(out)
            continue
        try:
            # Slight prompt variation across clips of same scene to avoid
            # looking like a loop.
            prompt = scene.visual_prompt
            if i > 0:
                prompt += f" --variation {i}"
            data = generate_runway_clip(prompt, duration=RUNWAY_CLIP_SECONDS, seed=scene.idx * 100 + i)
            out.write_bytes(data)
            paths.append(out)
        except Exception as e:
            logger.error(f"Runway clip failed for scene {scene.idx} clip {i}: {e}")
            # Graceful fallback: reuse first clip of this scene if we have one
            if paths:
                paths.append(paths[0])
            else:
                # Last resort: plain black placeholder
                paths.append(_make_black_placeholder(out, RUNWAY_CLIP_SECONDS))

    return paths


def _make_black_placeholder(path: Path, duration: int) -> Path:
    """Fallback if Runway can't produce a clip. Dark gradient, not pure black."""
    clip = ColorClip(size=(1280, 768), color=(15, 20, 30), duration=duration)
    clip.write_videofile(str(path), fps=24, codec="libx264", audio=False, logger=None)
    clip.close()
    return path


# --------------------- Assembly ---------------------

def _chyron(heading: str, duration: float) -> TextClip:
    """Lower-third text graphic for scene heading."""
    txt = TextClip(
        heading,
        fontsize=48,
        color="white",
        font="DejaVu-Sans-Bold",
        stroke_color="black",
        stroke_width=2,
    ).set_duration(duration).set_position(("center", 650))
    return fadein(fadeout(txt, 0.5), 0.5)


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
        # Concatenate the scene's Runway clips, then trim/loop to narration length
        scene_clips = [VideoFileClip(str(p)).without_audio()
                       for p in clip_paths_by_scene[scene.idx]]
        video_seg = concatenate_videoclips(scene_clips, method="compose")
        if video_seg.duration < scene.duration_sec:
            video_seg = loop(video_seg, duration=scene.duration_sec)
        video_seg = video_seg.subclip(0, scene.duration_sec)

        # Overlay chyron for first 4 seconds
        chyron = _chyron(scene.heading, min(4.0, scene.duration_sec))
        composite = CompositeVideoClip([video_seg, chyron])

        # Attach narration audio for this scene
        narration = AudioFileClip(str(audio_paths_by_scene[scene.idx]))
        composite = composite.set_audio(narration)

        segments.append(composite)
        logger.info(f"Scene {scene.idx} assembled: {composite.duration:.1f}s")

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
