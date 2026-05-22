"""ffmpeg audio/video composition and transform helpers.

Functions that combine or reshape media: merge_av, concat, ken_burns,
frames_to_video, mix_music, quote_card_clip, end_card_clip, black_clip,
make_vertical, zoom_clip, loop_and_trim, resize_video.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path
from typing import Any, Union

import numpy as np
from loguru import logger

from src.ffmpeg_probe import _LONG_TIMEOUT, _run, duration, video_size


# ─────────────────────────────────────────────────────────────────────────────
# Audio / video construction
# ─────────────────────────────────────────────────────────────────────────────

def merge_av(
    video: Path,
    audio: Path,
    output: Path,
    *,
    loop_video: bool = False,
) -> Path:
    """Merge separate video + audio tracks into *output*.

    If *loop_video* is True the video is looped to match the audio duration.
    Looping requires re-encoding (libx264) because -stream_loop with -c:v copy
    produces duplicate timestamps and causes ffmpeg to exit with error 228.
    When not looping, stream copy is used to avoid unnecessary re-encoding.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    if loop_video:
        _run([
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(video),
            "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "aac",
            "-shortest",
            str(output),
        ])
    else:
        _run([
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac",
            "-shortest",
            str(output),
        ])
    return output


def concat(paths: list[Path], output: Path, *, video_only: bool = False) -> Path:
    """Concatenate *paths* into *output* using the concat demuxer.

    Tries stream copy first (fast, lossless); falls back to re-encoding with
    veryfast preset when inputs have incompatible stream parameters.
    Pass *video_only=True* when all inputs have no audio stream (e.g. quote
    card clip + raw Ken Burns clip before merge_av).
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    list_file = output.with_suffix("").parent / (output.stem + "_list.txt")
    try:
        list_file.write_text(
            "\n".join(
                "file '{}'".format(str(p.resolve()).replace("'", "'\\''"))
                for p in paths
            ) + "\n"
        )

        # Fast path: stream copy — no re-encoding when inputs share codec/resolution.
        copy_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
        ]
        if video_only:
            copy_cmd += ["-an"]
        copy_cmd.append(str(output))
        try:
            _run(copy_cmd, timeout=_LONG_TIMEOUT)
            return output
        except Exception:
            output.unlink(missing_ok=True)
            logger.debug("concat: stream copy failed, falling back to re-encode")

        # Fallback: re-encode with veryfast preset to stay within CI time limits.
        encode_cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        ]
        if video_only:
            encode_cmd += ["-an"]
        else:
            encode_cmd += ["-c:a", "aac"]
        encode_cmd.append(str(output))
        _run(encode_cmd, timeout=_LONG_TIMEOUT)
    finally:
        list_file.unlink(missing_ok=True)
    return output


def ken_burns(
    img_path: Path,
    output: Path,
    *,
    duration_sec: float,
    variant: int,
    in_w: int = 1792,
    in_h: int = 1024,
    out_w: int = 1280,
    out_h: int = 720,
) -> Path:
    """Animate a static image with a Ken Burns (pan/zoom) effect.

    variant 0: pan left → right
    variant 1: pan right → left
    variant 2: zoom in
    variant 3: zoom out

    All ffmpeg filter values are Python literals — no iw/ih expressions — so
    the filter graph is valid even when -loop 1 defers stream initialization.
    When the source image is smaller than out_w × out_h (e.g. 1024×1024
    DALL-E images vs 1280×720 output), kb_w/kb_h are scaled up first so that
    pan and crop offsets are never negative.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    dur = float(duration_sec)

    # Scale canvas up to at least out_w × out_h preserving aspect ratio.
    ratio = max(out_w / in_w, out_h / in_h)
    if ratio > 1.0:
        kb_w = max(math.ceil(in_w * ratio), out_w)
        kb_h = max(math.ceil(in_h * ratio), out_h)
    else:
        kb_w, kb_h = in_w, in_h

    pan = int((kb_w - out_w) * 0.40)   # always >= 0
    y0  = int((kb_h - out_h) * 0.25)   # always >= 0

    if variant == 0:
        # Pan left → right: simple static crop
        vf = (
            f"scale={kb_w}:{kb_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}:x={int(pan*0.5)}:y={y0},"
            f"scale={out_w}:{out_h}"
        )
    elif variant == 1:
        # Pan right → left: simple static crop
        vf = (
            f"scale={kb_w}:{kb_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}:x={int(pan*0.8)}:y={y0},"
            f"scale={out_w}:{out_h}"
        )
    elif variant == 2:
        # Zoom in: simple static centre crop
        vf = (
            f"scale={kb_w}:{kb_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}:x={int((kb_w-out_w)/2)}:y={int((kb_h-out_h)/2)},"
            f"scale={out_w}:{out_h}"
        )
    else:
        # Zoom out: simple static centre crop
        vf = (
            f"scale={kb_w}:{kb_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h}:x={int((kb_w-out_w)/2)}:y={int((kb_h-out_h)/2)},"
            f"scale={out_w}:{out_h}"
        )

    _run([
        "ffmpeg", "-y",
        "-loop", "1",
        "-framerate", "24",
        "-i", str(img_path),
        "-vf", vf,
        "-t", str(dur),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-an",
        str(output),
    ])
    return output


def frames_to_video(
    frames: list["np.ndarray[Any, np.dtype[np.uint8]]"],
    output: Path,
    *,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Pipe a list of RGB24 numpy frames to ffmpeg → libx264 MP4."""
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-an",
            str(output),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for frame in frames:
        # Ensure the frame is the right size
        if frame.shape != (height, width, 3):
            from PIL import Image as _PIL
            frame = np.array(
                _PIL.fromarray(frame.astype(np.uint8)).resize((width, height))
            )
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    try:
        proc.wait(timeout=_LONG_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(
            f"frames_to_video ffmpeg timed out after {_LONG_TIMEOUT}s — "
            f"{len(frames)} frames at {fps}fps"
        ) from None
    if proc.returncode != 0:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(f"frames_to_video ffmpeg failed: {err}")
    return output


def mix_music(
    video: Path,
    music: Path,
    output: Path,
    *,
    volume: float = 0.10,
    fade_in: float = 2.0,
    fade_out: float = 3.0,
) -> Path:
    """Mix looped background music under the video's existing audio.

    The music is looped to match the video duration, volume-adjusted,
    and faded in/out.  The result is amixed with the original audio track.
    Non-fatal: errors are logged and the source video path is returned.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        vid_dur = duration(video)
        fade_out_start = max(0.0, vid_dur - fade_out)
        # amix filter: loop music, apply volume + fades, mix with original
        af = (
            f"[1:a]aloop=loop=-1:size=2e9,atrim=duration={vid_dur:.3f},"
            f"volume={volume},"
            f"afade=t=in:st=0:d={fade_in},"
            f"afade=t=out:st={fade_out_start:.3f}:d={fade_out}"
            f"[bg];"
            f"[0:a][bg]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )
        _run([
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", str(music),
            "-filter_complex", af,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac",
            str(output),
        ], timeout=_LONG_TIMEOUT)
    except Exception as exc:
        logger.warning(f"mix_music failed (non-fatal): {exc} — returning source video")
        return video
    return output


def quote_card_clip(
    png_path: Path,
    output: Path,
    *,
    duration_sec: float = 4.0,
) -> Path:
    """Convert a static quote-card PNG into a short video with fade in/out.

    The clip is silent (no audio stream).  It is designed to be prepended to
    the scene's Ken Burns clip so the first *duration_sec* seconds of narration
    play over the quote card before cutting to the visual b-roll.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    fade_d = 0.35
    vf = (
        f"fade=t=in:st=0:d={fade_d},"
        f"fade=t=out:st={duration_sec - fade_d:.2f}:d={fade_d}"
    )
    _run([
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(png_path),
        "-vf", vf,
        "-t", str(duration_sec),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-an",
        str(output),
    ])
    return output


def end_card_clip(
    png_path: Path,
    output: Path,
    *,
    duration_sec: float = 10.0,
) -> Path:
    """Convert the end-card PNG into a video clip with silent AAC audio."""
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    fade_d = 0.5
    vf = (
        f"fade=t=in:st=0:d={fade_d},"
        f"fade=t=out:st={duration_sec - fade_d:.2f}:d={fade_d}"
    )
    _run([
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", str(png_path),
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration_sec}",
        "-vf", vf,
        "-t", str(duration_sec),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-shortest",
        "-pix_fmt", "yuv420p",
        str(output),
    ])
    return output


def black_clip(
    output: Path,
    *,
    width: int = 1280,
    height: int = 720,
    duration_sec: float,
) -> Path:
    """Generate a solid dark (#0F1417) placeholder video clip."""
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    _run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x0F1417:size={width}x{height}:rate=24:duration={duration_sec}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        str(output),
    ])
    return output


def make_vertical(
    video: Path,
    output: Path,
    *,
    crop_x: Union[int, None] = None,
    out_w: int = 1080,
    out_h: int = 1920,
) -> Path:
    """Crop + scale a landscape video to 9:16 vertical format.

    If *crop_x* is None, the crop is centred horizontally.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    src_w, src_h = video_size(video)
    target_ratio = out_w / out_h   # 0.5625
    src_ratio    = src_w / src_h

    if src_ratio > target_ratio:
        crop_w = int(src_h * target_ratio)
        if crop_x is None:
            crop_x = (src_w - crop_w) // 2
        vf = f"crop={crop_w}:{src_h}:{crop_x}:0,scale={out_w}:{out_h}"
    else:
        vf = f"scale={out_w}:{out_h}"

    _run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "copy",
        str(output),
    ])
    return output


def zoom_clip(
    video: Path,
    output: Path,
    *,
    scale: float = 1.15,
) -> Path:
    """Scale up and centre-crop a rendered video to create a zoom-in effect.

    Used by AvatarEngine for Shorts mode to push the face closer to the viewer.
    scale=1.15 → 15% tighter crop; safe default that avoids quality loss.
    Non-fatal: returns *video* unchanged on error.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        w, h = video_size(video)
        crop_w = int(w / scale)
        crop_h = int(h / scale)
        x = (w - crop_w) // 2
        y = (h - crop_h) // 2
        _run([
            "ffmpeg", "-y",
            "-i", str(video),
            "-vf", f"crop={crop_w}:{crop_h}:{x}:{y},scale={w}:{h}",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            str(output),
        ])
    except Exception as exc:
        logger.warning(f"zoom_clip failed (non-fatal): {exc}")
        return video
    return output


def loop_and_trim(
    video: Path,
    output: Path,
    *,
    target_sec: float,
) -> Path:
    """Loop *video* indefinitely and trim to *target_sec* seconds.

    Uses -stream_loop -1 + -t for fast copy when possible.
    Falls back to re-encode if -c copy fails (e.g. B-frames at loop point).
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        _run([
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", str(video),
            "-t", str(target_sec),
            "-c", "copy",
            str(output),
        ])
    except subprocess.CalledProcessError:
        logger.debug("loop_and_trim: -c copy failed, re-encoding")
        _run([
            "ffmpeg", "-y",
            "-stream_loop", "-1",
            "-i", str(video),
            "-t", str(target_sec),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",
            str(output),
        ])
    return output


def resize_video(
    video: Path,
    output: Path,
    *,
    width: int,
    height: int,
    keep_audio: bool = False,
) -> Path:
    """Scale *video* to *width*×*height*.

    Pass *keep_audio=True* to preserve the audio stream (e.g. presenter clips
    that have speech embedded).  Default strips audio for size efficiency.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    audio_flags = ["-c:a", "aac"] if keep_audio else ["-an"]
    _run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", f"scale={width}:{height}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        *audio_flags,
        str(output),
    ])
    return output
