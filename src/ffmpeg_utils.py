"""Utility wrappers around ffmpeg/ffprobe subprocesses.

All functions accept Path objects and return Path or simple values.
Intermediate files go to out_dir/assembled/ for cache-ability.

Internal helpers:
  _run(args)   – subprocess.run with check=True, logs at DEBUG level
  _esc(text)   – escape text for ffmpeg drawtext filter
"""
from __future__ import annotations

import json
import math
import subprocess
import textwrap
from pathlib import Path
from typing import Union

import numpy as np
from loguru import logger

# Font used for on-screen text
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(args: list[str]) -> subprocess.CompletedProcess:
    """Run an ffmpeg/ffprobe command.  Logs at DEBUG; raises on non-zero exit."""
    logger.debug("ffmpeg cmd: " + " ".join(str(a) for a in args))
    return subprocess.run(args, capture_output=True, text=True, check=True)


def _esc(text: str) -> str:
    """Escape *text* for use inside an ffmpeg drawtext filter value.

    Order matters: backslash first, then the others.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace("'",  r"'\''")
    text = text.replace(":",  r"\:")
    text = text.replace("[",  r"\[")
    text = text.replace("]",  r"\]")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Probe / info
# ─────────────────────────────────────────────────────────────────────────────

def probe(path: Path) -> dict:
    """Return ffprobe JSON output for *path*."""
    result = _run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ])
    return json.loads(result.stdout)


def duration(path: Path) -> float:
    """Return audio or video duration in seconds."""
    info = probe(path)
    # Try format duration first (most reliable for containers)
    fmt_dur = info.get("format", {}).get("duration")
    if fmt_dur:
        return float(fmt_dur)
    # Fallback: first stream with a duration
    for stream in info.get("streams", []):
        d = stream.get("duration")
        if d:
            return float(d)
    raise ValueError(f"Cannot determine duration of {path}")


def video_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of the first video stream."""
    info = probe(path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            return int(stream["width"]), int(stream["height"])
    raise ValueError(f"No video stream found in {path}")


def has_audio_stream(path: Path) -> bool:
    """Return True when *path* contains at least one audio stream."""
    if not path.exists():
        return False
    info = probe(path)
    return any(stream.get("codec_type") == "audio" for stream in info.get("streams", []))


def get_frame(video: Path, t: float) -> np.ndarray:
    """Extract a single RGB frame at time *t* seconds. Returns H×W×3 uint8 array."""
    w, h = video_size(video)
    result = subprocess.run(
        [
            "ffmpeg", "-v", "quiet",
            "-ss", str(t),
            "-i", str(video),
            "-frames:v", "1",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    arr = np.frombuffer(result.stdout, dtype=np.uint8)
    return arr.reshape((h, w, 3))


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

    Re-encodes with libx264/aac so all streams are compatible.
    Pass *video_only=True* when all inputs have no audio stream (e.g. quote
    card clip + raw Ken Burns clip before merge_av).
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    list_file = output.with_suffix("").parent / (output.stem + "_list.txt")
    try:
        list_file.write_text(
            "\n".join(f"file '{p.resolve()}'" for p in paths) + "\n"
        )
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        ]
        if video_only:
            cmd += ["-an"]
        else:
            cmd += ["-c:a", "aac"]
        cmd.append(str(output))
        _run(cmd)
    finally:
        list_file.unlink(missing_ok=True)
    return output


def title_card(
    heading: str,
    output: Path,
    *,
    duration_sec: float = 2.5,
) -> Path:
    """Render a full-screen title card using lavfi + drawtext + drawbox.

    Background: #0D1117 (dark navy)
    Accent bars: #38BDF8 (sky blue) above and below the centred heading.
    Fade in/out 0.4 s each side.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    # Wrap heading to at most 2 lines
    lines: list[str] = []
    for width in (34, 48, 64):
        lines = textwrap.wrap(heading, width=width)
        if len(lines) <= 2:
            break

    W, H = 1280, 720
    line_h = 70           # font size (56) + leading (14)
    total_h = len(lines) * line_h
    mid_y = H // 2

    # Accent bars: y above = mid - total_h//2 - 20; y below = mid + total_h//2 + 14
    bar_above_y = mid_y - total_h // 2 - 20
    bar_below_y = mid_y + total_h // 2 + 14

    # Build drawtext filters for each line
    filters: list[str] = []

    # Fade in/out
    fade = (
        f"fade=t=in:st=0:d=0.4:alpha=1,"
        f"fade=t=out:st={duration_sec - 0.4:.2f}:d=0.4:alpha=1"
    )

    # Accent bar above
    filters.append(
        f"drawbox=x=80:y={bar_above_y}:w={W - 160}:h=3:color=0x38BDF8:t=fill"
    )
    # Accent bar below
    filters.append(
        f"drawbox=x=80:y={bar_below_y}:w={W - 160}:h=3:color=0x38BDF8:t=fill"
    )

    # Text lines
    y_start = mid_y - total_h // 2
    for i, line in enumerate(lines):
        esc_line = _esc(line)
        y = y_start + i * line_h
        filters.append(
            f"drawtext=fontfile={_FONT}:text='{esc_line}'"
            f":fontcolor=white:fontsize=56"
            f":x=(w-text_w)/2:y={y}"
            f":shadowx=2:shadowy=2:shadowcolor=black"
        )

    vf = ",".join(filters) + "," + fade

    _run([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x0D1117:size={W}x{H}:rate=24:duration={duration_sec}",
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration_sec}",
        "-vf", vf,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "aac", "-shortest",
        str(output),
    ])
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


def burn_chyron(video: Path, heading: str, output: Path) -> Path:
    """Burn a lower-third chyron (heading text) onto *video* for the first few seconds.

    The chyron is shown for min(4, duration) seconds.
    White text with black shadow, centered horizontally near the bottom.
    Non-fatal: if the ffmpeg call fails, the original video path is returned.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        dur = duration(video)
        chyron_dur = min(4.0, dur)
        esc_heading = _esc(heading)
        vf = (
            f"drawtext=fontfile={_FONT}:text='{esc_heading}'"
            f":fontcolor=white:fontsize=48"
            f":x=(w-text_w)/2:y=h-100"
            f":shadowx=2:shadowy=2:shadowcolor=black@0.8"
            f":enable='between(t,0,{chyron_dur:.2f})'"
        )
        _run([
            "ffmpeg", "-y",
            "-i", str(video),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "copy",
            str(output),
        ])
    except Exception as exc:
        logger.warning(f"burn_chyron failed (non-fatal): {exc} — returning source video")
        return video
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
    frames: list[np.ndarray],
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
    for frame in frames:
        # Ensure the frame is the right size
        if frame.shape != (height, width, 3):
            from PIL import Image as _PIL
            frame = np.array(
                _PIL.fromarray(frame.astype(np.uint8)).resize((width, height))
            )
        proc.stdin.write(frame.astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
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
        ])
    except Exception as exc:
        logger.warning(f"mix_music failed (non-fatal): {exc} — returning source video")
        return video
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


def burn_captions(
    video: Path,
    text: str,
    output: Path,
    *,
    font_size: int = 80,
    y_pct: float = 0.60,
    color: str = "yellow",
) -> Path:
    """Burn animated captions (5-word chunks with timed enable) onto *video*.

    Each chunk is shown for an equal share of the total duration.
    Non-fatal: errors are logged and the source video path is returned.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    try:
        dur = duration(video)
        words  = text.split()
        chunks = [" ".join(words[i: i + 5]) for i in range(0, len(words), 5)]
        if not chunks:
            return video

        per    = dur / max(len(chunks), 1)
        _, h   = video_size(video)
        y_pos  = int(h * y_pct)

        drawtext_filters = []
        for i, chunk in enumerate(chunks):
            t_start = i * per
            t_end   = (i + 1) * per
            esc_chunk = _esc(chunk)
            drawtext_filters.append(
                f"drawtext=fontfile={_FONT}:text='{esc_chunk}'"
                f":fontcolor={color}:fontsize={font_size}"
                f":x=(w-text_w)/2:y={y_pos}"
                f":borderw=4:bordercolor=black"
                f":enable='between(t,{t_start:.3f},{t_end:.3f})'"
            )

        vf = ",".join(drawtext_filters)
        _run([
            "ffmpeg", "-y",
            "-i", str(video),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-c:a", "copy",
            str(output),
        ])
    except Exception as exc:
        logger.warning(f"burn_captions failed (non-fatal): {exc} — returning source video")
        return video
    return output
