"""ffmpeg text/graphic overlay helpers.

Functions that draw text or graphics on top of existing video:
title_card, overlay_title_card, burn_chyron, burn_captions.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

from loguru import logger

from src.ffmpeg_probe import _FONT, _esc, _run, duration, video_size

# ─────────────────────────────────────────────────────────────────────────────
# Overlay helpers
# ─────────────────────────────────────────────────────────────────────────────

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


def overlay_title_card(
    video: Path,
    heading: str,
    output: Path,
    *,
    duration_sec: float = 2.5,
) -> Path:
    """Overlay a full-screen title card on the first seconds of *video*.

    Unlike ``title_card()``, this preserves the source audio and does not extend
    the clip duration.
    """
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    for width in (34, 48, 64):
        lines = textwrap.wrap(heading, width=width)
        if len(lines) <= 2:
            break

    W, H = 1280, 720
    line_h = 70
    total_h = len(lines) * line_h
    mid_y = H // 2
    bar_above_y = mid_y - total_h // 2 - 20
    bar_below_y = mid_y + total_h // 2 + 14
    end_t = max(0.1, duration_sec)

    filters: list[str] = [
        (
            f"drawbox=x=0:y=0:w={W}:h={H}:color=0x0D1117:t=fill:"
            f"enable='between(t,0,{end_t:.2f})'"
        ),
        (
            f"drawbox=x=80:y={bar_above_y}:w={W - 160}:h=3:color=0x38BDF8:t=fill:"
            f"enable='between(t,0,{end_t:.2f})'"
        ),
        (
            f"drawbox=x=80:y={bar_below_y}:w={W - 160}:h=3:color=0x38BDF8:t=fill:"
            f"enable='between(t,0,{end_t:.2f})'"
        ),
    ]

    y_start = mid_y - total_h // 2
    for i, line in enumerate(lines):
        esc_line = _esc(line)
        y = y_start + i * line_h
        filters.append(
            f"drawtext=fontfile={_FONT}:text='{esc_line}'"
            f":fontcolor=white:fontsize=56"
            f":x=(w-text_w)/2:y={y}"
            f":shadowx=2:shadowy=2:shadowcolor=black"
            f":enable='between(t,0,{end_t:.2f})'"
        )

    _run([
        "ffmpeg", "-y",
        "-i", str(video),
        "-vf", ",".join(filters),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "copy",
        str(output),
    ])
    return output


def burn_chyron(
    video: Path,
    heading: str,
    output: Path,
    *,
    start_sec: float = 0.0,
) -> Path:
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
        start_sec = max(0.0, min(start_sec, dur))
        chyron_end = min(dur, start_sec + 4.0)
        esc_heading = _esc(heading)
        vf = (
            f"drawtext=fontfile={_FONT}:text='{esc_heading}'"
            f":fontcolor=white:fontsize=48"
            f":x=(w-text_w)/2:y=h-100"
            f":shadowx=2:shadowy=2:shadowcolor=black@0.8"
            f":enable='between(t,{start_sec:.2f},{chyron_end:.2f})'"
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
