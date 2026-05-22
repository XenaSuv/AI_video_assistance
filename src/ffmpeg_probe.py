"""ffmpeg/ffprobe probe and info helpers.

Shared internal helpers (_run, _esc, constants) and functions that inspect
media without modifying it: probe, duration, video_size, has_audio_stream,
ensure_audio_stream, get_frame.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

# Font used for on-screen text
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Per-operation subprocess timeouts (seconds).
# ffmpeg can hang indefinitely on corrupted/truncated media; without a timeout
# the GitHub Actions job consumes its full 180-min budget before failing.
# Configurable via FFMPEG_PROBE_TIMEOUT / FFMPEG_CLIP_TIMEOUT / FFMPEG_LONG_TIMEOUT.
_PROBE_TIMEOUT: int = settings.ffmpeg_probe_timeout
_CLIP_TIMEOUT:  int = settings.ffmpeg_clip_timeout
_LONG_TIMEOUT:  int = settings.ffmpeg_long_timeout


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run(args: list[str], timeout: int = _CLIP_TIMEOUT) -> subprocess.CompletedProcess[bytes]:
    """Run an ffmpeg/ffprobe command.  Logs at DEBUG; raises on non-zero exit."""
    logger.debug("ffmpeg cmd: " + " ".join(str(a) for a in args))
    try:
        return subprocess.run(  # type: ignore[return-value]
            args, capture_output=True, text=True, check=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"ffmpeg timed out after {timeout}s — possible corrupted input or hung encoder. "
            f"Command: {' '.join(str(a) for a in args[:4])}…"
        ) from exc


def _esc(text: str) -> str:
    """Escape *text* for use inside an ffmpeg drawtext filter value.

    Applies two layers of escaping (order matters):

    1. Strip control characters — newlines, null bytes, and other non-printable
       chars would silently corrupt the filter string passed to ffmpeg.
    2. FFmpeg filter-graph escaping — backslash, colon, square brackets, and
       the shell-convention single-quote escape (FFmpeg mirrors POSIX here).
    3. drawtext printf expansion — `%` introduces `%{pts}`, `%{expr:…}` etc.;
       must be doubled to produce a literal percent sign.
    """
    # Strip control characters (keep normal spaces)
    text = "".join(ch for ch in text if ch == " " or ch.isprintable())
    # FFmpeg filter-graph level (backslash must come first)
    text = text.replace("\\", "\\\\")
    text = text.replace("'",  r"'\''")
    text = text.replace(":",  r"\:")
    text = text.replace("[",  r"\[")
    text = text.replace("]",  r"\]")
    # drawtext printf expansion — must be after backslash escaping
    text = text.replace("%",  "%%")
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Probe / info
# ─────────────────────────────────────────────────────────────────────────────

def probe(path: Path) -> dict[str, Any]:
    """Return ffprobe JSON output for *path*."""
    result = _run([
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        str(path),
    ], timeout=_PROBE_TIMEOUT)
    data: dict[str, Any] = json.loads(result.stdout)
    return data


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


def ensure_audio_stream(path: Path, output: Path) -> Path:
    """Return *path* if it already has audio, else mux in a silent AAC track."""
    if has_audio_stream(path):
        return path
    if output.exists():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)

    dur = duration(path)
    _run([
        "ffmpeg", "-y",
        "-i", str(path),
        "-f", "lavfi",
        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={dur:.3f}",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        str(output),
    ])
    return output


def get_frame(video: Path, t: float) -> "np.ndarray[Any, np.dtype[np.uint8]]":
    """Extract a single RGB frame at time *t* seconds. Returns H×W×3 uint8 array."""
    w, h = video_size(video)
    try:
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
            timeout=_PROBE_TIMEOUT,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"get_frame timed out after {_PROBE_TIMEOUT}s extracting frame at t={t} from {video}"
        ) from exc
    arr = np.frombuffer(result.stdout, dtype=np.uint8)
    return arr.reshape((h, w, 3))
