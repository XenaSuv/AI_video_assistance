"""A/B thumbnail generation with Thompson-sampling style selection.

Three visually distinct styles are generated from every video. The style that
historically produces the highest CTR is picked more often via a Beta bandit.

Styles
------
bottom_gradient  – current default: best-scored frame + dark bottom-third
                   gradient + white title text (most neutral baseline)
impact_text      – same frame + bold yellow/stroke text centred, no gradient,
                   sky-blue accent bar — more "clickbait" energy
top_bar          – frame sampled from 20-30 % into video (likely a scene
                   transition, different content) + reversed top gradient
                   + title in the upper area

Public API
----------
generate_thumbnail_variants(video_path, title, out_dir, context_key) -> list[Path]
    Produce up to 3 variant thumbnails (cached).  Returns paths.

pick_thumbnail(variant_paths, data_dir, context_key) -> Path
    Thompson-sample the style arms; return the winning thumbnail path.

record_thumbnail_usage(style, video_id, data_dir, context_key) -> None
    Persist the chosen style so performance can be linked later.

record_thumbnail_performance(video_id, ctr_pct, data_dir) -> None
    Update the arm's Beta distribution from YouTube Analytics CTR data.

top_styles(data_dir, context_key, n) -> list[dict]
    Ranked arm stats for reporting.
"""
from __future__ import annotations

import json
import random
import sys
import textwrap
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.ffmpeg_utils as ffmpeg_utils

THUMB_W, THUMB_H = 1280, 720
_FONT_PATH       = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_CANDIDATES      = 20   # frames sampled for scoring

# Style identifiers — must be stable (used as arm keys in the bandit)
STYLE_BOTTOM   = "bottom_gradient"
STYLE_IMPACT   = "impact_text"
STYLE_TOP      = "top_bar"
_ALL_STYLES    = [STYLE_BOTTOM, STYLE_IMPACT, STYLE_TOP]


# ─────────────────── shared helpers ───────────────────

def _load_font(size: int):
    try:
        return ImageFont.truetype(_FONT_PATH, size)
    except OSError:
        return ImageFont.load_default()


def _score_frame(arr: np.ndarray) -> float:
    mean = float(arr.mean())
    if mean < 35 or mean > 225:
        return -1.0
    colorfulness = float(arr.std(axis=(0, 1)).mean())
    contrast     = float(arr.std())
    return colorfulness * 0.65 + contrast * 0.35


def _wrap(title: str, max_chars: int = 38) -> list[str]:
    lines = textwrap.wrap(title, width=max_chars)
    if len(lines) > 2:
        lines = textwrap.wrap(title, width=max_chars + 12)[:2]
    return lines


def _sample_frames(video_path: Path, n: int = _CANDIDATES) -> list[tuple[float, np.ndarray]]:
    dur = ffmpeg_utils.duration(video_path)
    timestamps = [dur * (0.05 + 0.90 * i / (n - 1)) for i in range(n)]
    return [(t, ffmpeg_utils.get_frame(video_path, t)) for t in timestamps]


def _best_frame(frames: list[tuple[float, np.ndarray]]) -> np.ndarray:
    return max(frames, key=lambda x: _score_frame(x[1]))[1]


def _mid_frame(video_path: Path) -> np.ndarray:
    """A frame from ~20-30 % of the video — different content from the scored best."""
    dur = ffmpeg_utils.duration(video_path)
    t = dur * 0.25
    return ffmpeg_utils.get_frame(video_path, t)


def _to_pil(arr: np.ndarray) -> Image.Image:
    img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return img.resize((THUMB_W, THUMB_H), Image.LANCZOS)


def _save(img: Image.Image, path: Path) -> Path:
    img.save(str(path), "JPEG", quality=92, optimize=True)
    logger.info(f"Thumbnail variant saved: {path.name} ({path.stat().st_size // 1024} KB)")
    return path


# ─────────────────── style A: bottom gradient ───────────────────

def _style_bottom_gradient(base_img: Image.Image, title: str) -> Image.Image:
    w, h  = base_img.size
    img   = base_img.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    grad_h  = h // 3
    for y in range(grad_h):
        alpha = int(200 * (y / grad_h))
        draw.rectangle([(0, h - grad_h + y), (w, h - grad_h + y + 1)],
                       fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, overlay)
    draw2 = ImageDraw.Draw(img)
    font  = _load_font(54)
    lines = _wrap(title)
    lh    = 54 + 10
    y0    = h - len(lines) * lh - 28
    for i, line in enumerate(lines):
        y = y0 + i * lh
        draw2.text((22, y + 3), line, font=font, fill=(0, 0, 0, 220))
        draw2.text((20, y),     line, font=font, fill=(255, 255, 255, 255))
    return img.convert("RGB")


# ─────────────────── style B: impact text ───────────────────

def _style_impact_text(base_img: Image.Image, title: str) -> Image.Image:
    """No gradient. Bold yellow centred text + sky-blue accent bar at bottom."""
    w, h  = base_img.size
    img   = base_img.copy().convert("RGBA")

    # Slight dark vignette (corners only) so text is readable on any background
    vig   = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dv    = ImageDraw.Draw(vig)
    for r in range(120):
        alpha = int(140 * (1 - r / 120) ** 2)
        dv.rectangle([(r, r), (w - r, h - r)], outline=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, vig)

    draw  = ImageDraw.Draw(img)
    font  = _load_font(64)
    lines = _wrap(title, max_chars=30)
    lh    = 64 + 14
    total = len(lines) * lh
    y0    = (h - total) // 2 - 20       # centred, slightly above middle

    for i, line in enumerate(lines):
        bbox  = font.getbbox(line)
        x     = (w - (bbox[2] - bbox[0])) // 2
        y     = y0 + i * lh
        # thick black stroke
        for dx in range(-4, 5):
            for dy in range(-4, 5):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), line, font=font,
                              fill=(0, 0, 0, 255))
        draw.text((x, y), line, font=font, fill=(255, 230, 0, 255))

    # Sky-blue accent bar at bottom
    draw.rectangle([(0, h - 14), (w, h)], fill=(56, 189, 248, 255))

    return img.convert("RGB")


# ─────────────────── style C: top bar ───────────────────

def _style_top_bar(base_img: Image.Image, title: str) -> Image.Image:
    """Gradient over the top quarter, title text in top area (different composition)."""
    w, h  = base_img.size
    img   = base_img.convert("RGBA")
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)
    grad_h  = h // 4
    for y in range(grad_h):
        alpha = int(200 * (1 - y / grad_h))
        draw.rectangle([(0, y), (w, y + 1)], fill=(0, 0, 0, alpha))
    img = Image.alpha_composite(img, overlay)
    draw2 = ImageDraw.Draw(img)
    font  = _load_font(52)
    lines = _wrap(title)
    lh    = 52 + 10
    y0    = 22
    for i, line in enumerate(lines):
        y = y0 + i * lh
        draw2.text((22, y + 3), line, font=font, fill=(0, 0, 0, 220))
        draw2.text((20, y),     line, font=font, fill=(255, 255, 255, 255))

    # Thin accent bar at the very top
    draw2.rectangle([(0, 0), (w, 8)], fill=(56, 189, 248, 255))

    return img.convert("RGB")


# ─────────────────── frame rendering ───────────────────

_STYLE_RENDERERS = {
    STYLE_BOTTOM: _style_bottom_gradient,
    STYLE_IMPACT: _style_impact_text,
    STYLE_TOP:    _style_top_bar,
}


# ─────────────────── bandit storage ───────────────────

def _perf_file(data_dir: Path) -> Path:
    return data_dir / "thumbnail_performance.json"


def _load_perf(data_dir: Path) -> dict:
    p = _perf_file(data_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def _save_perf(data: dict, data_dir: Path) -> None:
    _perf_file(data_dir).write_text(json.dumps(data, indent=2))


def _get_arm(ctx: dict, style: str) -> dict:
    if style not in ctx:
        ctx[style] = {"alpha": 1.0, "beta": 1.0, "uses": 0, "wins": 0.0}
    return ctx[style]


# ─────────────────── public API ───────────────────

def generate_thumbnail_variants(
    video_path: Path,
    title: str,
    out_dir: Path,
) -> list[Path]:
    """Generate (or reuse cached) 3 thumbnail variants. Returns their paths."""
    frames = _sample_frames(video_path)
    best   = _to_pil(_best_frame(frames))
    mid    = _to_pil(_mid_frame(video_path))

    # Style A and B use the best-scored frame; Style C uses the mid-video frame.
    source = {
        STYLE_BOTTOM: best,
        STYLE_IMPACT: best,
        STYLE_TOP:    mid,
    }

    paths: list[Path] = []
    for style in _ALL_STYLES:
        out_path = out_dir / f"thumbnail_{style}.jpg"
        if out_path.exists():
            logger.info(f"Reusing cached thumbnail variant: {out_path.name}")
            paths.append(out_path)
            continue
        rendered = _STYLE_RENDERERS[style](source[style].copy(), title)
        _save(rendered, out_path)
        paths.append(out_path)

    return paths


def pick_thumbnail(
    variant_paths: list[Path],
    data_dir: Path,
    context_key: str,
) -> Path:
    """Thompson-sample style arms, return the winning thumbnail path."""
    if len(variant_paths) == 1:
        return variant_paths[0]

    data = _load_perf(data_dir)
    ctx  = data.setdefault(context_key, {})

    # Map style name back from filename (thumbnail_{style}.jpg)
    styles = [p.stem.removeprefix("thumbnail_") for p in variant_paths]

    scores  = [random.betavariate(_get_arm(ctx, s)["alpha"],
                                  _get_arm(ctx, s)["beta"]) for s in styles]
    best_i  = scores.index(max(scores))
    chosen  = variant_paths[best_i]
    logger.info(
        f"[thumbnail_ab/{context_key}] picked style={styles[best_i]} "
        f"(score={scores[best_i]:.3f}): {chosen.name}"
    )
    _save_perf(data, data_dir)
    return chosen


def record_thumbnail_usage(
    thumbnail_path: Path,
    video_id: str,
    data_dir: Path,
    context_key: str,
) -> None:
    """Log which thumbnail style went to *video_id*."""
    style = thumbnail_path.stem.removeprefix("thumbnail_")
    data  = _load_perf(data_dir)
    ctx   = data.setdefault(context_key, {})
    arm   = _get_arm(ctx, style)
    arm["uses"] += 1
    data.setdefault("_video_thumbnails", {})[video_id] = {
        "context": context_key,
        "style":   style,
    }
    _save_perf(data, data_dir)
    logger.info(f"[thumbnail_ab] recorded usage: video={video_id} style={style}")


def record_thumbnail_performance(
    video_id: str,
    ctr_pct: float,
    data_dir: Path,
    context_key: str | None = None,
) -> None:
    """Update arm Beta params from YouTube Analytics CTR.

    reward = ctr_pct / 10, clamped to [0, 1].
    (A 5 % CTR → 0.5 reward; a 10 %+ CTR → full reward.)
    """
    data  = _load_perf(data_dir)
    entry = data.get("_video_thumbnails", {}).get(video_id)
    if not entry:
        logger.warning(f"[thumbnail_ab] no record for video_id={video_id}")
        return

    ctx_key = context_key or entry["context"]
    style   = entry["style"]
    ctx     = data.setdefault(ctx_key, {})
    arm     = _get_arm(ctx, style)

    reward       = min(1.0, max(0.0, ctr_pct / 10.0))
    arm["alpha"] += reward
    arm["beta"]  += 1.0 - reward
    arm["wins"]  += reward

    logger.info(
        f"[thumbnail_ab/{ctx_key}] style={style}: reward={reward:.3f}, "
        f"alpha={arm['alpha']:.2f}, beta={arm['beta']:.2f}"
    )
    _save_perf(data, data_dir)


def top_styles(data_dir: Path, context_key: str, n: int = 5) -> list[dict]:
    """Return top-n thumbnail styles by estimated mean CTR for *context_key*."""
    data = _load_perf(data_dir)
    ctx  = data.get(context_key, {})
    arms = [
        {
            "style":  style,
            "mean":   a["alpha"] / (a["alpha"] + a["beta"]),
            "uses":   a["uses"],
            "wins":   a["wins"],
        }
        for style, a in ctx.items()
        if style in _ALL_STYLES
    ]
    return sorted(arms, key=lambda x: x["mean"], reverse=True)[:n]
