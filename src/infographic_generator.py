"""Animated infographic clips for data-rich scenes.

Generates 1280×720 MP4 clips using PIL (no extra deps beyond what's already
in the project).  Four types are supported:

  bar_chart   – horizontal bars growing left-to-right
  timeline    – events appearing on a horizontal line
  stat_card   – big number counting up with context text
  comparison  – two vertical bars growing simultaneously

Each renderer produces PIL Image frames; moviepy stitches them into an MP4.

Public API
----------
generate_infographic_clip(data, out_path, duration_sec) -> Path
    Create (or reuse cached) an animated infographic MP4.
    *data* must contain at least {"type": "<one of the four types>", ...}
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_W, OUT_H = 1280, 720
FPS          = 24
ANIM_SHARE   = 0.40   # first 40 % of the clip is animation; rest is hold

# ─── palette ──────────────────────────────────────────────────────────────────
_BG     = (13,  17,  23)       # #0D1117  dark navy
_ACCENT = (56, 189, 248)       # #38BDF8  sky blue
_WHITE  = (255, 255, 255)
_DIM    = (139, 156, 178)      # #8B9CB2
_GREEN  = (16,  185, 129)      # #10B981
_AMBER  = (245, 158,  11)      # #F59E0B
_PANEL  = (22,  27,  34)       # slightly lighter than BG for cards

_FONT  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_FONTR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _f(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_FONT, size)
    except OSError:
        return ImageFont.load_default()


def _fr(size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_FONTR, size)
    except OSError:
        return ImageFont.load_default()


def _ease_out(t: float) -> float:
    """Cubic ease-out: fast start, slow finish."""
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3


def _anim_t(frame_t: float) -> float:
    """Map overall frame time [0,1] to animation progress [0,1]."""
    return _ease_out(min(1.0, frame_t / ANIM_SHARE))


def _cx(draw: ImageDraw.Draw, text: str, font, y: int, color=_WHITE) -> None:
    """Draw *text* centred horizontally at *y*."""
    bbox = font.getbbox(text)
    x = (OUT_W - (bbox[2] - bbox[0])) // 2
    draw.text((x, y), text, font=font, fill=color)


def _title(draw: ImageDraw.Draw, text: str) -> None:
    font  = _f(42)
    lines = []
    for width in (46, 60, 80):
        import textwrap
        lines = textwrap.wrap(text, width=width)
        if len(lines) <= 2:
            break
    for i, line in enumerate(lines):
        _cx(draw, line, font, 28 + i * 50)


def _accent_bar(draw: ImageDraw.Draw, y: int = 8) -> None:
    draw.rectangle([(0, y), (OUT_W, y + 6)], fill=_ACCENT)


# ─────────────────── bar chart ────────────────────────────────────────────────

def _bar_chart_frame(data: dict, t: float) -> Image.Image:
    items  = data.get("items", [])
    title  = data.get("title", "")
    unit   = data.get("unit", "")
    n      = len(items)
    if n == 0:
        return Image.new("RGB", (OUT_W, OUT_H), _BG)

    img  = Image.new("RGB", (OUT_W, OUT_H), _BG)
    draw = ImageDraw.Draw(img)
    _accent_bar(draw)
    _title(draw, title)

    max_val   = max(it.get("value", 1) for it in items)
    bar_area_x1, bar_area_x2 = 310, 1200
    bar_h     = min(52, (OUT_H - 160) // n - 12)
    spacing   = (OUT_H - 160 - n * bar_h) // (n + 1)
    anim      = _anim_t(t)

    for i, item in enumerate(items):
        val   = float(item.get("value", 0))
        label = item.get("label", "")
        frac  = (val / max_val) * anim
        bar_w = int((bar_area_x2 - bar_area_x1) * frac)
        y_top = 140 + i * (bar_h + spacing)

        # Bar fill
        draw.rectangle(
            [(bar_area_x1, y_top), (bar_area_x1 + bar_w, y_top + bar_h)],
            fill=_ACCENT,
        )
        # Subtle bar BG
        draw.rectangle(
            [(bar_area_x1, y_top), (bar_area_x2, y_top + bar_h)],
            outline=(*_DIM, 60),
        )

        # Label (left, vertically centred on bar)
        lf = _f(32)
        ly = y_top + bar_h // 2 - 16
        bbox = lf.getbbox(label)
        draw.text((bar_area_x1 - (bbox[2] - bbox[0]) - 16, ly),
                  label, font=lf, fill=_WHITE)

        # Value at bar end
        if anim > 0.05:
            vf   = _fr(28)
            vtxt = f"{val:g}{unit}"
            vx   = bar_area_x1 + bar_w + 8
            vy   = y_top + bar_h // 2 - 14
            draw.text((vx, vy), vtxt, font=vf, fill=_DIM)

    return img


# ─────────────────── timeline ─────────────────────────────────────────────────

def _timeline_frame(data: dict, t: float) -> Image.Image:
    events = data.get("events", [])
    title  = data.get("title", "")
    n      = len(events)
    if n == 0:
        return Image.new("RGB", (OUT_W, OUT_H), _BG)

    img  = Image.new("RGB", (OUT_W, OUT_H), _BG)
    draw = ImageDraw.Draw(img)
    _accent_bar(draw)
    _title(draw, title)

    line_y   = OUT_H // 2 + 30
    x_margin = 100
    xs       = [x_margin + i * (OUT_W - 2 * x_margin) // (max(n - 1, 1)) for i in range(n)]

    # Reveal events one by one; each takes (1/n) of ANIM_SHARE
    events_visible = int(_anim_t(t) * n) + 1

    # Dotted timeline base
    for i in range(OUT_W):
        if i % 12 < 6:
            draw.point((i, line_y), fill=_DIM)

    # Draw visible segment solid
    if events_visible >= 2:
        x_end = xs[min(events_visible - 1, n - 1)]
        draw.line([(xs[0], line_y), (x_end, line_y)], fill=_ACCENT, width=3)

    for i, ev in enumerate(events[:events_visible]):
        x    = xs[i]
        date = ev.get("date", "")
        lbl  = ev.get("label", "")

        # Dot
        r = 10
        draw.ellipse([(x - r, line_y - r), (x + r, line_y + r)], fill=_ACCENT)
        draw.ellipse([(x - 5, line_y - 5), (x + 5, line_y + 5)], fill=_BG)

        df = _f(26)
        lf = _fr(30)

        if i % 2 == 0:   # date above, label below line
            draw.text((_centred_x(date, df), line_y - 55), date, font=df, fill=_DIM)
            draw.text((_centred_x(lbl, lf),  line_y + 28), lbl,  font=lf, fill=_WHITE)
        else:             # alternate: label above, date below
            draw.text((_centred_x(lbl, lf),  line_y - 72), lbl,  font=lf, fill=_WHITE)
            draw.text((_centred_x(date, df), line_y + 28), date, font=df, fill=_DIM)

    return img


def _centred_x(text: str, font) -> int:
    bbox = font.getbbox(text)
    return (OUT_W - (bbox[2] - bbox[0])) // 2


# ─────────────────── stat card ────────────────────────────────────────────────

def _parse_numeric(s: str) -> tuple[float, str, str]:
    """Return (number, suffix, prefix). E.g. '$4.2B' → (4.2, 'B', '$')."""
    s = str(s).strip()
    m = re.match(r'^([^0-9]*)([0-9]+(?:\.[0-9]+)?)([^0-9]*)$', s)
    if m:
        return float(m.group(2)), m.group(3), m.group(1)
    return 0.0, s, ""


def _stat_card_frame(data: dict, t: float) -> Image.Image:
    value   = str(data.get("value", "0"))
    label   = data.get("label", "")
    context = data.get("context", "")

    img  = Image.new("RGB", (OUT_W, OUT_H), _BG)
    draw = ImageDraw.Draw(img)
    _accent_bar(draw)

    # Panel
    draw.rounded_rectangle([(120, 100), (OUT_W - 120, OUT_H - 100)],
                            radius=24, fill=_PANEL)

    num, suffix, prefix = _parse_numeric(value)
    anim = _anim_t(t)

    if num > 0:
        current = num * anim
        if isinstance(num, float) and not num.is_integer():
            val_str = f"{prefix}{current:.1f}{suffix}"
        else:
            val_str = f"{prefix}{int(current)}{suffix}"
    else:
        # Non-numeric: just fade in (no animation)
        val_str = value

    # Big number
    vf = _f(128)
    bbox = vf.getbbox(val_str)
    vw = bbox[2] - bbox[0]
    vx = (OUT_W - vw) // 2
    draw.text((vx, OUT_H // 2 - 110), val_str, font=vf, fill=_ACCENT)

    # Label
    lf  = _fr(40)
    _cx(draw, label, lf, OUT_H // 2 + 50, color=_WHITE)

    # Context
    if context and anim > 0.5:
        alpha = min(1.0, (anim - 0.5) / 0.3)
        cf    = _fr(30)
        c_alpha = int(alpha * 200)
        _cx(draw, context, cf, OUT_H // 2 + 110,
            color=(int(_DIM[0]), int(_DIM[1]), int(_DIM[2])))

    # Bottom accent
    draw.rectangle([(120, OUT_H - 108), (OUT_W - 120, OUT_H - 102)],
                   fill=(*_ACCENT, 120))

    return img


# ─────────────────── comparison ───────────────────────────────────────────────

def _comparison_frame(data: dict, t: float) -> Image.Image:
    title = data.get("title", "")
    left  = data.get("left",  {"label": "A", "value": 50})
    right = data.get("right", {"label": "B", "value": 50})

    lv = float(left.get("value", 0))
    rv = float(right.get("value", 0))

    img  = Image.new("RGB", (OUT_W, OUT_H), _BG)
    draw = ImageDraw.Draw(img)
    _accent_bar(draw)
    _title(draw, title)

    anim     = _anim_t(t)
    max_bar  = OUT_H - 320
    bar_w    = 180
    bar_y0   = 140
    left_cx  = OUT_W // 4
    right_cx = 3 * OUT_W // 4
    max_val  = max(lv, rv, 1)

    winner = "left" if lv >= rv else "right"

    def _draw_bar(cx: int, val: float, lbl: str, side: str) -> None:
        bar_h    = int(max_bar * (val / max_val) * anim)
        x0       = cx - bar_w // 2
        x1       = cx + bar_w // 2
        bar_top  = bar_y0 + max_bar - bar_h

        # BG column
        draw.rectangle(
            [(x0, bar_y0), (x1, bar_y0 + max_bar)],
            fill=_PANEL, outline=(*_DIM, 40),
        )
        # Filled portion
        colour = _GREEN if (side == winner and anim > 0.9) else _ACCENT
        draw.rectangle([(x0, bar_top), (x1, bar_y0 + max_bar)], fill=colour)

        # Value at top of bar
        vf   = _f(36)
        unit = data.get("unit", "")
        vtxt = f"{val:g}{unit}"
        bbox = vf.getbbox(vtxt)
        draw.text((cx - (bbox[2] - bbox[0]) // 2, bar_top - 50), vtxt, font=vf, fill=_WHITE)

        # Label below
        lf   = _fr(34)
        bbox = lf.getbbox(lbl)
        draw.text(
            (cx - (bbox[2] - bbox[0]) // 2, bar_y0 + max_bar + 20),
            lbl, font=lf, fill=_WHITE,
        )

    _draw_bar(left_cx,  lv, left.get("label", "A"),  "left")
    _draw_bar(right_cx, rv, right.get("label", "B"), "right")

    # VS badge
    vf  = _f(44)
    _cx(draw, "VS", vf, OUT_H // 2 - 18, color=_DIM)

    return img


# ─────────────────── dispatcher ───────────────────────────────────────────────

_RENDERERS = {
    "bar_chart":  _bar_chart_frame,
    "timeline":   _timeline_frame,
    "stat_card":  _stat_card_frame,
    "comparison": _comparison_frame,
}


def _render_frames(data: dict, total_seconds: float) -> list[np.ndarray]:
    chart_type = data.get("type", "")
    renderer   = _RENDERERS.get(chart_type)
    if renderer is None:
        raise ValueError(f"Unknown infographic type: {chart_type!r}. "
                         f"Choose from {list(_RENDERERS)}")

    n      = max(1, int(total_seconds * FPS))
    frames = []
    for i in range(n):
        t   = i / max(n - 1, 1)
        img = renderer(data, t)
        frames.append(np.array(img))
    return frames


def _frames_to_mp4(frames: list[np.ndarray], out_path: Path) -> Path:
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=FPS)
    clip.write_videofile(
        str(out_path),
        codec="libx264",
        audio=False,
        preset="medium",
        logger=None,
    )
    clip.close()
    return out_path


# ─────────────────── public API ───────────────────────────────────────────────

def generate_infographic_clip(
    data: dict[str, Any],
    out_path: Path,
    duration_sec: float = 8.0,
) -> Path:
    """Render an animated infographic MP4.  Returns *out_path* (cached if exists)."""
    if out_path.exists():
        logger.info(f"Reusing cached infographic: {out_path.name}")
        return out_path

    chart_type = data.get("type", "unknown")
    logger.info(f"Rendering infographic ({chart_type}) → {out_path.name} ({duration_sec:.0f}s)")

    frames = _render_frames(data, duration_sec)
    _frames_to_mp4(frames, out_path)
    logger.info(f"Infographic written: {out_path.name} ({len(frames)} frames)")
    return out_path
