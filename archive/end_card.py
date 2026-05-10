"""Render a branded channel end card as a 1280×720 PNG.

Visual design (matches existing title-card palette):
  Background   : #0D1117  (dark navy)
  Accent bars  : #38BDF8  (sky blue)
  Channel name : #FFFFFF  (white, DejaVu Serif Bold)
  Subtext      : #94A3B8  (slate, DejaVu Sans)
  Handle       : #38BDF8  (sky blue, DejaVu Sans Bold)

Usage:
    png  = render_end_card_png("AI Today", "@AIToday", out_path)
    clip = ffmpeg_utils.end_card_clip(png, clip_out, duration_sec=10.0)
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

W, H = 1280, 720

_BG    = (13,  17,  23)    # #0D1117
_WHITE = (255, 255, 255)
_BLUE  = (56,  189, 248)   # #38BDF8
_SLATE = (148, 163, 184)   # #94A3B8

_SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
_SANS       = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
_SANS_BOLD  = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def render_end_card_png(
    channel_name: str,
    handle: str,
    out_path: Path,
    *,
    cta: str = "Subscribe · Daily AI News",
) -> Path:
    """Render the channel end card and save to *out_path*.

    Skips rendering if the file already exists (cache-friendly).
    """
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img  = Image.new("RGB", (W, H), color=_BG)
    draw = ImageDraw.Draw(img)

    # ── Fonts ─────────────────────────────────────────────────────────────────
    name_font   = _load_font(_SERIF_BOLD, 72)
    cta_font    = _load_font(_SANS,       28)
    handle_font = _load_font(_SANS_BOLD,  34)

    # ── Measure all text blocks ───────────────────────────────────────────────
    name_bbox   = name_font.getbbox(channel_name)
    cta_bbox    = cta_font.getbbox(cta)
    handle_bbox = handle_font.getbbox(handle)

    name_w   = name_bbox[2] - name_bbox[0]
    name_h   = name_bbox[3] - name_bbox[1]
    cta_h    = cta_bbox[3]  - cta_bbox[1]
    handle_h = handle_bbox[3] - handle_bbox[1]

    bar_h        = 3
    bar_w        = 640
    gap_bar_name = 24   # space between bar and channel name
    gap_name_bar = 24
    gap_bar_cta  = 28
    gap_cta_hdl  = 18

    # Total block height for vertical centering
    block_h = (
        bar_h + gap_bar_name +
        name_h + gap_name_bar +
        bar_h + gap_bar_cta +
        cta_h + gap_cta_hdl +
        handle_h
    )
    y = (H - block_h) // 2

    bar_x = (W - bar_w) // 2

    # ── Top accent bar ────────────────────────────────────────────────────────
    draw.rectangle([bar_x, y, bar_x + bar_w, y + bar_h], fill=_BLUE)
    y += bar_h + gap_bar_name

    # ── Channel name ──────────────────────────────────────────────────────────
    name_x = (W - name_w) // 2
    draw.text((name_x, y), channel_name, font=name_font, fill=_WHITE)
    y += name_h + gap_name_bar

    # ── Bottom accent bar ─────────────────────────────────────────────────────
    draw.rectangle([bar_x, y, bar_x + bar_w, y + bar_h], fill=_BLUE)
    y += bar_h + gap_bar_cta

    # ── CTA text ──────────────────────────────────────────────────────────────
    cta_w = cta_bbox[2] - cta_bbox[0]
    cta_x = (W - cta_w) // 2
    draw.text((cta_x, y), cta, font=cta_font, fill=_SLATE)
    y += cta_h + gap_cta_hdl

    # ── Channel handle ────────────────────────────────────────────────────────
    handle_w = handle_bbox[2] - handle_bbox[0]
    handle_x = (W - handle_w) // 2
    draw.text((handle_x, y), handle, font=handle_font, fill=_BLUE)

    img.save(str(out_path), "PNG")
    logger.debug(f"End card rendered: {out_path.name}")
    return out_path
