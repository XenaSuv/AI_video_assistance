"""Render a broadcast-style quote card as a 1280×720 PNG.

Visual design:
  Background : #0A0E1A  (dark navy — matches existing title cards)
  Decorative " : #1B3A5C  (dark steel, top-left)
  Quote text  : #EFF6FF  (near-white, DejaVu Serif Bold)
  Separator   : #38BDF8  (sky blue accent — matches title card bars)
  Attribution : #7EA8C3  (slate blue, DejaVu Sans)

Usage:
    png  = render_quote_card_png(scene, out_path)
    clip = quote_card_clip(png, clip_out, duration_sec=4.0)   # via ffmpeg_utils
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.script_generator import Scene

W, H = 1280, 720

_BG      = (10,  14, 26)      # #0A0E1A
_QMARK   = (27,  58, 92)      # #1B3A5C  — large decorative quotation mark
_QUOTE   = (239, 246, 255)    # #EFF6FF
_SEP     = (56,  189, 248)    # #38BDF8
_ATTR    = (126, 168, 195)    # #7EA8C3

_SERIF_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
_SERIF_REG  = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
_SANS       = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

# Margins
_PAD_X = 100
_MAX_W  = W - 2 * _PAD_X      # 1080 px usable width


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_px: int) -> list[str]:
    """Word-wrap *text* so each line fits within *max_px* pixels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        bbox = font.getbbox(candidate)
        if bbox[2] > max_px and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _quote_font_size(quote: str) -> int:
    n = len(quote.split())
    if n <= 12:
        return 52
    if n <= 18:
        return 46
    return 40


def render_quote_card_png(scene: Scene, out_path: Path) -> Path:
    """Render a quote card for *scene* and save as PNG to *out_path*."""
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    quote   = (scene.source_quote or "").strip()
    attr    = (scene.quote_attribution or "").strip()
    if not quote:
        raise ValueError(f"Scene {scene.idx} has no source_quote to render")

    img  = Image.new("RGB", (W, H), color=_BG)
    draw = ImageDraw.Draw(img)

    # ── Decorative opening quotation mark ────────────────────────────────────
    qmark_font = _load_font(_SERIF_BOLD, 180)
    draw.text((70, -20), "“", font=qmark_font, fill=_QMARK)

    # ── Quote text ────────────────────────────────────────────────────────────
    q_font    = _load_font(_SERIF_BOLD, _quote_font_size(quote))
    line_h    = int(_quote_font_size(quote) * 1.45)
    lines     = _wrap(quote, q_font, _MAX_W)
    total_q_h = len(lines) * line_h

    # ── Attribution ───────────────────────────────────────────────────────────
    attr_font = _load_font(_SANS, 26)
    attr_h    = 26 + 16    # font size + padding
    sep_h     = 2 + 20     # separator line + spacing

    # Centre the whole block vertically (quote + separator + attribution)
    block_h = total_q_h + sep_h + attr_h
    y_start = (H - block_h) // 2 + 20   # slight upward bias (quote mark at top)

    # Draw lines
    for i, line in enumerate(lines):
        bbox = q_font.getbbox(line)
        x = (W - bbox[2]) // 2       # centre each line
        y = y_start + i * line_h
        draw.text((x, y), line, font=q_font, fill=_QUOTE)

    # ── Separator bar ─────────────────────────────────────────────────────────
    sep_y   = y_start + total_q_h + 20
    sep_w   = 520
    sep_x   = (W - sep_w) // 2
    draw.rectangle([sep_x, sep_y, sep_x + sep_w, sep_y + 2], fill=_SEP)

    # ── Attribution text ─────────────────────────────────────────────────────
    if attr:
        attr_y = sep_y + sep_h
        bbox   = attr_font.getbbox(attr)
        attr_x = (W - bbox[2]) // 2
        draw.text((attr_x, attr_y), attr, font=attr_font, fill=_ATTR)

    img.save(str(out_path), "PNG")
    logger.debug(f"Quote card rendered: {out_path.name}")
    return out_path
