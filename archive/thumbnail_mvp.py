from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageFilter
from loguru import logger

COLORS = ["yellow", "white", "red", "cyan", "lime"]
POSITIONS = [(60, 120), (60, 180), (60, 260), (60, 320)]
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SIZE = 120


def _resolve_out_dir(out_dir: Path | None = None) -> Path:
    base = Path(out_dir or Path.cwd() / "output" / "thumbnails_mvp")
    base.mkdir(parents=True, exist_ok=True)
    return base


def load_background(path: str | Path) -> Image.Image:
    img = Image.open(str(path)).convert("RGB")
    return img.resize((1280, 720), Image.LANCZOS)


def darken_background(img: Image.Image) -> Image.Image:
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 140))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def generate_thumbnail_text(title: str) -> str:
    words = [w for w in title.split() if w.strip()]
    if not words:
        return "AI NEWS"
    short_words = words[:4]
    if len(short_words) > 2:
        half = (len(short_words) + 1) // 2
        lines = [" ".join(short_words[:half]), " ".join(short_words[half:])]
        return "\n".join(line.upper() for line in lines)
    return " ".join(short_words).upper()


def draw_text(img: Image.Image, text: str) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    color = random.choice(COLORS)
    x, y = random.choice(POSITIONS)

    # Support multi-line text
    lines = text.split("\n")
    line_height = FONT_SIZE + 10
    for i, line in enumerate(lines):
        position = (x, y + i * line_height)
        draw.text(
            position,
            line,
            font=font,
            fill=color,
            stroke_width=6,
            stroke_fill="black",
        )
    return img


def add_highlight(img: Image.Image) -> Image.Image:
    draw = ImageDraw.Draw(img)
    x0 = random.randint(780, 860)
    y0 = random.randint(140, 260)
    x1 = x0 + random.randint(260, 360)
    y1 = y0 + random.randint(160, 240)
    border_color = random.choice(["red", "orange", "white"])

    draw.rectangle([(x0, y0), (x1, y1)], outline=border_color, width=10)
    arrow = [(x1 - 20, y0 + 30), (x1 + 50, y0 + 10), (x1 + 50, y0 + 50)]
    draw.polygon(arrow, fill=border_color)
    return img


def create_thumbnail(
    bg_path: str | Path,
    title: str,
    out_dir: str | Path | None = None,
) -> Path:
    out_dir = _resolve_out_dir(Path(out_dir) if out_dir else None)
    img = load_background(bg_path)
    img = img.filter(ImageFilter.GaussianBlur(2))
    img = darken_background(img)
    text = generate_thumbnail_text(title)
    img = draw_text(img, text)
    img = add_highlight(img)

    output_path = out_dir / f"thumbnail_mvp_{random.randint(0, 9999):04d}.jpg"
    img.convert("RGB").save(output_path, quality=95)
    logger.info(f"Generated thumbnail MVP: {output_path}")
    return output_path


def generate_thumbnails(
    news_item: dict | object,
    title: str,
    bg_images: Iterable[str | Path],
    out_dir: str | Path | None = None,
) -> list[Path]:
    thumbnails: list[Path] = []
    for bg_path in list(bg_images)[:3]:
        try:
            thumb = create_thumbnail(bg_path, title, out_dir=out_dir)
            thumbnails.append(thumb)
        except Exception as exc:
            logger.warning(f"thumbnail_mvp failed for {bg_path}: {exc}")
    return thumbnails


def pick_best_thumbnail(thumbnails: list[Path]) -> Path | None:
    if not thumbnails:
        return None
    return random.choice(thumbnails)
