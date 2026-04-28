from __future__ import annotations

from pathlib import Path
from loguru import logger

from config import settings
from src.hooks_generator import generate_hooks
from src.script_generator import generate_short_script
from src.shorts_generator import create_short_video
from src.youtube_uploader import upload_short


def _get_field(news_item, field: str) -> str:
    if isinstance(news_item, dict):
        return str(news_item.get(field, "") or "")
    return str(getattr(news_item, field, "") or "")


def run_shorts_mvp(news_item) -> list[tuple[str, Path]]:
    logger.info("🚀 Running Shorts MVP")
    summary_text = _get_field(news_item, "summary")
    title_text = _get_field(news_item, "title")
    hooks = generate_hooks(summary_text, n=3)

    out_dir = settings.output_dir / "shorts_mvp"
    out_dir.mkdir(parents=True, exist_ok=True)

    videos: list[tuple[str, Path]] = []
    for i, hook in enumerate(hooks, start=1):
        logger.info(f"👉 Hook: {hook}")
        script = generate_short_script(news_item, hook)
        video_path = create_short_video(
            script,
            out_dir=out_dir,
            out_name=f"shorts_mvp_{i}.mp4",
        )
        videos.append((hook, video_path))

    for hook, video_path in videos:
        upload_short(video_path, title=hook, description=title_text)

    return videos
