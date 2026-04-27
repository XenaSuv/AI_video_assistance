"""Generate a 15-minute video script + YouTube metadata from scraped news items.

Structured output: scenes/segments so the video generator can make matching b-roll.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.scraper import NewsItem


@dataclass
class Scene:
    """A single narrated segment. Each scene gets one b-roll clip + TTS chunk."""
    idx: int
    heading: str
    narration: str              # what the voice says
    visual_prompt: str          # DALL-E 3 prompt; used as fallback when no screenshot
    duration_sec: int = 0       # filled after TTS timing is known
    screenshot_key: str | None = None   # weekly tutorials: real screenshot from curated library


@dataclass
class VideoScript:
    title: str
    description: str
    tags: list[str]
    hook: str                   # 5-10s opening line used for the Shorts cut
    scenes: list[Scene] = field(default_factory=list)
    raw_json: dict[str, Any] = field(default_factory=dict)

    @property
    def full_narration(self) -> str:
        return "\n\n".join(s.narration for s in self.scenes)

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "title": self.title,
                    "description": self.description,
                    "tags": self.tags,
                    "hook": self.hook,
                    "scenes": [s.__dict__ for s in self.scenes],
                },
                indent=2,
            )
        )


# --------------------- Prompt ---------------------

SYSTEM_PROMPT = """You are the writer and producer for a daily AI news YouTube channel.
You write punchy, accurate, engaging scripts that sound like a knowledgeable friend
explaining things — NOT like a press release. You cite specific papers/companies.
Avoid hype; be honest about limitations. No 'welcome back to the channel' filler.

Output MUST be a single valid JSON object, no commentary, no markdown fences."""


USER_PROMPT_TMPL = """Using the AI news items below, write a {target_words}-word script
(roughly 15 minutes at 150 wpm) broken into {num_scenes} scenes.

For each scene provide:
- heading: short chyron-style title (max 8 words)
- narration: the actual spoken text (natural, conversational, contractions OK)
- visual_prompt: a concrete text-to-video prompt for RunwayML Gen-3. Describe
  cinematography, subject, setting, lighting, camera motion. Avoid real
  company logos or named public figures (generates safer / works better).

Open with a 5-10s hook that teases the 2-3 biggest stories.
Close with a sign-off that invites likes/subscribes without being cringey.

Also produce:
- title: catchy YouTube title, under 70 chars, include one emoji max
- description: 200-400 words, summary + chapter timestamps placeholder
  '(TIMESTAMPS_AUTOFILL)', plus 'Sources:' section with URLs
- tags: 15-25 lowercase YouTube tags, comma-separated concepts

News items (ranked by relevance):
{items_block}

Return this JSON schema exactly:
{{
  "title": "...",
  "description": "...",
  "tags": ["..."],
  "hook": "...",
  "scenes": [
    {{"heading": "...", "narration": "...", "visual_prompt": "..."}},
    ...
  ]
}}"""


def _build_items_block(items: list[NewsItem]) -> str:
    lines = []
    for i, it in enumerate(items, 1):
        authors = ", ".join(it.authors[:3]) + (" et al." if len(it.authors) > 3 else "")
        lines.append(
            f"{i}. [{it.source}] {it.title}\n"
            f"   Authors: {authors}\n"
            f"   URL: {it.url}\n"
            f"   Summary: {it.summary[:500]}\n"
        )
    return "\n".join(lines)


def generate_script(items: list[NewsItem], num_scenes: int = 8) -> VideoScript:
    """Call GPT to produce a structured VideoScript."""
    if not items:
        raise ValueError("No news items provided")

    client = OpenAI(api_key=settings.openai_api_key)
    user_prompt = USER_PROMPT_TMPL.format(
        target_words=settings.script_target_words,
        num_scenes=num_scenes,
        items_block=_build_items_block(items),
    )

    logger.info(f"Requesting script from {settings.openai_model} "
                f"({settings.script_target_words} words, {num_scenes} scenes)")

    resp = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.7,
    )

    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)

    scenes = [
        Scene(
            idx=i,
            heading=s["heading"],
            narration=s["narration"],
            visual_prompt=s["visual_prompt"],
        )
        for i, s in enumerate(data.get("scenes", []))
    ]

    script = VideoScript(
        title=data["title"],
        description=data["description"],
        tags=data.get("tags", []),
        hook=data["hook"],
        scenes=scenes,
        raw_json=data,
    )

    word_count = sum(len(s.narration.split()) for s in scenes)
    logger.info(f"Generated script: {word_count} words across {len(scenes)} scenes")
    return script


if __name__ == "__main__":
    from src.scraper import scrape_all

    news = scrape_all(top_n=8)
    script = generate_script(news)
    script.save(settings.output_dir / "script.json")
    print(f"Saved → {settings.output_dir / 'script.json'}")
    print(f"Title: {script.title}")
