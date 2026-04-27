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
    short_narration: str | None = None  # ~120 words written for a standalone Shorts cut
    infographic_data: dict | None = None  # animated chart/stat; skips DALL-E when set


@dataclass
class VideoScript:
    title: str
    description: str
    tags: list[str]
    hook: str                        # selected hook (best variant chosen at runtime)
    hook_variants: list[str] = field(default_factory=list)  # all GPT-generated options
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
                    "hook_variants": self.hook_variants,
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


_INFOGRAPHIC_GUIDE = """
For each scene you may optionally set "infographic_data" to replace the
static DALL-E image with an animated chart. Use it for scenes that quote
specific numbers, compare models, or show a timeline of events. Leave it
null for narrative/opinion scenes.  Supported types:

  bar_chart   {"type":"bar_chart","title":"...","unit":"%",
               "items":[{"label":"GPT-4o","value":87.2}, ...]}
               Up to 8 items. Values are plain numbers (no units in value).

  timeline    {"type":"timeline","title":"...",
               "events":[{"date":"Jan 2024","label":"Sora"}, ...]}
               Up to 6 events in chronological order.

  stat_card   {"type":"stat_card","value":"70B","label":"Parameters",
               "context":"10× more than GPT-3"}
               value may include prefix/suffix: "$4.2B", "87%", "#1".

  comparison  {"type":"comparison","title":"HumanEval Score","unit":"%",
               "left":{"label":"GPT-3.5","value":48.1},
               "right":{"label":"GPT-4o","value":90.2}}
"""

USER_PROMPT_TMPL = """Using the AI news items below, write a {target_words}-word script
(roughly 15 minutes at 150 wpm) broken into {num_scenes} scenes.

For each scene provide:
- heading: short chyron-style title (max 8 words)
- narration: the actual spoken text (natural, conversational, contractions OK)
- visual_prompt: a DALL-E 3 image prompt (always required as fallback)
- infographic_data: animated chart data when the scene has concrete numbers
  or comparisons — otherwise null. See the infographic guide below.
{infographic_guide}
Close with a sign-off that invites likes/subscribes without being cringey.

Also produce:
- title: catchy YouTube title, under 70 chars, include one emoji max
- description: 200-400 words, summary + chapter timestamps placeholder
  '(TIMESTAMPS_AUTOFILL)', plus 'Sources:' section with URLs
- tags: 15-25 lowercase YouTube tags, comma-separated concepts
- hook_variants: exactly 3 distinct 1-2 sentence hooks (each 5-10s when spoken).
  Each must tease the 2-3 biggest stories differently:
  variant 0 — lead with the most surprising fact or number
  variant 1 — open with a provocative question
  variant 2 — bold statement of the biggest implication

News items (ranked by relevance):
{items_block}

Return this JSON schema exactly:
{{
  "title": "...",
  "description": "...",
  "tags": ["..."],
  "hook_variants": ["...", "...", "..."],
  "scenes": [
    {{
      "heading": "...",
      "narration": "...",
      "visual_prompt": "...",
      "infographic_data": null
    }},
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


def generate_script(
    items: list[NewsItem],
    num_scenes: int = 8,
    data_dir: Path | None = None,
) -> VideoScript:
    """Call GPT to produce a structured VideoScript.

    *data_dir* is passed to hook_selector so the best hook variant is chosen
    automatically. When omitted the first variant is used (useful in tests).
    """
    if not items:
        raise ValueError("No news items provided")

    client = OpenAI(api_key=settings.openai_api_key)
    user_prompt = USER_PROMPT_TMPL.format(
        target_words=settings.script_target_words,
        num_scenes=num_scenes,
        items_block=_build_items_block(items),
        infographic_guide=_INFOGRAPHIC_GUIDE,
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

    scenes = []
    for i, s in enumerate(data.get("scenes", [])):
        infographic = s.get("infographic_data") or None
        if infographic and not isinstance(infographic, dict):
            infographic = None
        scenes.append(Scene(
            idx=i,
            heading=s["heading"],
            narration=s["narration"],
            visual_prompt=s["visual_prompt"],
            infographic_data=infographic,
        ))
    infographic_count = sum(1 for sc in scenes if sc.infographic_data)

    # Dynamic hook selection
    hook_variants: list[str] = data.get("hook_variants") or []
    if not hook_variants:
        # Fallback: GPT returned old-style single hook
        hook_variants = [data.get("hook", "")]
    hook_variants = [h for h in hook_variants if h]

    if data_dir and len(hook_variants) > 1:
        from src.hook_selector import pick_hook
        chosen_hook = pick_hook(hook_variants, data_dir, context_key="daily")
    else:
        chosen_hook = hook_variants[0] if hook_variants else ""

    script = VideoScript(
        title=data["title"],
        description=data["description"],
        tags=data.get("tags", []),
        hook=chosen_hook,
        hook_variants=hook_variants,
        scenes=scenes,
        raw_json=data,
    )

    word_count = sum(len(s.narration.split()) for s in scenes)
    logger.info(
        f"Generated script: {word_count} words across {len(scenes)} scenes | "
        f"hook variant {hook_variants.index(chosen_hook) + 1}/{len(hook_variants)} | "
        f"{infographic_count} infographic scene(s)"
    )
    return script


if __name__ == "__main__":
    from src.scraper import scrape_all

    news = scrape_all(top_n=8)
    script = generate_script(news)
    script.save(settings.output_dir / "script.json")
    print(f"Saved → {settings.output_dir / 'script.json'}")
    print(f"Title: {script.title}")
