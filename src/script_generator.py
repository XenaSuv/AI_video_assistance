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
    screenshot_key: str | None = None    # weekly tutorials: real screenshot from curated library
    short_narration: str | None = None   # ~120 words written for a standalone Shorts cut
    infographic_data: dict | None = None # animated chart/stat; skips DALL-E when set
    video_query: str | None = None       # stock B-roll search term; beats DALL-E, loses to infographic


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
- video_query: a 2-5 word stock-video search term for scenes with real-world
  action (people working, server rooms, office, phone, typing, presenting).
  Set for 2-3 scenes out of {num_scenes} — the ones with most physical activity.
  Use null for data scenes (use infographic instead) and abstract concepts.
  Keep generic — avoid brand names, logos, or overly specific interfaces.
  Examples: "developer coding laptop", "data center servers", "team meeting office",
  "person presenting screen", "smartphone app scrolling", "AI robot arm factory"
- infographic_data: animated chart data when the scene has concrete numbers
  or comparisons — otherwise null. See the infographic guide below.
- short_narration: YOU MUST fill this for EXACTLY 2-3 middle scenes
  (not scene 0 intro, not the last sign-off scene). Each must be a
  self-contained ~120-word Shorts script a viewer can watch without seeing
  the full video. Hook immediately ("Breaking:", "Just in:", "Here's what
  happened with..."). End with: "Subscribe for daily AI news."
  Set null only for the opening and closing scenes.
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
      "video_query": null,
      "infographic_data": null,
      "short_narration": null
    }},
    {{
      "heading": "...",
      "narration": "...",
      "visual_prompt": "...",
      "video_query": null,
      "infographic_data": null,
      "short_narration": "Breaking: [~120 words ending with 'Subscribe for daily AI news.']"
    }},
    ...
  ]
}}"""


_DAILY_SHORTS_FILL_PROMPT = """\
For each news scene below, write a standalone YouTube Shorts script (~120 words).

Rules:
- First sentence must be a hook: "Breaking:", "Just in:", "Here's what happened with..."
- Cover exactly ONE story from that scene — self-contained, no context needed
- Last sentence: "Subscribe for daily AI news."
- No filler, no "In today's video", direct and punchy

Return JSON: {{"scenes": [{{"idx": <N>, "short_narration": "..."}}]}}

Scenes to fill:
{scenes_block}
"""


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


def _fill_missing_short_narrations(
    scenes: list[Scene],
    client: OpenAI,
    min_count: int = 2,
) -> None:
    """Ensure at least *min_count* scenes have short_narration; fill gaps via GPT."""
    have = [s for s in scenes if s.short_narration]
    if len(have) >= min_count:
        return
    needed    = min_count - len(have)
    candidates = [s for s in scenes[1:-1] if not s.short_narration]
    if not candidates:
        candidates = [s for s in scenes if not s.short_narration]
    to_fill = candidates[:needed]
    if not to_fill:
        return

    logger.info(f"GPT returned {len(have)} short_narrations; auto-filling {len(to_fill)} more")
    scenes_block = "\n\n".join(
        f"idx={s.idx}, heading={s.heading!r}:\n{s.narration[:500]}"
        for s in to_fill
    )
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=[{
                "role": "user",
                "content": _DAILY_SHORTS_FILL_PROMPT.format(scenes_block=scenes_block),
            }],
            temperature=0.7,
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        idx_map = {s.idx: s for s in to_fill}
        for item in data.get("scenes", []):
            idx  = item.get("idx")
            text = (item.get("short_narration") or "").strip()
            if text and idx in idx_map:
                idx_map[idx].short_narration = text
                logger.info(f"  auto-filled short_narration for scene {idx}")
    except Exception as exc:
        logger.warning(f"short_narration auto-fill failed (non-fatal): {exc}")


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
        video_q   = (s.get("video_query") or "").strip() or None
        short_nar = (s.get("short_narration") or "").strip() or None
        scenes.append(Scene(
            idx=i,
            heading=s["heading"],
            narration=s["narration"],
            visual_prompt=s["visual_prompt"],
            infographic_data=infographic,
            video_query=video_q,
            short_narration=short_nar,
        ))
    infographic_count = sum(1 for sc in scenes if sc.infographic_data)
    broll_count       = sum(1 for sc in scenes if sc.video_query)

    _fill_missing_short_narrations(scenes, client, min_count=2)

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

    shorts_count = sum(1 for sc in scenes if sc.short_narration)
    word_count   = sum(len(s.narration.split()) for s in scenes)
    logger.info(
        f"Generated script: {word_count} words across {len(scenes)} scenes | "
        f"hook variant {hook_variants.index(chosen_hook) + 1}/{len(hook_variants)} | "
        f"{broll_count} B-roll | {infographic_count} infographic | {shorts_count} shorts"
    )
    return script


if __name__ == "__main__":
    from src.scraper import scrape_all

    news = scrape_all(top_n=8)
    script = generate_script(news)
    script.save(settings.output_dir / "script.json")
    print(f"Saved → {settings.output_dir / 'script.json'}")
    print(f"Title: {script.title}")
