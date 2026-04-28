"""Generate a short "Breaking News" script for a single AI announcement.

The breaking script is tight and urgent: 5 scenes, ~550-600 words total
(~5 minutes at 150 wpm). Structure:
  0. Hook — what just happened, why it matters RIGHT NOW
  1. What is it — the announcement in plain language
  2. Key details — specs, pricing, availability
  3. Why it matters — impact on users/devs/the industry
  4. Sign-off — where to learn more + teaser for full coverage
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.scraper import NewsItem
from src.script_generator import Scene, VideoScript

_SYSTEM_PROMPT = """\
You write urgent, factual "Breaking News" scripts for a YouTube AI news channel.
Breaking news videos are short (5 minutes), fast-paced, and laser-focused on ONE
announcement. No filler. No padding. Get to the facts immediately.

Rules:
- Exactly 5 scenes, each ~100-120 words of narration (~550-600 words total)
- Tone: urgent but accurate — use "just announced", "breaking", "this just dropped"
- Be specific: quote numbers, dates, model names, pricing — whatever is in the source
- visual_prompt: vivid DALL-E 3 prompt (no real logos, no named public figures)

Structure the 5 scenes as:
  0. Hook — what just happened, why it matters RIGHT NOW
  1. What is it — the announcement in plain language
  2. Key details — specs, pricing, availability
  3. Why it matters — impact on users, devs, and the industry
  4. Sign-off — where to learn more + teaser for full coverage

=== SOURCE QUOTES ===
For 1-2 scenes, pull a real quote directly from the announcement (press release, blog
post, CEO tweet, or official statement).
- source_quote: ≤25 words, verbatim. Never fabricate — only use text in the source.
  Leave null if no direct quote exists.
- quote_attribution: "Name, Role · Organization · domain.com"
  Leave null when source_quote is null.

=== OUTPUT FORMAT ===
Return ONLY valid JSON, no markdown fences, no commentary.
{
  "title": "BREAKING: ... (max 70 chars, include company name)",
  "description": "2-3 sentences + (TIMESTAMPS_AUTOFILL)",
  "tags": ["8-12 tags"],
  "hook": "1 punchy sentence for the first 5 seconds",
  "scenes": [
    {
      "heading": "short chyron-style title",
      "narration": "~100-120 words of spoken text",
      "visual_prompt": "DALL-E 3 image prompt",
      "source_quote": null,
      "quote_attribution": null
    },
    ...
  ]
}"""

_USER_TEMPLATE = """\
A major AI announcement just dropped. Write an urgent 5-minute breaking news script.

SOURCE:   {source}
HEADLINE: {title}
URL:      {url}
DETAILS:  {summary}"""


def _log_usage(usage, label: str = "") -> None:
    if not usage:
        return
    details = usage.prompt_tokens_details
    cached = getattr(details, "cached_tokens", 0) if details else 0
    pct = int(cached / usage.prompt_tokens * 100) if usage.prompt_tokens else 0
    tag = f"[{label}] " if label else ""
    logger.debug(
        f"{tag}OpenAI tokens — prompt: {usage.prompt_tokens} "
        f"({cached} cached, {pct}% hit) | completion: {usage.completion_tokens}"
    )


def generate_breaking_script(item: NewsItem) -> VideoScript:
    """Call GPT to produce an urgent breaking-news VideoScript for *item*."""
    client = OpenAI(api_key=settings.openai_api_key)
    logger.info(f"Generating breaking script: [{item.source}] {item.title}")

    resp = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(
                source  = item.source,
                title   = item.title,
                url     = item.url,
                summary = item.summary or "(no additional details available)",
            )},
        ],
        temperature=0.65,
    )
    _log_usage(resp.usage, "breaking")

    raw = json.loads(resp.choices[0].message.content)
    scenes = [
        Scene(
            idx=i,
            heading=s.get("heading") or "Breaking News",
            narration=s.get("narration") or "",
            visual_prompt=s.get("visual_prompt") or s.get("heading", "Breaking news event"),
            source_quote=(s.get("source_quote") or "").strip() or None,
            quote_attribution=(s.get("quote_attribution") or "").strip() or None,
            infographic_data=s.get("infographic_data"),
            video_query=s.get("video_query"),
        )
        for i, s in enumerate(raw.get("scenes", []))
    ]

    script = VideoScript(
        title=raw["title"],
        description=raw["description"],
        tags=raw["tags"],
        hook=raw["hook"],
        scenes=scenes,
    )

    word_count = sum(len(s.narration.split()) for s in scenes)
    logger.info(f"Breaking script ready: '{script.title}' | {len(scenes)} scenes | {word_count} words")
    return script
