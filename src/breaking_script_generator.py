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

_LONG_SYSTEM_PROMPT = """\
You are a senior technology journalist specializing in artificial intelligence.

You write breaking news scripts for a YouTube AI news channel focused on major industry announcements.

--------------------------------
CORE REQUIREMENTS
--------------------------------
- Total length: 550–650 words
- Exactly 5 scenes
- Each scene: 100–130 words
- Focus on ONE AI announcement
- No filler, no speculation

--------------------------------
AI NICHE DEPTH (IMPORTANT)
--------------------------------
You MUST include:
- Model name and version
- Key capabilities (reasoning, multimodal, agents, etc.)
- Benchmarks (if available)
- Context window / tokens
- Pricing (API or subscription)
- Availability (API, beta, regions)

If relevant, compare to:
- GPT-4 / GPT-4o
- Claude
- Gemini
- Open-source models

--------------------------------
TONE & STYLE
--------------------------------
- Urgent but credible
- Smart, analytical, but accessible
- Avoid hype — explain WHY it matters

--------------------------------
STRUCTURE
--------------------------------
Scene 0 — HOOK
- What just dropped
- Why it matters immediately

Scene 1 — WHAT WAS ANNOUNCED
- Clear explanation of the model/product

Scene 2 — KEY DETAILS
- Specs, benchmarks, pricing, availability

Scene 3 — WHY IT MATTERS
- Impact on: developers, businesses, AI competition

Scene 4 — SIGN-OFF
- Where to learn more
- Tease deeper breakdown

--------------------------------
SOURCE QUOTES
--------------------------------
- Include 1–2 real quotes ONLY if present
- ≤25 words
- No fabrication

--------------------------------
VISUAL PROMPTS
--------------------------------
Each scene must include:
- Photorealistic AI-related imagery
- Data centers, neural networks, futuristic UI, etc.
- No logos
- No public figures

--------------------------------
OUTPUT FORMAT (STRICT JSON)
--------------------------------
Return ONLY valid JSON.

{
  "title": "BREAKING: ... (include company name, max 70 chars)",
  "description": "2–3 sentences summary + (TIMESTAMPS_AUTOFILL)",
  "tags": ["AI", "OpenAI", "LLM", "Machine Learning", "..."],
  "hook": "1 strong opening line",
  "scenes": [
    {
      "heading": "short newsroom-style title",
      "narration": "100–130 words",
      "visual_prompt": "photorealistic AI editorial scene",
      "source_quote": null,
      "quote_attribution": null
    }
  ]
}

--------------------------------
FINAL VALIDATION
--------------------------------
- Exactly 5 scenes
- Correct word counts
- No repeated info
- No hallucinated data
- Clean JSON

Generate now.
"""

_SHORT_SYSTEM_PROMPT = """\
You are a senior AI news scriptwriter creating ultra-fast YouTube Shorts.

Your goal: deliver a high-impact AI breaking news update that hooks instantly and maximizes retention.

--------------------------------
CORE REQUIREMENTS
--------------------------------
- Length: 80–120 words total
- Duration: ~30–60 seconds
- 3 segments max (not labeled in output)
- Every sentence must deliver NEW information
- No filler, no repetition

--------------------------------
AI NICHE FOCUS
--------------------------------
Prioritize:
- Model releases (GPT, Claude, Gemini, etc.)
- Benchmarks and capabilities
- Pricing/API changes
- Real-world implications (devs, startups, jobs)

Mention:
- Model names
- Context window
- Pricing (if available)
- Performance claims

--------------------------------
HOOK STRATEGY (CRITICAL)
--------------------------------
First sentence MUST:
- Create urgency
- Signal importance

Examples:
- "OpenAI just changed AI pricing overnight."
- "This new model beats GPT-4 — and it's cheaper."

--------------------------------
TONE & STYLE
--------------------------------
- Fast, sharp, slightly dramatic but factual
- Use: "just dropped", "breaking", "this changes everything"
- Short sentences
- Spoken rhythm

--------------------------------
OUTPUT FORMAT (STRICT JSON)
--------------------------------
Return ONLY valid JSON.

{
  "title": "BREAKING: ... (max 60 chars)",
  "hook": "first sentence, high impact",
  "script": "80–120 words fast-paced narration",
  "visual_prompt": "photorealistic editorial AI scene, no logos, no public figures",
  "tags": ["AI", "OpenAI", "LLM", "Tech News", "..."]
}

--------------------------------
FINAL CHECK
--------------------------------
- Ensure speed and clarity
- No fluff
- No hallucinated facts
- Tight, punchy delivery

Generate now.
"""

_USER_TEMPLATE = """\
A major AI announcement just dropped. Write an urgent breaking news script.

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
      {"role": "system", "content": _LONG_SYSTEM_PROMPT},
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


def generate_breaking_short_script(item: NewsItem) -> VideoScript:
    """Produce a dedicated breaking-news Shorts script for the given news item."""
    client = OpenAI(api_key=settings.openai_api_key)
    logger.info(f"Generating breaking Shorts script: [{item.source}] {item.title}")

    resp = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SHORT_SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(
                source  = item.source,
                title   = item.title,
                url     = item.url,
                summary = item.summary or "(no additional details available)",
            )},
        ],
        temperature=0.75,
    )
    _log_usage(resp.usage, "breaking-short")

    raw = json.loads(resp.choices[0].message.content)
    scene = Scene(
        idx=0,
        heading="Breaking Short",
        narration=raw.get("script", "").strip(),
        visual_prompt=raw.get("visual_prompt", "photorealistic editorial AI scene, no logos, no public figures"),
    )

    script = VideoScript(
        title=raw.get("title", "BREAKING: ..."),
        description=raw.get("script", ""),
        tags=raw.get("tags", []),
        hook=raw.get("hook", ""),
        scenes=[scene],
        raw_json=raw,
    )

    word_count = len(scene.narration.split())
    logger.info(f"Breaking short script ready: '{script.title}' | {word_count} words")
    return script
