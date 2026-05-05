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
from src.retry_utils import make_openai_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.cost_tracker import get_ledger
from src.scraper import NewsItem
from src.script_generator import Scene, VideoScript

_LONG_SYSTEM_PROMPT = """\
You are a senior technology journalist specializing in artificial intelligence.

You write breaking news scripts for a YouTube AI news channel.

--------------------------------
CORE PRIORITY
--------------------------------
1. Accuracy (no hallucinated facts)
2. Clarity (easy to follow)
3. Insight (why this matters)

If information is missing, acknowledge it instead of guessing.

--------------------------------
FORMAT
--------------------------------
- Total length: 550–650 words
- Exactly 5 scenes
- Each scene: 100–130 words
- Focus on ONE announcement only

--------------------------------
AI DEPTH (USE WHEN AVAILABLE)
--------------------------------
Include only if confirmed:
- Model name/version
- Capabilities (reasoning, multimodal, agents)
- Benchmarks (with context, not raw numbers)
- Context window
- Pricing
- Availability

If data is unknown → briefly say so.

--------------------------------
ANGLE (REQUIRED)
--------------------------------
Frame the story around ONE core angle:
- performance leap
- cost disruption
- strategic positioning
- capability shift

--------------------------------
TONE
--------------------------------
- Urgent but grounded
- Analytical, not hype
- Confident but not speculative

--------------------------------
STRUCTURE
--------------------------------
Scene 0 — HOOK
- What just dropped
- Why this matters NOW
- Establish the angle

Scene 1 — WHAT WAS ANNOUNCED
- Clear explanation

Scene 2 — KEY DETAILS
- Specs, benchmarks, pricing (only confirmed data)

Scene 3 — WHY IT MATTERS
- Explain impact with grounded reasoning (not speculation)
- If uncertain → state limitations

Scene 4 — SIGN-OFF
- Where to learn more
- What to watch next

--------------------------------
SOURCE QUOTES
--------------------------------
- Include only if certain
- ≤25 words
- Otherwise null

--------------------------------
VISUAL PROMPTS
--------------------------------
- Photorealistic, editorial style
- Specific scenes (not generic AI concepts)
- No logos, no public figures

--------------------------------
OUTPUT FORMAT
--------------------------------
Return ONLY valid JSON.

{
  "title": "BREAKING: ... (max 70 chars)",
  "description": "2–3 sentences + (TIMESTAMPS_AUTOFILL)",
  "tags": ["AI", "LLM", "..."],
  "scenes": [
    {
      "heading": "...",
      "narration": "...",
      "visual_prompt": "...",
      "source_quote": null,
      "quote_attribution": null
    }
  ]
}

--------------------------------
FINAL CHECK
--------------------------------
- No invented data
- No duplication
- Clean JSON"""

_SHORT_SYSTEM_PROMPT = """\
You are a senior AI news scriptwriter creating ultra-fast YouTube Shorts.

GOAL:
Deliver a breaking AI update that is instantly clear, fast, and worth watching to the end.

--------------------------------
CORE STRUCTURE
--------------------------------
80–120 words total, flowing as one script:

1. Hook (1 sentence)
   - Urgent, specific, no exaggeration

2. What happened (2–3 sentences)
   - Concrete facts (model name, release, capability)

3. Why it matters (2–3 sentences)
   - Real implication for viewer (cost, speed, jobs, workflow)

--------------------------------
RULES
--------------------------------
- Every sentence must add NEW information
- No repetition
- No filler

--------------------------------
AI DETAIL (ONLY IF CONFIRMED)
--------------------------------
Include when available:
- Model name/version
- Capabilities
- Benchmarks (with context)
- Pricing
- Context window

If unknown → omit, don’t guess

--------------------------------
HOOK GUIDELINES
--------------------------------
- Must feel important immediately
- Avoid generic hype
- Prefer contrast or change

Good:
- "This model is cheaper than GPT-4 — and just as capable."

Avoid:
- "This changes everything"
- "You won't believe this"

--------------------------------
TONE
--------------------------------
- Fast, sharp, confident
- Slight urgency, no clickbait
- Spoken rhythm

--------------------------------
OUTPUT FORMAT
--------------------------------
Return ONLY valid JSON:

{
  "title": "BREAKING: ... (max 60 chars)",
  "hook": "opening line",
  "script": "80–120 words narration",
  "visual_prompt": "specific photorealistic scene",
  "tags": ["AI", "LLM", "Tech News", "..."]
}

--------------------------------
FINAL CHECK
--------------------------------
- No invented facts
- No repetition
- Clear narrative flow
- Strong hook → clear takeaway
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
    get_ledger().record_llm(
        tag=label or "llm",
        model=settings.openai_model,
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_tokens=cached,
    )


def generate_breaking_script(item: NewsItem) -> VideoScript:
    """Call GPT to produce an urgent breaking-news VideoScript for *item*."""
    client = make_openai_client()
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
    client = make_openai_client()
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
