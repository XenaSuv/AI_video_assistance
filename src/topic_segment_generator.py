"""Generate deep-dive topic segment scripts for the AI YouTube channel.

Supported formats:
  honest_review   – "I tried X — honest take" (tool review)
  hidden_gems     – "5 AI tools you've never heard of"
  ai_project_build– "I built [project] using only AI"
  news_with_opinion – "Big AI news + what it actually means for you"

Topic ideas are persisted in data/topic_ideas.json and auto-refreshed via GPT
whenever the queue runs low.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from loguru import logger
from src.retry_utils import make_openai_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.cost_tracker import get_ledger
from src.script_generator import Scene, VideoScript

# ── Constants ─────────────────────────────────────────────────────────────────

TopicFormat = Literal[
    "honest_review",
    "hidden_gems",
    "ai_project_build",
    "news_with_opinion",
]

_IDEA_FILE = Path("data/topic_ideas.json")
_USED_FILE = Path("data/topic_ideas_used.json")
_REFILL_THRESHOLD = 3   # ask GPT for more ideas when queue drops below this


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class TopicIdea:
    format: TopicFormat
    subject: str            # main subject/tool/project
    angle: str              # unique angle or conflict for this episode
    suggested_title: str    # draft YouTube title
    created: str = field(default_factory=lambda: date.today().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TopicIdea":
        return cls(**d)


# ── System prompts per format ─────────────────────────────────────────────────

_FORMAT_PROMPTS: dict[str, str] = {
    "honest_review": """\
You are writing a YouTube script for an "honest tool review" episode.
Style: first-person, tried it yourself, candid about both wins and frustrations.
Structure: 1 hook → 2 what is it / who it's for → 3 real walkthrough of key features →
4 where it shines → 5 where it falls short → 6 verdict + who should pay for it.
Speak directly to the viewer: "If you're a freelancer you'll love X. If you need Y, skip it."
Include at least ONE unexpected finding that wasn't in the marketing copy.
Never be purely positive — real reviews have friction.""",

    "hidden_gems": """\
You are writing a YouTube script for a "hidden gems" list episode about underrated AI tools.
Style: enthusiastic curator who digs past the hype. Not ChatGPT, not Midjourney.
Structure: 1 hook (why the mainstream tools miss something) → one scene per tool (5 tools):
  - tool name + what it does in one sentence
  - the one thing it does better than anything else
  - real-world use case in 2–3 sentences
  - pricing / availability
→ final scene: quick recap + which to try first.
Each tool scene is self-contained so viewers who jump around still get value.""",

    "ai_project_build": """\
You are writing a YouTube script for a "I built X using only AI" episode.
Style: documentary-style personal journey. Show the messy reality, not a polished ad.
Structure: 1 hook (show the finished result first) → 2 the challenge / zero prior skills →
3 picking the right AI tools → 4 what worked on first try →
5 what broke and how you fixed it (with AI) → 6 final result + honest time/cost breakdown.
Be specific: actual tool names, prompts used, where it got stuck.
The viewer should feel they could replicate this tomorrow.""",

    "news_with_opinion": """\
You are writing a YouTube script for a "AI news + my take" episode.
Style: informed commentator with a clear point of view. Not a press release reader.
Structure: 1 hook (the headline + why it's bigger than it looks) →
2 what actually happened (facts, numbers, quotes) → 3 what the company/researchers claim →
4 your honest opinion: what's real, what's spin, what's missing →
5 concrete implication for the average person watching →
6 close with an open question that makes viewers think.
Every scene must include at least one opinionated statement. Avoid neutral tone.""",
}

# Shared constraints appended to every format prompt
_SHARED_RULES = """
=== SHARED SCRIPT RULES ===
- Conversational contractions everywhere ("it's", "you'll", "here's")
- Vary sentence length: punchy short ones for impact, longer for explanation
- No "welcome back to the channel" filler
- No generic AI hype phrases ("revolutionary", "game-changing", "the future of")
- Include concrete specifics: real tool names, prices, benchmark numbers where relevant
- Fill "short_narration" for 2-3 middle scenes (~120 words, self-contained,
  end with "Subscribe for daily AI news.")
- Fill "video_query" for 2-3 scenes with physical/real-world stock footage potential
- Fill "infographic_data" when a scene has concrete numbers to compare (bar_chart,
  stat_card, comparison, timeline — same JSON schema as the daily news script)

=== OUTPUT FORMAT ===
Return a single valid JSON object, no markdown fences:
{
  "title": "...",
  "description": "200-400 word YouTube description. End with '(TIMESTAMPS_AUTOFILL)'",
  "tags": ["tag1", ...],
  "hook_variants": ["fact hook", "question hook", "bold statement hook"],
  "scenes": [
    {
      "heading": "max 8 words",
      "narration": "spoken text",
      "visual_prompt": "DALL-E 3 photorealistic description",
      "video_query": null,
      "infographic_data": null,
      "short_narration": null,
      "source_quote": null,
      "quote_attribution": null
    }
  ]
}"""


# ── Idea queue management ─────────────────────────────────────────────────────

def _load_ideas() -> list[TopicIdea]:
    if not _IDEA_FILE.exists():
        return []
    return [TopicIdea.from_dict(d) for d in json.loads(_IDEA_FILE.read_text())]


def _save_ideas(ideas: list[TopicIdea]) -> None:
    _IDEA_FILE.parent.mkdir(parents=True, exist_ok=True)
    _IDEA_FILE.write_text(json.dumps([i.to_dict() for i in ideas], indent=2))


def _load_used() -> list[str]:
    if not _USED_FILE.exists():
        return []
    return json.loads(_USED_FILE.read_text())


def _mark_used(idea: TopicIdea) -> None:
    used = _load_used()
    used.append(idea.suggested_title)
    _USED_FILE.parent.mkdir(parents=True, exist_ok=True)
    _USED_FILE.write_text(json.dumps(used, indent=2))


def _log_usage(usage, label: str = "") -> None:
    if not usage:
        return
    details = usage.prompt_tokens_details
    cached = getattr(details, "cached_tokens", 0) if details else 0
    tag = f"[{label}] " if label else ""
    logger.debug(
        f"{tag}tokens — prompt: {usage.prompt_tokens} "
        f"({cached} cached) | completion: {usage.completion_tokens}"
    )
    get_ledger().record_llm(
        tag=label or "topic",
        model=settings.openai_model,
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_tokens=cached,
    )


# ── GPT: generate new topic ideas ─────────────────────────────────────────────

_IDEA_GEN_PROMPT = """\
You produce YouTube video ideas for an AI-niche channel (tech-savvy audience, 18-35).
The channel covers AI tools, practical use cases, honest reviews, and news analysis.

Generate {count} fresh video topic ideas using these formats:
  honest_review   – "I tried X — honest take" (one specific AI tool)
  hidden_gems     – "N AI tools you've never heard of" (underrated, not ChatGPT/Midjourney)
  ai_project_build– "I built [real project] using only AI" (site, design, business plan…)
  news_with_opinion – "Big AI news + what it means for YOU" (one recent trend/announcement)

Rules:
- Mix all four formats
- Be specific in subject/angle — avoid generic phrases
- Avoid topics already used: {used_titles}
- today is {today}

Return JSON:
{{
  "ideas": [
    {{
      "format": "honest_review | hidden_gems | ai_project_build | news_with_opinion",
      "subject": "main topic / tool / project name",
      "angle": "unique conflict or angle that makes this episode interesting",
      "suggested_title": "YouTube title under 70 chars, max 1 emoji"
    }}
  ]
}}"""


def generate_next_topic_ideas(count: int = 8) -> list[TopicIdea]:
    """Ask GPT to generate *count* new topic ideas and append them to the queue."""
    client = make_openai_client()
    used = _load_used()[-20:]  # only recent history to keep prompt short
    prompt = _IDEA_GEN_PROMPT.format(
        count=count,
        used_titles=", ".join(f'"{t}"' for t in used) if used else "none yet",
        today=date.today().isoformat(),
    )
    logger.info(f"Generating {count} new topic ideas via GPT…")
    resp = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    _log_usage(resp.usage, "topic-ideas")
    data = json.loads(resp.choices[0].message.content or "{}")
    new_ideas = [TopicIdea.from_dict(d) for d in data.get("ideas", [])]
    existing = _load_ideas()
    existing.extend(new_ideas)
    _save_ideas(existing)
    logger.info(f"Added {len(new_ideas)} ideas → queue now has {len(existing)}")
    return new_ideas


def pick_next_topic(format_filter: TopicFormat | None = None) -> TopicIdea:
    """Pop the next idea from the queue (auto-refills when low)."""
    ideas = _load_ideas()
    if len(ideas) <= _REFILL_THRESHOLD:
        logger.info("Topic queue running low — auto-refilling…")
        generate_next_topic_ideas(count=8)
        ideas = _load_ideas()

    if format_filter:
        candidates = [i for i in ideas if i.format == format_filter]
        if not candidates:
            logger.warning(f"No ideas with format={format_filter!r}; ignoring filter")
            candidates = ideas
    else:
        candidates = ideas

    chosen = candidates[0]
    remaining = [i for i in ideas if i is not chosen]
    _save_ideas(remaining)
    _mark_used(chosen)
    logger.info(f"Picked topic: [{chosen.format}] {chosen.suggested_title}")
    return chosen


# ── GPT: generate script from topic idea ─────────────────────────────────────

def generate_topic_script(
    idea: TopicIdea,
    num_scenes: int | None = None,
) -> VideoScript:
    """Generate a full VideoScript for the given TopicIdea."""
    client = make_openai_client()

    default_scenes = {
        "honest_review": 6,
        "hidden_gems": 7,
        "ai_project_build": 7,
        "news_with_opinion": 6,
    }
    scenes_count = num_scenes or default_scenes.get(idea.format, 7)
    target_words = scenes_count * 250   # ~250 words per scene

    system_prompt = _FORMAT_PROMPTS[idea.format] + _SHARED_RULES
    user_prompt = (
        f"Write a {target_words}-word YouTube script broken into {scenes_count} scenes.\n\n"
        f"Topic: {idea.subject}\n"
        f"Angle / conflict: {idea.angle}\n"
        f"Draft title: {idea.suggested_title}\n"
    )

    logger.info(
        f"Generating topic script [{idea.format}] '{idea.subject}' "
        f"({scenes_count} scenes, ~{target_words} words)"
    )
    resp = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.75,
    )
    _log_usage(resp.usage, f"topic-script-{idea.format}")

    data = json.loads(resp.choices[0].message.content or "{}")
    scenes = []
    for i, s in enumerate(data.get("scenes", [])):
        infographic = s.get("infographic_data") or None
        if infographic and not isinstance(infographic, dict):
            infographic = None
        scenes.append(Scene(
            idx=i,
            heading=s.get("heading", f"Scene {i+1}"),
            narration=s.get("narration", ""),
            visual_prompt=s.get("visual_prompt", ""),
            infographic_data=infographic,
            video_query=(s.get("video_query") or "").strip() or None,
            short_narration=(s.get("short_narration") or "").strip() or None,
            source_quote=(s.get("source_quote") or "").strip() or None,
            quote_attribution=(s.get("quote_attribution") or "").strip() or None,
        ))

    hook_variants: list[str] = data.get("hook_variants") or []
    chosen_hook = hook_variants[0] if hook_variants else ""

    word_count = sum(len(sc.narration.split()) for sc in scenes)
    shorts_count = sum(1 for sc in scenes if sc.short_narration)
    logger.info(
        f"Topic script ready: {word_count} words, {len(scenes)} scenes, "
        f"{shorts_count} shorts narrations"
    )

    return VideoScript(
        title=data.get("title", idea.suggested_title),
        description=data.get("description", ""),
        tags=data.get("tags", []),
        hook=chosen_hook,
        hook_variants=hook_variants,
        scenes=scenes,
        raw_json=data,
    )
