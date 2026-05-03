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
You are writing a YouTube script for an honest tool review.

GOAL:
Make the viewer feel like you actually used the tool and formed a real opinion.

STYLE:
- First-person, conversational, direct
- Specific > general
- Show experience, not just conclusions

STRUCTURE:
1. Hook — strong opinion or surprising outcome (not generic)
2. What it is / who it's for (clear positioning, no fluff)
3. Real walkthrough — describe actual usage:
   - what you tried
   - what worked
   - what confused or slowed you down
4. Where it shines — concrete benefits (who exactly benefits and why)
5. Where it falls short — real friction, trade-offs, limitations
6. Verdict — clear recommendation:
   - who should pay
   - who should avoid it
   - what you'd personally do

REQUIREMENTS:
- Include at least ONE unexpected finding:
  (e.g. hidden limitation, surprising strength, misleading feature)
- Include at least 2 concrete micro-details:
  (time saved, clicks needed, pricing friction, UX issue, etc.)
- Include at least one moment of friction or mild frustration
- Avoid generic praise (“it’s great”, “very powerful”)

TONE:
- Honest, slightly opinionated, but fair
- No hype, no corporate tone

AVOID:
- Vague claims
- Purely positive framing
- “It depends” conclusions without a clear stance

OPTIONAL (STRONG):
- Brief comparison to a known alternative if relevant

OUTPUT:
Write the full script in natural spoken format (no bullet points).""",

    "hidden_gems": """\
You are writing a YouTube script for a "hidden gems" episode about underrated AI tools.

GOAL:
Make viewers feel like they just discovered tools they *should have known earlier*.

STYLE:
- Enthusiastic but credible
- Curious, slightly contrarian (“people are sleeping on this”)
- Specific, no generic hype

HOOK:
- Start with a clear angle:
  (e.g. mainstream tools are overpriced, limited, or overhyped)
- Create curiosity: “these tools solve things the popular ones don’t”

STRUCTURE:
1. Hook
2–6. One scene per tool (5 tools total)
7. Final recap

FOR EACH TOOL:
- Tool name + what it does (1 sentence, clear)
- Why it’s underrated (what people are missing)
- Its unfair advantage:
  → one specific thing it does noticeably better than alternatives
- Real-world use case (2–3 sentences, concrete scenario)
- One moment of surprise:
  → something unexpected, clever, or unusually powerful
- Pricing / availability (brief, practical)

IMPORTANT:
- Each tool must feel distinct (avoid overlap in use cases)
- No generic tools (avoid obvious picks)
- Avoid repeating the same benefit structure

FINAL SCENE:
- Rapid recap (1 line per tool max)
- Strong recommendation:
  → “Start with X if you want [specific outcome]”

TONE:
- “I can’t believe more people aren’t using this”
- Direct, energetic, but grounded

AVOID:
- Generic praise (“super useful”, “very powerful”)
- Tools everyone already knows
- Vague descriptions

OUTPUT:
Write the full script in natural spoken format.""",

    "ai_project_build": """\
You are writing a YouTube script for a "I built X using only AI" episode.

GOAL:
Make the viewer feel: “I could try this tomorrow — and avoid your mistakes.”

STYLE:
- First-person, documentary-style
- Honest, slightly messy, not overproduced
- Show the process, not just results

STRUCTURE:
1. Hook — show the final result immediately + quick tease of how hard it was
2. The challenge:
   - what you're building
   - constraints (time, budget, no prior skills)
3. Tool selection:
   - which AI tools you chose and why (brief, practical)
4. What worked immediately:
   - early wins that gave momentum
5. What broke:
   - specific failures, confusion, bad outputs
   - what you tried that didn’t work
   - how you fixed it using AI
6. Final result:
   - honest evaluation (not just success)
   - time spent, cost, effort
   - would you do it again?

REQUIREMENTS:
- Include at least 2 real prompts (short, but specific)
- Include at least 2 moments of friction or failure
- Include at least 1 “I didn’t expect this” moment
- Mention actual tools by name
- Show at least one iteration (bad → improved result)

REPLICABILITY:
- Avoid unnecessary steps
- Highlight shortcuts or “do this instead” insights
- Make the process feel achievable, not expert-only

TONE:
- Curious, honest, slightly self-critical
- No hype, no “AI magic” language

AVOID:
- Polished success story
- Vague descriptions of the process
- Skipping over failures

OUTPUT:
Write the full script in natural spoken format.""",

    "news_with_opinion": """\
You are writing a YouTube script for an "AI news + my take" episode.

GOAL:
Turn a news story into a clear, opinionated perspective that viewers will agree or disagree with.

CORE RULE:
This is not neutral reporting. You must take a stance.

STRUCTURE:
1. Hook:
   - State the headline
   - Immediately explain why this is bigger than it looks
   - Include a strong opinion or framing

2. What actually happened:
   - Concrete facts (numbers, timelines, specific claims)
   - Avoid vague language
   - Include at least one grounded interpretation (not just facts)

3. What they claim:
   - Summarize the company/research narrative
   - Then challenge or question it

4. Your take:
   - What’s real vs what’s hype/spin
   - What’s missing or not being said
   - Be specific and slightly provocative

5. Why it matters:
   - Concrete implication for the viewer
   - “This means that if you…, then…”

6. Close:
   - A sharp, discussion-driving question
   - Should invite disagreement or personal stance

REQUIREMENTS:
- Every scene must include at least one clear opinion
- At least one opinion should be debatable (not universally agreeable)
- Prefer specificity over generalization (numbers, examples > vague claims)

TONE:
- Confident, sharp, informed
- Slightly contrarian where appropriate
- No corporate neutrality

AVOID:
- Fence-sitting (“it depends” without a stance)
- Generic observations
- Repeating the same point across scenes

OUTPUT:
Write the full script in natural spoken format.""",
}

# Shared constraints appended to every format prompt
_SHARED_RULES = """
=== CORE PRIORITY ===
1. Clear, engaging narration
2. Logical scene structure
3. Useful visual + data enhancements (only when relevant)

If trade-offs occur, prioritize narration quality over extras.

=== STYLE RULES ===
- Use conversational contractions ("it's", "you'll", "here's")
- Vary sentence length (short for impact, longer for explanation)
- No filler intros
- No generic AI hype phrases
- Use concrete specifics (tools, prices, benchmarks)

=== OPTIONAL ENHANCEMENTS (USE SELECTIVELY) ===

short_narration:
- Use for 2–3 middle scenes ONLY if they benefit from a standalone cut
- Must be a tighter, punchier version (not a duplicate)
- End with: "Subscribe for daily AI news."

video_query:
- Only include if a clear real-world visual exists
- Must describe a specific shot:
  (e.g. "freelancer editing video on laptop in dim room, timeline visible")

infographic_data:
- Only include when comparing numbers
- Use ONE of these formats:
  {
    "type": "bar_chart",
    "labels": [...],
    "values": [...]
  }
  or
  {
    "type": "stat_card",
    "label": "...",
    "value": "..."
  }
  or
  {
    "type": "comparison",
    "items": [
      {"label": "...", "value": "..."}
    ]
  }

=== SCREENSHOT RULES ===
- Use only real, known public URLs
- Max 3 scenes
- Only for tools with visual UI relevance
- If unsure → set null

=== VISUAL PROMPTS ===
- Always include (fallback)
- Must describe a specific, realistic scene
- Avoid abstract "AI concept" visuals

=== OUTPUT FORMAT ===
Return a single valid JSON object:

{
  "title": "...",
  "description": "... (200–400 words, end with TIMESTAMPS_AUTOFILL)",
  "tags": [...],
  "hook_variants": [...],
  "scenes": [
    {
      "heading": "...",
      "narration": "...",
      "visual_prompt": "...",
      "screenshot_url": null,
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
        raw_url = (s.get("screenshot_url") or "").strip()
        screenshot_url = raw_url if raw_url.startswith("http") else None
        scenes.append(Scene(
            idx=i,
            heading=s.get("heading", f"Scene {i+1}"),
            narration=s.get("narration", ""),
            visual_prompt=s.get("visual_prompt", ""),
            infographic_data=infographic,
            video_query=(s.get("video_query") or "").strip() or None,
            screenshot_url=screenshot_url,
            short_narration=(s.get("short_narration") or "").strip() or None,
            source_quote=(s.get("source_quote") or "").strip() or None,
            quote_attribution=(s.get("quote_attribution") or "").strip() or None,
        ))

    hook_variants: list[str] = data.get("hook_variants") or []
    chosen_hook = hook_variants[0] if hook_variants else ""

    word_count = sum(len(sc.narration.split()) for sc in scenes)
    shorts_count = sum(1 for sc in scenes if sc.short_narration)
    screenshot_count = sum(1 for sc in scenes if sc.screenshot_url)
    logger.info(
        f"Topic script ready: {word_count} words, {len(scenes)} scenes, "
        f"{shorts_count} shorts, {screenshot_count} live screenshots"
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
