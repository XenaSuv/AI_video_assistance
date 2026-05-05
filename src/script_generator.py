"""Generate a 15-minute video script + YouTube metadata from scraped news items.

Structured output: scenes/segments so the video generator can make matching b-roll.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from openai import OpenAI
from src.retry_utils import make_openai_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.cost_tracker import get_ledger
from src.analytics import get_proven_hooks, get_proven_titles, top_keywords
from src.scraper import NewsItem
from src.editorial_brain import editorial_plan_to_prompt, EditorialPlan


@dataclass
class Scene:
    """A single narrated segment. Each scene gets one b-roll clip + TTS chunk."""
    idx: int
    heading: str
    narration: str              # what the voice says
    visual_prompt: str          # DALL-E 3 prompt; used as fallback when no screenshot
    duration_sec: int = 0       # filled after TTS timing is known
    screenshot_key: str | None = None    # weekly tutorials: real screenshot from curated library
    screenshot_url: str | None = None    # topic segments: capture any public URL on the fly
    short_narration: str | None = None   # ~120 words written for a standalone Shorts cut
    infographic_data: dict | None = None # animated chart/stat; skips DALL-E when set
    video_query: str | None = None       # stock B-roll search term; beats DALL-E, loses to infographic
    source_quote: str | None = None      # verbatim quote (≤30 words) from a named source
    quote_attribution: str | None = None # "Name, Role · Org · domain.com"
    scene_type: str = "image"            # assigned by SceneVarietyEngine
    scene_intent: str = "explain"        # explain | shock | data | reaction


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
Your scripts sound like a human AI YouTuber with an editorial voice —
a knowledgeable friend, not a press release or corporate blog post. Write like a real creator with energy, curiosity, skepticism, and specific detail. Cite papers, companies, researchers, and real numbers. Be honest about limitations; avoid hype. No "welcome back to the channel" filler — assume viewers know the channel.

=== NARRATION STYLE ===
- Write conversationally: contractions ("it's", "you'll", "they've", "here's") are good
- Every scene: topic hook → key facts → implication for viewers → bridge to next topic
- Use concrete specifics: model names, benchmark scores, funding amounts, release dates
- Vary sentence length: short punchy sentences for impact, longer ones for explanation
- Speak as "we" when contextualising the field; never "I"
- Close the final scene with a genuine, non-cringey call to like/subscribe

=== INFOGRAPHIC DATA ===
Set "infographic_data" to replace the static DALL-E image with an animated chart whenever
a scene has concrete numbers, model comparisons, benchmark scores, or chronological events.
Leave null for narrative, opinion, or abstract concept scenes.

Supported types and exact JSON shapes:

  bar_chart — ranked comparison (up to 8 items):
    {"type":"bar_chart","title":"HumanEval Scores","unit":"%",
     "items":[{"label":"GPT-4o","value":90.2},{"label":"Claude 3","value":88.5}]}

  timeline — chronological sequence (up to 6 events):
    {"type":"timeline","title":"GPT Release History",
     "events":[{"date":"Jun 2020","label":"GPT-3"},{"date":"Mar 2023","label":"GPT-4"}]}

  stat_card — single headline number with context:
    {"type":"stat_card","value":"$6.6B","label":"OpenAI Funding Round",
     "context":"Valuation reaches $157B"}

  comparison — head-to-head two items:
    {"type":"comparison","title":"HumanEval","unit":"%",
     "left":{"label":"GPT-3.5","value":48.1},"right":{"label":"GPT-4o","value":90.2}}

Values are always plain numbers (no currency/percent in the number itself).

=== B-ROLL VIDEO QUERY ===
Set "video_query" on 2-3 scenes with real-world physical action where stock footage adds
visual interest. Keep terms generic — no brand names, logos, or UI-specific terms.
Good examples: "developer coding laptop", "data center servers blinking lights",
"team meeting office whiteboard", "person presenting screen audience",
"smartphone scrolling app", "AI robot arm factory floor", "researcher laboratory"
Use null for: data/comparison scenes (use infographic), abstract AI concepts,
policy/opinion scenes, any scene where a DALL-E illustration communicates better.

=== VISUAL PROMPTS ===
Each "visual_prompt" should be a photorealistic DALL-E 3 description that uses the
headline's actual company, product, or service names. Aim for a serious editorial
news feel rather than colorful fantasy. For OpenAI, Anthropic, or Google stories,
prefer a brand-inspired background prompt with subtle corporate color cues and
no obvious logos.

=== SHORT NARRATION ===
Fill "short_narration" for EXACTLY 2-3 middle scenes (not scene 0 intro, not the last
sign-off). Each must be fully self-contained (~120 words):
- First sentence: news hook — "Breaking:", "Just in:", "Here's what just happened:"
- Body: one story with all context included; viewer has NOT seen the full video
- Final sentence exactly: "Subscribe for daily AI news."
- Never use "In today's video", "As I mentioned", or cross-references to other scenes
Set short_narration to null for the opening intro and closing sign-off only.

=== SOURCE QUOTES ===
For 2-4 scenes, pull a verbatim or near-verbatim quote directly from the source material.
- source_quote: ≤30 words. Must be something a named person or official org actually said
  or wrote. Never fabricate or paraphrase into quote form — only use text present in the
  news item summaries or titles. Leave null if no attributable quote exists.
- quote_attribution: "Full Name, Title · Organization · domain.com"
  Examples: "Sam Altman, CEO · OpenAI · openai.com"
            "Google DeepMind Research Team · deepmind.google"
            "Anthropic · May 2025 · anthropic.com"
  Leave null when source_quote is null.

=== YOUTUBE METADATA ===
title: under 70 characters, one emoji maximum, tease the #1 story
description: 200-400 words — concise summary, then "(TIMESTAMPS_AUTOFILL)", then
  "Sources:" section listing all article URLs from the news items
tags: 15-25 lowercase tags, mix broad ("ai news", "artificial intelligence") and
  specific ("gpt-4o", "anthropic", "llama") terms for YouTube algorithm discovery
hook_variants: exactly 3 distinct hooks, each 5-10 seconds when spoken aloud:
  variant 0 — lead with the most surprising single fact or number from today's news
  variant 1 — open with a provocative "what if" or rhetorical question
  variant 2 — bold declarative statement of the biggest implication

=== STRICT REQUIREMENTS ===
- You MUST follow the provided hook exactly as the opening line
- You MUST reflect the persona tone in every paragraph
- You MUST structure the script according to scene_plan and the editorial plan
- You MUST write like a human YouTuber, not a robotic summary
- You MUST include at least one opinionated statement per scene
- Avoid neutral news tone

=== OUTPUT FORMAT ===
Output MUST be a single valid JSON object. No markdown fences. No commentary.
{
  "title": "...",
  "description": "...",
  "tags": ["tag1", "tag2", "..."],
  "hook_variants": ["fact-lead hook", "question hook", "bold statement hook"],
  "scenes": [
    {
            "heading": "chyron title max 8 words",
            "narration": "spoken text, conversational, contractions OK",
            "visual_prompt": "DALL-E 3 prompt — always required even when infographic_data is set. Use photorealistic news-style imagery with company/product specifics.",
      "video_query": null,
      "infographic_data": null,
      "short_narration": null,
      "source_quote": null,
      "quote_attribution": null
    },
    {
      "heading": "...",
      "narration": "...",
      "visual_prompt": "...",
      "video_query": null,
      "infographic_data": null,
      "short_narration": "Breaking: [self-contained story ~120 words]. Subscribe for daily AI news.",
      "source_quote": "We achieved state-of-the-art results across all major benchmarks simultaneously",
      "quote_attribution": "Sam Altman, CEO · OpenAI · openai.com"
    }
  ]
}"""


USER_PROMPT_TMPL = """Write a {target_words}-word daily AI news script (~{target_minutes} min at 150 wpm) \
broken into {num_scenes} scenes.

Close with a sign-off that invites likes/subscribes without being cringey.

News items (ranked by relevance):
{items_block}"""


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


def _log_usage(usage, label: str = "", model: str = "") -> None:
    """Log OpenAI token usage and record in the cost ledger."""
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
        model=model or settings.openai_model,
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_tokens=cached,
    )


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


def _build_recommendation_block() -> str:
    """Return a prompt block with historical hook/title guidance."""
    proven_hooks = get_proven_hooks(limit=3)
    proven_titles = get_proven_titles(limit=3)
    hot_terms = [term for term, _ in top_keywords(min_views=150, top_n=6)]

    sections = []
    if proven_hooks:
        sections.append("Proven hook styles from past top videos:\n" + "\n".join(f"- {h}" for h in proven_hooks))
    if proven_titles:
        sections.append("Proven title formulas from past top videos:\n" + "\n".join(f"- {t}" for t in proven_titles))
    if hot_terms:
        sections.append("Trending keyword ideas to use naturally in title/tags/description:\n" + ", ".join(hot_terms))

    if not sections:
        return ""
    return "=== ANALYTICS INSIGHTS ===\n" + "\n\n".join(sections)


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
        _log_usage(resp.usage, "shorts-fill")
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
    editorial_plan: EditorialPlan | dict[str, Any] | None = None,
) -> VideoScript:
    """Call GPT to produce a structured VideoScript.

    *data_dir* is passed to hook_selector so the best hook variant is chosen
    automatically. When omitted the first variant is used (useful in tests).
    """
    if not items:
        raise ValueError("No news items provided")

    client = make_openai_client()
    target_minutes = max(1, round(settings.script_target_words / 150))
    user_prompt = USER_PROMPT_TMPL.format(
        target_words=settings.script_target_words,
        target_minutes=target_minutes,
        num_scenes=num_scenes,
        items_block=_build_items_block(items),
    )
    if data_dir is not None:
        try:
            analytics_block = _build_recommendation_block()
            if analytics_block:
                user_prompt += "\n\n" + analytics_block
                logger.info("Added analytics guidance to script prompt")
        except Exception as exc:
            logger.warning(f"Failed to add analytics guidance: {exc}")
    if editorial_plan is not None:
        try:
            if isinstance(editorial_plan, dict):
                editorial_plan = EditorialPlan(
                    selected_stories=editorial_plan.get("selected_stories", []),
                    editorial_plan=editorial_plan.get("editorial_plan", []),
                    global_style=editorial_plan.get("global_style", {}),
                )
            user_prompt += "\n\n" + editorial_plan_to_prompt(editorial_plan)
            logger.info("Added editorial plan guidance to script prompt")
        except Exception as exc:
            logger.warning(f"Failed to add editorial plan guidance: {exc}")

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
    _log_usage(resp.usage, "script")

    raw = resp.choices[0].message.content or "{}"
    data = json.loads(raw)

    scenes = []
    for i, s in enumerate(data.get("scenes", [])):
        infographic = s.get("infographic_data") or None
        if infographic and not isinstance(infographic, dict):
            infographic = None
        video_q  = (s.get("video_query") or "").strip() or None
        short_nar = (s.get("short_narration") or "").strip() or None
        src_quote = (s.get("source_quote") or "").strip() or None
        src_attr  = (s.get("quote_attribution") or "").strip() or None
        visual_prompt = (s.get("visual_prompt") or "").strip()
        if not visual_prompt:
            visual_prompt = (s.get("heading") or "").strip()
        if not visual_prompt:
            visual_prompt = (s.get("narration") or "").strip().split(".")[0]
        visual_prompt = visual_prompt or "A cinematic AI newsroom scene"
        scenes.append(Scene(
            idx=i,
            heading=s["heading"],
            narration=s["narration"],
            visual_prompt=visual_prompt,
            infographic_data=infographic,
            video_query=video_q,
            short_narration=short_nar,
            source_quote=src_quote,
            quote_attribution=src_attr,
        ))
    infographic_count = sum(1 for sc in scenes if sc.infographic_data)
    broll_count       = sum(1 for sc in scenes if sc.video_query)
    quote_count       = sum(1 for sc in scenes if sc.source_quote)

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
        f"{broll_count} B-roll | {infographic_count} infographic | "
        f"{shorts_count} shorts | {quote_count} quote cards"
    )
    return script


def generate_short_script(news_item, hook: str) -> str:
    """Generate a compact script for a 30-second YouTube Short."""

    content = getattr(news_item, "summary", None)
    if content is None and isinstance(news_item, dict):
        content = news_item.get("summary", "")
    content = str(content or "")

    client = make_openai_client()
    angle = random.choice(["shock", "useful", "money", "lazy"])
    prompt = f"""
    You are creating a viral YouTube Shorts script.

    IMPORTANT:
    Do NOT sound like news.
    Do NOT say "today", "recently", "announced".
    Do NOT sound like AI-generated content.

    Act like a real person who just tested this AI tool.

    ---

    News:
    {content}

    Hook:
    {hook}

    ---

    Script rules:

    1. Start with a strong hook (first 1–2 seconds)
    2. Speak in FIRST PERSON ("I tried", "I tested")
    3. Show a RESULT (what happened)
    4. Explain WHY it matters in simple terms
    5. Make it feel real and practical

    ---

    Tone:
    - natural
    - slightly эмоциональный
    - conversational
    - confident

    ---

    Style:
    - short sentences
    - no fluff
    - no generic phrases
    - no "this technology allows..."

    ---

    Length:
    Max 80 words.

    ---

    Example style:

    "I tried this new AI and it built a website in 10 seconds.
    Like… fully working.
    You just type what you want — and it does everything.
    Honestly, this could replace basic dev work."

    ---

    Return ONLY the script.
    """
    prompt += f"\nFocus angle: {angle}"

    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )
    return response.choices[0].message.content or ""


if __name__ == "__main__":
    from src.scraper import scrape_all

    news = scrape_all(top_n=8)
    script = generate_script(news)
    script.save(settings.output_dir / "script.json")
    print(f"Saved → {settings.output_dir / 'script.json'}")
    print(f"Title: {script.title}")
