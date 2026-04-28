"""Generate weekly tutorial scripts about major AI tools.

Supported tools: claude, chatgpt, gemini
Each has its own topic pool and tailored GPT system prompt.
Topic rotation is tracked per-tool in data/weekly_topics_{tool}.json.

Script reuses VideoScript / Scene so all downstream components
(voice, video, shorts, thumbnail, upload) work unchanged.
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from loguru import logger
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.screenshot_capturer import available_keys, has_key, url_for
from src.script_generator import Scene, VideoScript

_REUSE_COOLDOWN_WEEKS = 26   # ~6 months before a topic repeats


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

_SHORTS_FILL_PROMPT = """\
For each scene below, write a standalone YouTube Shorts script (~120 words).

Rules:
- First sentence must be a hook: "Here's a trick...", "Did you know...", "Stop doing X..."
- Cover exactly ONE actionable tip from that scene
- Must make sense without watching the rest of the video
- Last sentence: "Watch the full tutorial for more tips like this."
- No filler, no "In this video", direct and punchy

Return JSON: {{"scenes": [{{"idx": <N>, "short_narration": "..."}}]}}

Scenes to fill:
{scenes_block}
"""


# ─────────────────── Tool definitions ───────────────────

@dataclass
class _Tool:
    key: str
    name: str
    topics: list[str]
    system_prompt: str


_WEEKLY_SYSTEM_SUFFIX = """\

=== VISUAL SOURCES (priority order — pick at most one non-null per scene) ===
1. screenshot_key   — real tool UI screenshot from the provided list
2. infographic_data — animated chart for pricing, benchmarks, step counts:
     bar_chart:  {"type":"bar_chart","title":"...","unit":"...","items":[{"label":"...","value":0}]}
     stat_card:  {"type":"stat_card","value":"70B","label":"Parameters","context":"10× GPT-3"}
     comparison: {"type":"comparison","title":"...","unit":"%","left":{"label":"A","value":0},"right":{"label":"B","value":0}}
     timeline:   {"type":"timeline","title":"...","events":[{"date":"Jan 2024","label":"..."}]}
3. video_query      — 2-5 word Pexels search for 1-2 scenes with physical action
     (typing, presenting, whiteboard). Generic only, no brand names.
     Examples: "developer coding laptop", "person presenting whiteboard", "team discussing screen"
4. visual_prompt    — DALL-E fallback (always required regardless of other fields)
Set all unused source fields to null.

=== SHORT NARRATION — MANDATORY ===
Exactly 3-4 scenes MUST have short_narration filled (non-null).
Choose scenes 1-5 (not the opening intro or closing outro) — each covering ONE
self-contained tip a viewer can use immediately without watching the full video.
Each short_narration must:
  - Open with a hook: "Here's a trick...", "Did you know...", "Stop doing X..."
  - Cover the ONE key point of that scene in ~120 words
  - End exactly with: "Watch the full tutorial for more tips like this."
Setting short_narration to null for ALL scenes is an error — Shorts drive channel growth.

=== SOURCE QUOTES ===
For 2-3 scenes, pull a real quote from the official documentation, release notes,
blog post, or tutorial being covered.
- source_quote: ≤30 words, verbatim from official source. Never fabricate.
  Leave null if no direct quote is available for that scene.
- quote_attribution: "Name, Role · Organization · domain.com"
  Examples: "Anthropic Engineering · anthropic.com"
            "OpenAI Documentation Team · platform.openai.com"
  Leave null when source_quote is null.

=== OUTPUT FORMAT ===
Return ONLY valid JSON, no markdown fences, no commentary.
{
  "title": "YouTube title <= 70 chars, include the tool name",
  "description": "YouTube description 3-5 sentences + (TIMESTAMPS_AUTOFILL)",
  "tags": ["10-15 tags"],
  "hook_variants": [
    "variant 0: most surprising fact or number (5-10s spoken)",
    "variant 1: provocative question (5-10s spoken)",
    "variant 2: bold statement of biggest benefit (5-10s spoken)"
  ],
  "scenes": [
    {
      "heading": "scene title",
      "narration": "full spoken text 280-340 words",
      "visual_prompt": "DALL-E 3 prompt (always required)",
      "screenshot_key": null,
      "short_narration": null,
      "video_query": null,
      "infographic_data": null,
      "source_quote": null,
      "quote_attribution": null
    },
    {
      "heading": "...",
      "narration": "...",
      "visual_prompt": "...",
      "screenshot_key": null,
      "short_narration": "Here's a trick most people miss... [~120 words ending with CTA]",
      "video_query": null,
      "infographic_data": null,
      "source_quote": "Prompt caching reduces costs by up to 90% for long repeated contexts",
      "quote_attribution": "Anthropic Engineering · anthropic.com"
    }
  ]
}
(7 scenes total; short_narration filled for 3-4 of the middle scenes)"""


_TOOLS: dict[str, _Tool] = {

    "claude": _Tool(
        key="claude",
        name="Claude",
        topics=[
            # Getting started
            "Getting started with the Claude API in Python — a complete beginner guide",
            "Claude.ai tips and hidden features most users don't know about",
            "Choosing the right Claude model: Opus vs Sonnet vs Haiku explained",
            # Prompt engineering
            "Prompt engineering masterclass: how to write prompts that actually work with Claude",
            "Chain-of-thought prompting with Claude — when and how to use it",
            "Role prompting and system prompts: unlocking Claude's full potential",
            "How to structure complex multi-step tasks for Claude",
            "Getting consistent JSON output from Claude every time",
            # Coding
            "Using Claude Code CLI — the complete guide for developers",
            "Claude for code review: catch bugs and improve code quality automatically",
            "Test-driven development with Claude: write tests first, let Claude implement",
            "Refactoring legacy code with Claude — practical walkthrough",
            "Claude as your pair programmer: best practices and workflows",
            "Debugging with Claude: turn error messages into fixes instantly",
            # Writing & content
            "Using Claude for long-form writing: reports, essays, and documentation",
            "Claude for email writing: professional, fast, and on-brand",
            "Content repurposing with Claude: one article to blog, tweet, LinkedIn, email",
            "How to use Claude for research: summarizing papers and finding key insights",
            # Advanced API
            "Tool use and function calling with Claude — build AI agents",
            "Prompt caching with Claude API: cut costs by up to 90%",
            "Streaming responses from Claude API for real-time UX",
            "Building a multi-turn chatbot with Claude API from scratch",
            "Claude vision: analyzing images, charts, and screenshots",
            "Batch processing with Claude API: handle thousands of requests efficiently",
            # Practical
            "Claude for data analysis: ask questions about your CSV files",
            "Building a personal knowledge base assistant with Claude",
            "Automating customer support with Claude — without losing the human touch",
            "Using Claude to learn anything faster: study guide generator",
            "Claude for translation and multilingual content at scale",
            "How to use Claude for SEO: keyword research to finished article",
            "Using Claude to summarize meetings, PDFs, and long documents",
            # Claude Code / developer workflows
            "Claude Code hooks and automation: customize your AI coding workflow",
            "MCP servers with Claude: connect Claude to your tools and data",
            "Building a Claude-powered CLI tool from scratch",
            "Integrating Claude into VS Code and JetBrains IDEs",
            "Extended thinking with Claude: solving problems that need deep reasoning",
        ],
        system_prompt="""\
You are a YouTube scriptwriter specialising in educational tutorials about Anthropic's Claude AI.
Your tutorials are clear, practical, and actionable. Viewers are developers and
knowledge workers who want to get more value from Claude.

Rules:
- 7 scenes, each ~280-340 words of narration (total ~2100-2400 words)
- Tone: friendly expert, like a senior developer sharing tips with a colleague
- Each scene has a clear purpose: intro → concept → demo steps → tips → summary
- visual_prompt: vivid DALL-E 3 description of what should appear on screen
  (UI mockups, code on screen, person at laptop, abstract tech visuals, etc.)
  Do NOT include text overlays or watermarks in the visual description."""
        + _WEEKLY_SYSTEM_SUFFIX,
    ),

    "chatgpt": _Tool(
        key="chatgpt",
        name="ChatGPT",
        topics=[
            # Getting started
            "Getting started with ChatGPT: the complete beginner guide for 2025",
            "ChatGPT Plus vs free tier: is it worth the upgrade?",
            "ChatGPT memory: how to set it up and get personalized responses",
            "ChatGPT voice mode: hands-free AI conversations explained",
            # Prompt engineering
            "Prompt engineering for ChatGPT: techniques that actually work",
            "How to write the perfect system prompt for ChatGPT custom GPTs",
            "Getting consistent structured output from ChatGPT every time",
            "Advanced ChatGPT prompting: few-shot, chain-of-thought, and personas",
            # Coding & data
            "ChatGPT for coding: from beginner scripts to complex debugging",
            "Advanced Data Analysis in ChatGPT: analyze Excel and CSV files with AI",
            "ChatGPT Code Interpreter tutorial: data visualization made easy",
            "Using ChatGPT to review and refactor your code",
            # Writing & content
            "ChatGPT for content creation: blog posts, social media, and emails",
            "How to use ChatGPT for copywriting without sounding robotic",
            "ChatGPT for academic writing: research, citations, and essays",
            "Content repurposing with ChatGPT: one idea, ten formats",
            # Custom GPTs & GPT Store
            "Building your own custom GPT — step-by-step tutorial",
            "Best custom GPTs in the GPT Store and how to use them",
            "How to publish your custom GPT to the GPT Store",
            # API & developers
            "OpenAI API tutorial: getting started with GPT-4o in Python",
            "Function calling with the OpenAI API — build intelligent agents",
            "Streaming ChatGPT responses with the OpenAI API",
            "Fine-tuning GPT models: when to do it and how",
            "OpenAI Assistants API: build a stateful AI agent from scratch",
            # Multimodal
            "ChatGPT vision: analyze images, screenshots, and documents",
            "DALL-E 3 inside ChatGPT: the complete image generation guide",
            "ChatGPT with browsing: real-time web search and research",
            # Practical use cases
            "ChatGPT for business: automate reports, emails, and analysis",
            "Using ChatGPT as a personal productivity system",
            "ChatGPT for customer support: templates and automation ideas",
            "How to use ChatGPT for market research and competitor analysis",
            "ChatGPT for language learning: practice conversations and grammar",
            "Using ChatGPT Projects to organize your AI workflows",
        ],
        system_prompt="""\
You are a YouTube scriptwriter specialising in educational tutorials about OpenAI's ChatGPT.
Your tutorials are clear, practical, and actionable. Viewers range from beginners to
professionals who want to get more productivity from ChatGPT.

Rules:
- 7 scenes, each ~280-340 words of narration (total ~2100-2400 words)
- Tone: approachable and enthusiastic, like a tech-savvy friend showing you cool tricks
- Each scene has a clear purpose: intro → concept → demo steps → tips → summary
- visual_prompt: vivid DALL-E 3 description of what should appear on screen
  (ChatGPT interface, code editor, person using laptop, abstract tech visuals, etc.)
  Do NOT include text overlays, logos, or watermarks in the visual description."""
        + _WEEKLY_SYSTEM_SUFFIX,
    ),

    "gemini": _Tool(
        key="gemini",
        name="Gemini",
        topics=[
            # Getting started
            "Getting started with Google Gemini: the complete 2025 guide",
            "Gemini Advanced vs free tier: features, pricing, and when to upgrade",
            "Gemini in Google Workspace: AI inside Gmail, Docs, and Sheets",
            "Google AI Studio tutorial: build Gemini-powered apps for free",
            # Prompt engineering
            "Prompt engineering for Gemini: tips and techniques that work",
            "Getting structured JSON output from Gemini every time",
            "Gemini system instructions: how to customize AI behavior",
            "Advanced Gemini prompting: multimodal, long-context, and reasoning",
            # Long context & documents
            "Gemini 1.5 Pro long context: analyze entire books and codebases",
            "Upload and analyze PDFs with Gemini — practical tutorial",
            "Using Gemini to summarize and extract insights from long documents",
            "Gemini for research: synthesize multiple papers and sources at once",
            # Coding
            "Gemini for coding: generate, debug, and explain code",
            "Using Gemini in VS Code with the Gemini Code Assist extension",
            "Gemini API in Python: getting started with the Google AI SDK",
            "Building an AI agent with the Gemini API and function calling",
            # Multimodal
            "Gemini vision: analyze images, charts, and screenshots",
            "Gemini video understanding: analyze YouTube videos and recordings",
            "Gemini Audio: transcribe and analyze audio files",
            "Gemini multimodal prompts: combining text, images, and data",
            # Google products integration
            "NotebookLM tutorial: AI-powered research assistant by Google",
            "Gemini in Google Search: how AI Overviews work and when to use them",
            "Using Gemini with Google Drive: summarize and chat with your files",
            "Gemini in Android: the complete guide to Google's AI assistant",
            # API & developers
            "Gemini API tutorial: models, pricing, and best practices",
            "Gemini function calling and tool use — build intelligent agents",
            "Gemini embeddings API: semantic search and similarity matching",
            "Vertex AI + Gemini: enterprise AI deployment on Google Cloud",
            # Practical use cases
            "Gemini for business productivity: emails, reports, and analysis",
            "Using Gemini for data analysis in Google Sheets",
            "Gemini for content creation: articles, social posts, and scripts",
            "How to use Gemini for competitive research and market analysis",
            "Gemini for education: study tools, quizzes, and explanations",
        ],
        system_prompt="""\
You are a YouTube scriptwriter specialising in educational tutorials about Google Gemini AI.
Your tutorials are clear, practical, and actionable. Viewers range from casual users to
developers who want to harness Gemini's unique strengths (long context, multimodal, Google integration).

Rules:
- 7 scenes, each ~280-340 words of narration (total ~2100-2400 words)
- Tone: calm and professional, like a Google power user sharing workflow tips
- Each scene has a clear purpose: intro → concept → demo steps → tips → summary
- visual_prompt: vivid DALL-E 3 description of what should appear on screen
  (Google interface, code editor, person at desk, abstract colorful visuals, etc.)
  Do NOT include text overlays, logos, or watermarks in the visual description."""
        + _WEEKLY_SYSTEM_SUFFIX,
    ),
}


def get_tool(key: str) -> _Tool:
    if key not in _TOOLS:
        raise ValueError(f"Unknown tool '{key}'. Choose from: {list(_TOOLS)}")
    return _TOOLS[key]


# ─────────────────── Topic selection ───────────────────

def _topics_file(data_dir: Path, tool_key: str) -> Path:
    return data_dir / f"weekly_topics_{tool_key}.json"


def _load_used_topics(data_dir: Path, tool_key: str) -> list[dict]:
    path = _topics_file(data_dir, tool_key)
    return json.loads(path.read_text()).get("used", []) if path.exists() else []


def save_used_topic(data_dir: Path, tool_key: str, topic: str) -> None:
    path = _topics_file(data_dir, tool_key)
    used = _load_used_topics(data_dir, tool_key)
    used.append({"topic": topic, "date": date.today().isoformat()})
    path.write_text(json.dumps({"used": used}, indent=2))


def pick_topic(data_dir: Path, tool_key: str) -> str:
    """Return a topic for *tool_key* not used in the last REUSE_COOLDOWN_WEEKS weeks."""
    tool   = get_tool(tool_key)
    cutoff = date.today() - timedelta(weeks=_REUSE_COOLDOWN_WEEKS)
    recently_used = {
        e["topic"] for e in _load_used_topics(data_dir, tool_key)
        if date.fromisoformat(e["date"]) >= cutoff
    }
    available = [t for t in tool.topics if t not in recently_used]
    if not available:
        logger.warning(f"{tool.name}: all topics used recently — restarting rotation")
        available = tool.topics
    chosen = random.choice(available)
    logger.info(f"{tool.name} weekly topic: {chosen}")
    return chosen


# ─────────────────── Script generation ───────────────────

_USER_TEMPLATE = """\
Create a complete YouTube tutorial script for this topic:

"{topic}"

Available screenshot keys for {tool_name}:
{keys_block}
"""


def _format_keys_block(tool_key: str) -> str:
    keys = available_keys(tool_key)
    if not keys:
        return "  (none — use DALL-E for every scene)"
    return "\n".join(f"  - {k}: {url_for(tool_key, k)}" for k in keys)


def _fill_missing_short_narrations(
    scenes: list[Scene],
    client: OpenAI,
    min_count: int = 3,
) -> None:
    """Ensure at least *min_count* scenes have short_narration; fill gaps via GPT."""
    have = [s for s in scenes if s.short_narration]
    if len(have) >= min_count:
        return

    needed = min_count - len(have)
    # Prefer middle scenes (skip first intro + last outro)
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
                "content": _SHORTS_FILL_PROMPT.format(scenes_block=scenes_block),
            }],
            temperature=0.7,
        )
        _log_usage(resp.usage, "weekly-shorts-fill")
        data = json.loads(resp.choices[0].message.content or "{}")
        idx_to_scene = {s.idx: s for s in to_fill}
        for item in data.get("scenes", []):
            idx  = item.get("idx")
            text = (item.get("short_narration") or "").strip()
            if text and idx in idx_to_scene:
                idx_to_scene[idx].short_narration = text
                logger.info(f"  auto-filled short_narration for scene {idx}")
    except Exception as exc:
        logger.warning(f"short_narration auto-fill failed (non-fatal): {exc}")


def generate_tutorial_script(
    topic: str,
    tool_key: str,
    data_dir: Path | None = None,
) -> VideoScript:
    """Call GPT to write a full tutorial script for *topic* about *tool_key*."""
    tool   = get_tool(tool_key)
    client = OpenAI(api_key=settings.openai_api_key)
    logger.info(f"Generating {tool.name} tutorial script for: {topic}")

    user_prompt = _USER_TEMPLATE.format(
        topic       = topic,
        tool_name   = tool.name,
        keys_block  = _format_keys_block(tool_key),
    )

    resp = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": tool.system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.8,
    )
    _log_usage(resp.usage, f"weekly-{tool_key}")

    raw    = json.loads(resp.choices[0].message.content)
    scenes = []
    for i, s in enumerate(raw["scenes"]):
        key = s.get("screenshot_key") or None
        if key and not has_key(tool_key, key):
            logger.warning(f"Scene {i}: GPT picked unknown screenshot_key '{key}' — falling back to DALL-E")
            key = None
        short_nar  = s.get("short_narration") or None
        infographic = s.get("infographic_data") or None
        if infographic and not isinstance(infographic, dict):
            infographic = None
        video_q   = (s.get("video_query") or "").strip() or None
        src_quote = (s.get("source_quote") or "").strip() or None
        src_attr  = (s.get("quote_attribution") or "").strip() or None
        scenes.append(Scene(
            idx=i,
            heading=s["heading"],
            narration=s["narration"],
            visual_prompt=s["visual_prompt"],
            screenshot_key=key,
            short_narration=short_nar,
            infographic_data=infographic,
            video_query=video_q,
            source_quote=src_quote,
            quote_attribution=src_attr,
        ))

    # Ensure 3-4 Shorts are present; auto-fill any missing ones
    _fill_missing_short_narrations(scenes, client, min_count=3)

    # Dynamic hook selection
    hook_variants: list[str] = raw.get("hook_variants") or []
    if not hook_variants:
        hook_variants = [raw.get("hook", "")]
    hook_variants = [h for h in hook_variants if h]

    if data_dir and len(hook_variants) > 1:
        from src.hook_selector import pick_hook
        chosen_hook = pick_hook(hook_variants, data_dir, context_key=tool_key)
    else:
        chosen_hook = hook_variants[0] if hook_variants else ""

    shorts_count      = sum(1 for sc in scenes if sc.short_narration)
    infographic_count = sum(1 for sc in scenes if sc.infographic_data)
    broll_count       = sum(1 for sc in scenes if sc.video_query)
    script = VideoScript(
        title=raw["title"],
        description=raw["description"],
        tags=raw["tags"],
        hook=chosen_hook,
        hook_variants=hook_variants,
        scenes=scenes,
    )
    logger.info(
        f"{tool.name} tutorial: '{script.title}' | {len(scenes)} scenes | "
        f"{shorts_count} shorts | {broll_count} B-roll | "
        f"{infographic_count} infographics | "
        f"hook variant {hook_variants.index(chosen_hook) + 1}/{len(hook_variants)}"
    )
    return script
