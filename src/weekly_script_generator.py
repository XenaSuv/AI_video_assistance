"""Generate a weekly tutorial script about Claude AI.

Topic pool rotates week-by-week; used topics are tracked in
data/weekly_topics.json so we never repeat within ~6 months.

Script reuses VideoScript / Scene so all downstream components
(voice, video, shorts, thumbnail, upload) work unchanged.
"""
from __future__ import annotations

import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

from loguru import logger
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import Scene, VideoScript

# ─────────────────── Topic pool ───────────────────

TOPIC_POOL: list[str] = [
    # Getting started
    "Getting started with the Claude API in Python — a complete beginner guide",
    "Claude.ai tips and hidden features most users don't know about",
    "Choosing the right Claude model: Opus vs Sonnet vs Haiku explained",

    # Prompt engineering
    "Prompt engineering masterclass: how to write prompts that actually work",
    "Chain-of-thought prompting with Claude — when and how to use it",
    "Role prompting and system prompts: unlocking Claude's full potential",
    "How to structure complex multi-step tasks for Claude",
    "Getting consistent JSON output from Claude every time",

    # Coding with Claude
    "Using Claude Code CLI — the complete guide for developers",
    "Claude for code review: catch bugs and improve code quality automatically",
    "Test-driven development with Claude: write tests first, let Claude implement",
    "Refactoring legacy code with Claude — practical walkthrough",
    "Claude as your pair programmer: best practices and workflows",
    "Debugging with Claude: turn error messages into fixes instantly",

    # Writing and content
    "Using Claude for long-form writing: reports, essays, and documentation",
    "Claude for email writing: professional, fast, and on-brand",
    "Content repurposing with Claude: one article → blog, tweet, LinkedIn, email",
    "How to use Claude for research: summarizing papers and finding key insights",

    # Advanced API features
    "Tool use and function calling with Claude — build AI agents",
    "Prompt caching with Claude API: cut costs by up to 90%",
    "Streaming responses from Claude API for real-time UX",
    "Building a multi-turn chatbot with Claude API from scratch",
    "Claude vision: analyzing images, charts, and screenshots",
    "Batch processing with Claude API: handle thousands of requests efficiently",

    # Practical use cases
    "Claude for data analysis: ask questions about your CSV files",
    "Building a personal knowledge base assistant with Claude",
    "Automating customer support with Claude — without losing the human touch",
    "Using Claude to learn anything faster: study guide generator",
    "Claude for translation and multilingual content at scale",
    "How to use Claude for SEO: keyword research to finished article",
    "Claude for legal and contract review: what it can and can't do",
    "Using Claude to summarize meetings, PDFs, and long documents",

    # Claude Code / developer workflows
    "Claude Code hooks and automation: customize your AI coding workflow",
    "MCP servers with Claude: connect Claude to your tools and data",
    "Building a Claude-powered CLI tool from scratch",
    "Integrating Claude into VS Code and JetBrains IDEs",
]

_TOPICS_FILE_NAME = "weekly_topics.json"
_REUSE_COOLDOWN_WEEKS = 26  # don't repeat a topic for ~6 months


# ─────────────────── Topic selection ───────────────────

def _load_used_topics(data_dir: Path) -> list[dict]:
    path = data_dir / _TOPICS_FILE_NAME
    if path.exists():
        return json.loads(path.read_text()).get("used", [])
    return []


def _save_used_topic(data_dir: Path, topic: str) -> None:
    path = data_dir / _TOPICS_FILE_NAME
    used = _load_used_topics(data_dir)
    used.append({"topic": topic, "date": date.today().isoformat()})
    path.write_text(json.dumps({"used": used}, indent=2))


def pick_topic(data_dir: Path) -> str:
    """Return a topic not used in the last REUSE_COOLDOWN_WEEKS weeks."""
    used = _load_used_topics(data_dir)
    cutoff = date.today() - timedelta(weeks=_REUSE_COOLDOWN_WEEKS)
    recently_used = {
        e["topic"] for e in used
        if date.fromisoformat(e["date"]) >= cutoff
    }
    available = [t for t in TOPIC_POOL if t not in recently_used]
    if not available:
        logger.warning("All topics used recently — restarting rotation")
        available = TOPIC_POOL
    chosen = random.choice(available)
    logger.info(f"Weekly topic: {chosen}")
    return chosen


# ─────────────────── Script generation ───────────────────

_SYSTEM_PROMPT = """\
You are a YouTube scriptwriter specialising in educational tutorials about Claude AI.
Your tutorials are clear, practical, and actionable. Viewers are developers and
knowledge workers who want to get more value from Claude.

Rules:
- 7 scenes, each ~280–340 words of narration (total ≈ 2100–2400 words)
- Tone: friendly, expert, like a senior developer sharing tips with a colleague
- Each scene has a clear purpose: intro → concept → demo steps → tips → summary
- visual_prompt: vivid DALL-E 3 description of what should appear on screen
  (UI mockups, code on screen, person at laptop, abstract tech visuals, etc.)
  Do NOT include text overlays or watermarks in the visual description.
- Return ONLY valid JSON, no markdown fences, no commentary outside the JSON.
"""

_USER_TEMPLATE = """\
Create a complete YouTube tutorial script for this topic:

"{topic}"

Return JSON with this exact structure:
{{
  "title": "...",           // YouTube title, ≤ 70 chars, include "Claude" keyword
  "description": "...",    // YouTube description, 3-5 sentences + (TIMESTAMPS_AUTOFILL)
  "tags": ["...", ...],    // 10–15 tags
  "hook": "...",           // 1–2 punchy sentences read in the first 5 seconds
  "scenes": [
    {{
      "heading": "...",        // short chapter title
      "narration": "...",      // full spoken text, 280-340 words
      "visual_prompt": "..."   // DALL-E 3 image prompt, 2-3 sentences
    }},
    ... (7 scenes total)
  ]
}}
"""


def generate_tutorial_script(topic: str) -> VideoScript:
    """Call GPT to write a full tutorial script for the given topic."""
    client = OpenAI(api_key=settings.openai_api_key)
    logger.info(f"Generating tutorial script for: {topic}")

    resp = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _USER_TEMPLATE.format(topic=topic)},
        ],
        temperature=0.8,
    )

    raw = json.loads(resp.choices[0].message.content)

    scenes = [
        Scene(
            idx=i,
            heading=s["heading"],
            narration=s["narration"],
            visual_prompt=s["visual_prompt"],
        )
        for i, s in enumerate(raw["scenes"])
    ]

    script = VideoScript(
        title=raw["title"],
        description=raw["description"],
        tags=raw["tags"],
        hook=raw["hook"],
        scenes=scenes,
    )
    logger.info(f"Tutorial script: '{script.title}' | {len(scenes)} scenes")
    return script
