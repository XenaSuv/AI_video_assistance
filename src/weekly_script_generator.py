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
from src.script_generator import Scene, VideoScript

_REUSE_COOLDOWN_WEEKS = 26   # ~6 months before a topic repeats


# ─────────────────── Tool definitions ───────────────────

@dataclass
class _Tool:
    key: str
    name: str
    topics: list[str]
    system_prompt: str


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
  Do NOT include text overlays or watermarks in the visual description.
- Return ONLY valid JSON, no markdown fences.""",
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
  Do NOT include text overlays, logos, or watermarks in the visual description.
- Return ONLY valid JSON, no markdown fences.""",
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
  Do NOT include text overlays, logos, or watermarks in the visual description.
- Return ONLY valid JSON, no markdown fences.""",
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

Return JSON with this exact structure:
{{
  "title": "...",           // YouTube title, <= 70 chars, include the tool name
  "description": "...",    // YouTube description, 3-5 sentences + (TIMESTAMPS_AUTOFILL)
  "tags": ["...", ...],    // 10-15 tags
  "hook": "...",           // 1-2 punchy sentences read in the first 5 seconds
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


def generate_tutorial_script(topic: str, tool_key: str) -> VideoScript:
    """Call GPT to write a full tutorial script for *topic* about *tool_key*."""
    tool   = get_tool(tool_key)
    client = OpenAI(api_key=settings.openai_api_key)
    logger.info(f"Generating {tool.name} tutorial script for: {topic}")

    resp = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": tool.system_prompt},
            {"role": "user",   "content": _USER_TEMPLATE.format(topic=topic)},
        ],
        temperature=0.8,
    )

    raw    = json.loads(resp.choices[0].message.content)
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
    logger.info(f"{tool.name} tutorial: '{script.title}' | {len(scenes)} scenes")
    return script
