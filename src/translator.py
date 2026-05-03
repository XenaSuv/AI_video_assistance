"""Translate a VideoScript to a target language via a single GPT call.

Only human-readable fields are translated:
  title, description, tags, hook, scenes[].heading, scenes[].narration

scenes[].visual_prompt is intentionally left untouched — DALL-E images
are language-agnostic and are reused across all language variants.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.cost_tracker import get_ledger
from src.script_generator import Scene, VideoScript

_SYSTEM = """\
You are a professional translator and YouTube content localiser.

Rules:
- The user message specifies the target language — translate into that language only
- Translate: title, description, hook, scenes[].heading, scenes[].narration
- For tags: produce target-language YouTube SEO tags (translate concepts, not just words)
- DO NOT translate or modify: scenes[].visual_prompt, scenes[].idx, scenes[].duration_sec
- Return valid JSON with exactly the same structure as the input
- Keep the same tone: educational, engaging, slightly conversational
"""


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


def translate_script(script: VideoScript, target_lang: str) -> VideoScript:
    """Return a new VideoScript with all text fields translated to *target_lang*.

    *target_lang* is a plain-language name, e.g. ``"Russian"`` or ``"Spanish"``.
    The original VideoScript is not mutated.
    """
    client = OpenAI(api_key=settings.openai_api_key)

    payload = {
        "title":       script.title,
        "description": script.description,
        "tags":        script.tags,
        "hook":        script.hook,
        "scenes": [
            {
                "idx":           s.idx,
                "heading":       s.heading,
                "narration":     s.narration,
                "visual_prompt": s.visual_prompt,
                "duration_sec":  s.duration_sec,
            }
            for s in script.scenes
        ],
    }

    user_content = f"Translate to {target_lang}:\n{json.dumps(payload, ensure_ascii=False)}"

    logger.info(f"Translating script to {target_lang} (~{len(script.scenes)} scenes)…")
    response = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_content},
        ],
    )
    _log_usage(response.usage, f"translate-{target_lang}")

    data = json.loads(response.choices[0].message.content)

    translated = VideoScript(
        title       = data["title"],
        description = data["description"],
        tags        = data["tags"],
        hook        = data["hook"],
        scenes      = [
            Scene(
                idx           = s["idx"],
                heading       = s["heading"],
                narration     = s["narration"],
                visual_prompt = s["visual_prompt"],   # original — not translated
                duration_sec  = s["duration_sec"],
            )
            for s in data["scenes"]
        ],
    )
    logger.info(f"Translation complete: {translated.title!r}")
    return translated
