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
from src.script_generator import Scene, VideoScript

_SYSTEM = """\
You are a professional translator and YouTube content localiser.
Translate the JSON object I provide into {lang}.

Rules:
- Translate: title, description, hook, scenes[].heading, scenes[].narration
- For tags: produce {lang} YouTube SEO tags (translate concepts, not just words)
- DO NOT translate or modify: scenes[].visual_prompt, scenes[].idx, scenes[].duration_sec
- Return valid JSON with exactly the same structure as the input.
- Keep the same tone: educational, engaging, slightly conversational.
"""


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

    logger.info(f"Translating script to {target_lang} (~{len(script.scenes)} scenes)…")
    response = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system",  "content": _SYSTEM.format(lang=target_lang)},
            {"role": "user",    "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )

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
