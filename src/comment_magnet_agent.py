"""Comment magnet agent: drive comments by ending with one sharp, polarizing question.

YouTube's algorithm heavily weighs comment velocity in the first hour.
A generic "let me know what you think!" gets ignored.
A specific, slightly controversial question tied to something in the video
makes people feel compelled to respond — especially if they disagree.

How it works:
  1. Reads all scene narrations to find the most debatable claim or surprising finding.
  2. Generates ONE question designed to polarize: people who agree AND people
     who disagree both feel the urge to comment.
  3. Appends it to the second-to-last scene (not the final sign-off scene)
     so it lands while engagement is still high, before the subscribe ask.

Non-fatal: if GPT fails the original script is returned unchanged.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from src.retry_utils import make_openai_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.cost_tracker import get_ledger
from src.script_generator import Scene, VideoScript


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class CommentMagnetResult:
    script: VideoScript
    question: str           # the generated question
    inserted_at_scene: int  # which scene it was appended to
    modified: bool


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
YYou are a YouTube engagement specialist focused on maximizing comments.

Your task: generate ONE high-performing comment-driving question based on the script.

PROCESS:
1. Identify the most debatable or tension-filled claim in the script
2. Choose ONE conflict angle (price, usefulness, trust, effort, hype vs reality)
3. Turn it into a sharp, personal question

REQUIREMENTS:
- One sentence only (max 16–18 words)
- Must be tied to a specific claim or moment in the script
- Must invite disagreement (not universally agreeable)
- Must feel personal ("you", "would you", "do you")
- Must include tension (trade-off, doubt, or challenge)
- Ends with "?"

STRONG FORMATS (prefer these):
- "Am I wrong that…?"
- "Be honest — would you…?"
- "Is X actually better than Y?"
- "Would you pay/do this, or is it not worth it?"

TONE:
- Natural, conversational, slightly provocative
- Not aggressive, but not neutral

AVOID:
- Generic questions
- Safe or obvious takes
- Multiple questions
- Vague wording

OUTPUT:
{
  "question": "...",
  "reasoning": "Which specific claim or moment in the script this question is based on"
}"""

_USER_TMPL = """\
Script topic: {subject}
Format: {format}

Key scenes (narration excerpts):
{scenes_block}

Write one comment-magnet question to append near the end of this video.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_usage(usage, label: str = "") -> None:
    if not usage:
        return
    details = usage.prompt_tokens_details
    cached = getattr(details, "cached_tokens", 0) if details else 0
    get_ledger().record_llm(
        tag=label or "comment-magnet",
        model=settings.openai_model,
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_tokens=cached,
    )


def _build_scenes_block(scenes: list[Scene]) -> str:
    # Send middle scenes (skip intro and final sign-off) — they have the meat
    core = scenes[1:-1] if len(scenes) > 2 else scenes
    lines = []
    for s in core:
        preview = s.narration[:300].replace("\n", " ")
        lines.append(f"[Scene {s.idx}: {s.heading}]\n{preview}")
    return "\n\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────

def apply_comment_magnet(
    script: VideoScript,
    subject: str = "",
    format_name: str = "",
) -> CommentMagnetResult:
    """Append a comment-magnet question to the second-to-last scene.

    *subject* and *format_name* help GPT write a more specific question.
    If omitted, GPT infers from the script content alone.
    """
    if len(script.scenes) < 2:
        return CommentMagnetResult(
            script=script, question="", inserted_at_scene=-1, modified=False
        )

    client = make_openai_client()
    scenes_block = _build_scenes_block(script.scenes)

    logger.info("Running comment-magnet agent…")
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TMPL.format(
                    subject=subject or script.title,
                    format=format_name or "topic deep-dive",
                    scenes_block=scenes_block,
                )},
            ],
            temperature=0.85,   # slightly higher — we want spicy, not safe
        )
        _log_usage(resp.usage, "comment-magnet")
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning(f"Comment-magnet agent failed (non-fatal): {exc}")
        return CommentMagnetResult(
            script=script, question="", inserted_at_scene=-1, modified=False
        )

    question = (data.get("question") or "").strip()
    if not question:
        logger.warning("Comment-magnet agent returned empty question — skipping")
        return CommentMagnetResult(
            script=script, question="", inserted_at_scene=-1, modified=False
        )

    # Target: second-to-last scene (has high retention, before the sign-off)
    target_idx = len(script.scenes) - 2

    modified_scenes = [Scene(**{**s.__dict__}) for s in script.scenes]
    modified_scenes[target_idx].narration = (
        modified_scenes[target_idx].narration.rstrip()
        + f"\n\n{question}"
    )

    modified_script = VideoScript(
        title=script.title,
        description=script.description,
        tags=script.tags,
        hook=script.hook,
        hook_variants=script.hook_variants,
        scenes=modified_scenes,
        raw_json=script.raw_json,
    )

    logger.info(
        f"Comment-magnet added to scene {target_idx}: "
        f"{question[:80]}{'…' if len(question) > 80 else ''}"
    )
    return CommentMagnetResult(
        script=modified_script,
        question=question,
        inserted_at_scene=target_idx,
        modified=True,
    )
