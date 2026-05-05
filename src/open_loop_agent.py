"""Open-loop agent: boost viewer retention by creating curiosity gaps.

How it works:
  1. Scans all scene narrations to find 2-3 surprising / counterintuitive moments.
  2. Rewrites the intro scene to tease those moments as open loops
     ("Stick around — by scene 4 I'll show you why this tool is actually worse
      than the free alternative nobody talks about").
  3. Inserts a short closing line at each loop's target scene so the payoff
     feels earned and intentional.

This runs as a post-processing pass after generate_topic_script() and before
voice synthesis. If it fails for any reason the original script is returned
unchanged (non-fatal).
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


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class OpenLoop:
    """A single curiosity gap: teased in the intro, closed at target_scene_idx."""
    teaser: str         # short phrase added to intro ("why X is actually broken")
    closing: str        # 1-2 sentences inserted at start of target scene
    target_scene_idx: int


@dataclass
class OpenLoopResult:
    script: VideoScript
    loops: list[OpenLoop]
    modified: bool      # False if GPT failed or returned nothing useful


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a YouTube retention specialist focused on maximizing watch time.

Your task: add 2–3 high-impact open loops to the script.

An open loop = a meaningful unresolved tension introduced early and paid off later.

--------------------------------
STEP 1 — IDENTIFY STRONG LOOPS
--------------------------------
Select 2–3 moments with REAL tension:
- clear reversal (expected vs actual)
- meaningful risk, failure, or trade-off
- surprising or non-obvious outcome

Prioritize moments that would make a viewer think:
“I need to see how this turns out.”

Avoid minor or low-stakes moments.

--------------------------------
STEP 2 — PRIORITIZE
--------------------------------
- 1 PRIMARY loop (strongest, most curiosity-driving)
- 1–2 secondary loops

--------------------------------
STEP 3 — WRITE TEASERS
--------------------------------
For each loop:
- 10–14 words max
- No spoilers
- Must hint at a SPECIFIC tension
- Must create curiosity, not just information

Use formats like:
- Direct question
- Contradiction
- Stakes

FORBIDDEN:
- “you won’t believe”
- “this changes everything”
- vague phrases

--------------------------------
STEP 4 — MATCH PAYOFF
--------------------------------
Write a 1–2 sentence closing line that:
- clearly resolves the SAME tension
- feels satisfying (not flat or purely technical)

--------------------------------
STEP 5 — REWRITE INTRO
--------------------------------
- Keep original meaning and pacing
- Embed teasers naturally (max 2 per ~3 sentences)
- Do not overload the intro

--------------------------------
PLACEMENT AWARENESS
--------------------------------
Prefer loops that resolve:
- mid-video (to prevent drop-off)
- or at key turning points

--------------------------------
CONSTRAINTS
--------------------------------
- Do not invent content
- Each loop must map to a real moment
- target_scene_idx must be accurate

--------------------------------
OUTPUT
--------------------------------
{
  "rewritten_intro": "...",
  "loops": [
    {
      "teaser": "...",
      "closing": "...",
      "target_scene_idx": 2,
      "priority": "primary"
    }
  ]
}"""

_USER_TMPL = """\
Script scenes (scene 0 is the intro):

{scenes_block}

Rewrite scene 0's narration to naturally embed 2-3 open-loop teasers.
Add a short closing line to each loop's target scene.
"""


# ── Agent ─────────────────────────────────────────────────────────────────────

def _log_usage(usage, label: str = "") -> None:
    if not usage:
        return
    details = usage.prompt_tokens_details
    cached = getattr(details, "cached_tokens", 0) if details else 0
    get_ledger().record_llm(
        tag=label or "open-loop",
        model=settings.openai_model,
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_tokens=cached,
    )


def _build_scenes_block(scenes: list[Scene]) -> str:
    lines = []
    for s in scenes:
        # Send only the first 400 chars per scene so the prompt stays compact
        preview = s.narration[:400].replace("\n", " ")
        lines.append(f"[Scene {s.idx}] {s.heading}\n{preview}\n")
    return "\n".join(lines)


def apply_open_loops(script: VideoScript) -> OpenLoopResult:
    """Run the open-loop pass on *script*. Returns OpenLoopResult (always safe to use)."""
    if len(script.scenes) < 3:
        logger.info("Open-loop agent skipped: script has fewer than 3 scenes")
        return OpenLoopResult(script=script, loops=[], modified=False)

    client = make_openai_client()
    scenes_block = _build_scenes_block(script.scenes)

    logger.info("Running open-loop agent…")
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TMPL.format(scenes_block=scenes_block)},
            ],
            temperature=0.7,
        )
        _log_usage(resp.usage, "open-loop")
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:
        logger.warning(f"Open-loop agent failed (non-fatal): {exc}")
        return OpenLoopResult(script=script, loops=[], modified=False)

    rewritten_intro = (data.get("rewritten_intro") or "").strip()
    raw_loops = data.get("loops") or []

    if not rewritten_intro or not raw_loops:
        logger.warning("Open-loop agent returned empty result — skipping")
        return OpenLoopResult(script=script, loops=[], modified=False)

    loops: list[OpenLoop] = []
    for item in raw_loops:
        idx = item.get("target_scene_idx")
        closing = (item.get("closing") or "").strip()
        teaser = (item.get("teaser") or "").strip()
        if not (teaser and closing and isinstance(idx, int)):
            continue
        # Clamp to valid scene range
        idx = max(1, min(idx, len(script.scenes) - 1))
        loops.append(OpenLoop(teaser=teaser, closing=closing, target_scene_idx=idx))

    if not loops:
        logger.warning("Open-loop agent: no valid loops parsed — skipping")
        return OpenLoopResult(script=script, loops=[], modified=False)

    # Apply changes to scene narrations (mutate copies, not originals)
    modified_scenes = [
        Scene(**{**s.__dict__}) for s in script.scenes
    ]

    # Rewrite intro
    modified_scenes[0].narration = rewritten_intro

    # Prepend closing lines to target scenes
    for loop in loops:
        target = modified_scenes[loop.target_scene_idx]
        target.narration = loop.closing + "\n\n" + target.narration

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
        f"Open-loop agent done: {len(loops)} loop(s) — "
        + ", ".join(f"scene 0→{l.target_scene_idx}" for l in loops)
    )
    return OpenLoopResult(script=modified_script, loops=loops, modified=True)
