"""Pacing auditor: detect flat zones in a script and insert pattern interrupts.

A "flat zone" is 2+ consecutive scenes with no engagement signals —
no questions, no numbers, no conflict words, no short punchy sentences.
Viewers drop off precisely in these zones.

How it works:
  1. Score each scene 0-4 based on rule-based signals (cheap, no LLM).
  2. Find runs of scenes all scoring below threshold → flat zones.
  3. For each flat zone, call GPT once to generate a 1-2 sentence
     pattern interrupt and append it to the last scene in the zone.
  4. Return the modified script + an audit report.

Pattern interrupts are one of:
  - Rhetorical question tied to content  ("Wait — does this work without coding?")
  - Unexpected reframe                   ("Here's the part the docs don't mention.")
  - Mini-cliffhanger                     ("And this is where it gets weird.")

Non-fatal: if GPT fails, the original script is returned unchanged.
"""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from src.retry_utils import make_openai_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.cost_tracker import get_ledger
from src.script_generator import Scene, VideoScript


# ── Config ────────────────────────────────────────────────────────────────────

_FLAT_THRESHOLD = 1          # scenes scoring ≤ this are considered flat
_FLAT_RUN_MIN   = 2          # how many consecutive flat scenes trigger an interrupt
_MAX_INTERRUPTS = 3          # cap so we don't over-inject

_CONFLICT_WORDS = {
    "but", "actually", "wrong", "worse", "problem", "catch", "however",
    "surprising", "unexpected", "wait", "careful", "warning", "except",
    "despite", "although", "yet", "though", "flaw", "downside", "risk",
    "costs", "hidden", "nobody", "lie", "myth", "overhyped",
}


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class SceneScore:
    idx: int
    score: int                  # 0-4 engagement signals found
    signals: list[str]          # which signals fired


@dataclass
class FlatZone:
    scene_indices: list[int]    # indices of the flat scenes
    interrupt_at: int           # which scene gets the inject (last of zone)
    interrupt_text: str = ""    # filled after GPT call


@dataclass
class PacingAuditResult:
    script: VideoScript
    scores: list[SceneScore]
    flat_zones: list[FlatZone]
    interrupts_added: int
    modified: bool


# ── Rule-based scoring ────────────────────────────────────────────────────────

def _score_scene(scene: Scene) -> SceneScore:
    text  = scene.narration
    lower = text.lower()
    signals: list[str] = []

    # 1. Rhetorical question
    if re.search(r"\?", text):
        signals.append("question")

    # 2. Specific number / stat (digits, %, $, x multiplier)
    if re.search(r"\b\d[\d,.]*\s*(%|\$|x\b|k\b|m\b|billion|million|thousand)", lower) \
            or re.search(r"\$\d", text):
        signals.append("number")

    # 3. Conflict / contrast word
    if any(w in lower.split() for w in _CONFLICT_WORDS):
        signals.append("conflict")

    # 4. Short punchy sentence (≤ 7 words) — signals pace variation
    sentences = re.split(r"[.!?]+", text)
    if any(1 <= len(s.split()) <= 7 for s in sentences):
        signals.append("short_sentence")

    return SceneScore(idx=scene.idx, score=len(signals), signals=signals)


def _find_flat_zones(scores: list[SceneScore]) -> list[FlatZone]:
    """Return flat zones, skipping the intro (idx 0) and final scene."""
    zones: list[FlatZone] = []
    run: list[int] = []

    interior = scores[1:-1]  # skip intro and sign-off
    for sc in interior:
        if sc.score <= _FLAT_THRESHOLD:
            run.append(sc.idx)
        else:
            if len(run) >= _FLAT_RUN_MIN:
                zones.append(FlatZone(
                    scene_indices=run[:],
                    interrupt_at=run[-1],
                ))
            run = []

    if len(run) >= _FLAT_RUN_MIN:
        zones.append(FlatZone(scene_indices=run[:], interrupt_at=run[-1]))

    return zones[:_MAX_INTERRUPTS]


# ── GPT interrupt generation ──────────────────────────────────────────────────

_SYSTEM = """\
You are a YouTube script editor fixing pacing. A "flat zone" is a run of scenes
with no questions, numbers, conflict, or punchy sentences — viewers drop off here.

Your job: write a 1-2 sentence pattern interrupt to append to the last scene
of the flat zone. The interrupt must:
- Feel like a natural transition, not an awkward insert
- Use ONE of these techniques:
    a) Rhetorical question tied to the specific content  ("Wait — does this work if you have no coding skills?")
    b) Unexpected reframe or counter-intuitive observation ("Here's the thing nobody in the reviews mentions.")
    c) Mini-cliffhanger that teases the next scene       ("And what I found next genuinely surprised me.")
- Be max 30 words
- Match the conversational, first-person tone of the surrounding narration
- NOT repeat something already said

Return JSON:
{
  "interrupt": "the 1-2 sentence pattern interrupt",
  "technique": "question | reframe | cliffhanger"
}"""

_USER_TMPL = """\
Topic: {subject}

Flat zone scenes (these are the dull spots):
{flat_block}

Next scene after the flat zone:
{next_scene}

Write a pattern interrupt to append to the last flat scene.
"""


def _log_usage(usage, label: str = "") -> None:
    if not usage:
        return
    details = usage.prompt_tokens_details
    cached  = getattr(details, "cached_tokens", 0) if details else 0
    get_ledger().record_llm(
        tag=label or "pacing",
        model=settings.openai_model,
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_tokens=cached,
    )


def _generate_interrupt(
    client,
    zone: FlatZone,
    all_scenes: list[Scene],
    subject: str,
) -> str:
    flat_block = "\n\n".join(
        f"[Scene {i}] {all_scenes[i].heading}\n{all_scenes[i].narration[:300]}"
        for i in zone.scene_indices
    )
    next_idx = zone.interrupt_at + 1
    next_scene = (
        f"[Scene {next_idx}] {all_scenes[next_idx].heading}\n"
        f"{all_scenes[next_idx].narration[:200]}"
        if next_idx < len(all_scenes)
        else "(end of script)"
    )
    try:
        resp = client.chat.completions.create(
            model=settings.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TMPL.format(
                    subject=subject,
                    flat_block=flat_block,
                    next_scene=next_scene,
                )},
            ],
            temperature=0.8,
        )
        _log_usage(resp.usage, "pacing-interrupt")
        data = json.loads(resp.choices[0].message.content or "{}")
        return (data.get("interrupt") or "").strip()
    except Exception as exc:
        logger.warning(f"Interrupt generation failed for zone {zone.scene_indices}: {exc}")
        return ""


# ── Public API ────────────────────────────────────────────────────────────────

def audit_pacing(script: VideoScript, subject: str = "") -> PacingAuditResult:
    """Score all scenes, find flat zones, inject GPT pattern interrupts.

    Returns a PacingAuditResult. script.scenes is never mutated in place —
    a new VideoScript is returned.
    """
    scores = [_score_scene(s) for s in script.scenes]
    flat_zones = _find_flat_zones(scores)

    _log_scores(scores, flat_zones)

    if not flat_zones:
        logger.info("Pacing audit: no flat zones detected — script looks good")
        return PacingAuditResult(
            script=script,
            scores=scores,
            flat_zones=flat_zones,
            interrupts_added=0,
            modified=False,
        )

    client = make_openai_client()
    subject = subject or script.title

    for zone in flat_zones:
        text = _generate_interrupt(client, zone, script.scenes, subject)
        zone.interrupt_text = text

    # Apply interrupts to scene copies
    modified_scenes = [Scene(**{**s.__dict__}) for s in script.scenes]
    added = 0
    for zone in flat_zones:
        if not zone.interrupt_text:
            continue
        target = modified_scenes[zone.interrupt_at]
        target.narration = target.narration.rstrip() + "\n\n" + zone.interrupt_text
        added += 1

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
        f"Pacing audit done: {len(flat_zones)} flat zone(s) found, "
        f"{added} interrupt(s) injected"
    )
    return PacingAuditResult(
        script=modified_script,
        scores=scores,
        flat_zones=flat_zones,
        interrupts_added=added,
        modified=added > 0,
    )


def _log_scores(scores: list[SceneScore], flat_zones: list[FlatZone]) -> None:
    flat_indices = {i for z in flat_zones for i in z.scene_indices}
    rows = []
    for sc in scores:
        flag = " ← FLAT" if sc.idx in flat_indices else ""
        rows.append(
            f"  scene {sc.idx:02d}: score={sc.score} [{', '.join(sc.signals) or 'none'}]{flag}"
        )
    logger.debug("Pacing scores:\n" + "\n".join(rows))
