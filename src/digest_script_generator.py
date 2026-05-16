"""Generate a "Week in AI" digest script from the most recent daily scripts.

Strategy:
  1. Collect the 6 most recent daily scripts before the digest date
  2. Extract each day's title + scene headings + narration summaries
  3. Call GPT to write a compact 7-8 minute recap video (~900-1100 words)

The digest script uses the same VideoScript / Scene types so the full
voice → video → shorts → thumbnail → upload pipeline works unchanged.
Output is written to output/digest/YYYY-MM-DD/ using Sunday's date.
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

from loguru import logger
from openai import OpenAI
from src.retry_utils import make_openai_client

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.cost_tracker import get_ledger
from src.script_generator import Scene, VideoScript
from src.response_validator import parse_llm_json, validate_llm_response

# Target word count for the digest (vs ~2200 for daily full episode)
_DIGEST_WORDS = 950
_DIGEST_SCENES = 8   # hook + 6 story scenes + sign-off


_SYSTEM_PROMPT = """\
You are the writer for a YouTube channel's weekly AI news digest.

GOAL:
Turn this week’s AI news into a clear narrative, not just a list of updates.

--------------------------------
EDITORIAL ANGLE (REQUIRED)
--------------------------------
Identify one underlying theme of the week
(e.g. "AI is getting cheaper", "agents are becoming real", "Big Tech vs open-source")

All stories should loosely connect to this theme.

--------------------------------
SELECTION RULES
--------------------------------
- 6 most impactful stories
- Prioritize real-world impact (money, users, regulation, capabilities)
- Avoid minor or speculative updates

--------------------------------
STRUCTURE
--------------------------------
8 scenes total (~950–1050 words):

1. Hook:
   - Highlight the week's theme
   - Create curiosity

2–7. Six story scenes:
   Each must include:
   - What happened (clear, specific)
   - Why it matters (non-generic, concrete)
   - One distinct angle per story (avoid repetition)

8. Sign-off:
   - Reinforce the weekly theme
   - Forward-looking insight (not generic)

--------------------------------
SCENE VARIETY (IMPORTANT)
--------------------------------
Vary pacing and focus across stories:
- Some more factual
- Some more analytical
- Some more practical (what viewers can do)

Avoid repeating the same structure 6 times.

--------------------------------
TONE
--------------------------------
- Smart, concise, slightly opinionated
- No filler, no clichés
- Write like explaining to an informed friend

--------------------------------
HOOK VARIANTS
--------------------------------
- Must reflect the weekly theme (not just one story)
- 3 distinct angles:
  1. surprising fact
  2. provocative question
  3. bold implication

--------------------------------
VISUAL PROMPTS
--------------------------------
- Photorealistic, grounded in real scenes
- Specific moments, not abstract AI visuals
- No logos, no public figures

--------------------------------
SOURCE QUOTES
--------------------------------
- Only include if certain
- Otherwise null

--------------------------------
OUTPUT FORMAT
--------------------------------
Return ONLY valid JSON:

{
  "title": "This Week in AI: [2-3 word theme] | Week of [date]",
  "description": "2–4 sentence summary + (TIMESTAMPS_AUTOFILL)",
  "tags": ["10–15 tags"],
  "hook_variants": ["...", "...", "..."],
  "scenes": [
    {
      "heading": "...",
      "narration": "...",
      "visual_prompt": "...",
      "source_quote": null,
      "quote_attribution": null
    }
  ]
}"""


_USER_TEMPLATE = """\
Here are the AI news highlights from this week (Mon-Sat), week of {week_of}.
Write a Sunday "This Week in AI" digest script covering the 6 most important
stories in a fast-paced ~8-minute video.

WEEK'S HIGHLIGHTS:
{week_block}"""


# ─────────────────── Collect daily scripts ───────────────────

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


def _load_daily_script(output_dir: Path, data_dir: Path, day: date) -> dict | None:
    """Try data/digest_scripts/{date}.json first (CI-persisted), then output/{date}/script.json (local)."""
    candidates = [
        data_dir / "digest_scripts" / f"{day.isoformat()}.json",
        output_dir / day.isoformat() / "script.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")
    return None


def _summarise_daily(day: date, data: dict) -> str:
    """Format one day's script into a concise block for the digest prompt."""
    day_name = day.strftime("%A %b %d")
    lines = [f"=== {day_name}: {data.get('title', '(untitled)')} ==="]
    for s in data.get("scenes", []):
        heading = s.get("heading", "")
        # First sentence of narration as the 1-line summary
        narration = s.get("narration", "")
        first_sentence = narration.split(".")[0].strip() + "."
        if heading:
            lines.append(f"  - {heading}: {first_sentence}")
    return "\n".join(lines)


def save_for_digest(data_dir: Path, day: date, script_path: Path) -> None:
    """Copy a daily script.json into data/digest_scripts/ for cross-run persistence."""
    dest_dir = data_dir / "digest_scripts"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{day.isoformat()}.json"
    dest.write_text(script_path.read_text())
    logger.debug(f"Digest: saved daily script to {dest}")


def collect_week_scripts(
    output_dir: Path,
    data_dir: Path | None = None,
    sunday: date | None = None,
) -> tuple[list[date], list[str]]:
    """Return (found_dates, summary_blocks) for the 6 most recent daily scripts."""
    sunday = sunday or date.today()
    data_dir = data_dir or (output_dir.parent / "data")
    found_dates: list[date] = []
    blocks: list[str] = []

    day = sunday - timedelta(days=1)
    while day >= sunday - timedelta(days=21) and len(found_dates) < 6:
        data = _load_daily_script(output_dir, data_dir, day)
        if data:
            found_dates.append(day)
            blocks.append(_summarise_daily(day, data))
            logger.info(f"Digest: loaded {day.isoformat()} — {data.get('title', '?')}")
        else:
            logger.debug(f"Digest: no script found for {day.isoformat()}")
        day -= timedelta(days=1)

    found_dates.reverse()
    blocks.reverse()
    return found_dates, blocks


# ─────────────────── Script generation ───────────────────

def generate_digest_script(
    week_blocks: list[str],
    week_of: str,
    data_dir: Path | None = None,
) -> VideoScript:
    """Call GPT to produce a weekly digest VideoScript."""
    if not week_blocks:
        raise ValueError("No daily scripts found for the week — cannot generate digest")

    client = make_openai_client()
    week_block = "\n\n".join(week_blocks)

    user_prompt = _USER_TEMPLATE.format(
        week_block=week_block,
        week_of=week_of,
    )

    logger.info(f"Generating Sunday digest script ({len(week_blocks)} days of content)")

    resp = client.chat.completions.create(
        model=settings.openai_model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.75,
    )
    _log_usage(resp.usage, "digest")

    raw = parse_llm_json(resp.choices[0].message.content, "generate_digest_script")
    validate_llm_response(raw, ["title", "description", "tags", "scenes"], "generate_digest_script")
    scenes = [
        Scene(
            idx=i,
            heading=s.get("heading") or f"Scene {i}",
            narration=s.get("narration") or "",
            visual_prompt=s.get("visual_prompt") or s.get("heading") or "AI news digest",
            source_quote=(s.get("source_quote") or "").strip() or None,
            quote_attribution=(s.get("quote_attribution") or "").strip() or None,
        )
        for i, s in enumerate(raw["scenes"])
    ]

    hook_variants: list[str] = raw.get("hook_variants") or []
    if not hook_variants:
        hook_variants = [raw.get("hook", "")]
    hook_variants = [h for h in hook_variants if h]

    if data_dir and len(hook_variants) > 1:
        from src.hook_selector import pick_hook
        chosen_hook = pick_hook(hook_variants, data_dir, context_key="digest")
    else:
        chosen_hook = hook_variants[0] if hook_variants else ""

    script = VideoScript(
        title=raw["title"],
        description=raw["description"],
        tags=raw["tags"],
        hook=chosen_hook,
        hook_variants=hook_variants,
        scenes=scenes,
    )

    word_count = sum(len(s.narration.split()) for s in scenes)
    logger.info(
        f"Digest script: '{script.title}' | {len(scenes)} scenes | {word_count} words | "
        f"hook variant {hook_variants.index(chosen_hook) + 1}/{len(hook_variants)}"
    )
    return script
