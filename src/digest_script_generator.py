"""Generate a Sunday "Week in AI" digest script from the week's daily scripts.

Strategy:
  1. Scan output/ for script.json files dated Mon–Sat of the current week
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import Scene, VideoScript

# Target word count for the digest (vs ~2200 for daily full episode)
_DIGEST_WORDS = 950
_DIGEST_SCENES = 8   # hook + 6 story scenes + sign-off


_SYSTEM_PROMPT = """\
You are the writer for a YouTube channel's weekly AI news digest.
Every Sunday you publish a punchy "This Week in AI" recap that covers
the biggest stories of the past week in one concise video.

Rules:
- Tone: informed, energetic, conversational — like a smart friend summarizing the week
- 8 scenes total (~950 words): one hook, six story scenes, one sign-off
- Each scene ~100-130 words of narration
- DO NOT say "welcome back" or pad with filler
- visual_prompt: vivid DALL-E 3 description (no logos, no real people)
- Return ONLY valid JSON, no markdown fences"""


_USER_TEMPLATE = """\
Here are the AI news highlights from this week (Mon-Sat).
Write a Sunday "This Week in AI" digest script that covers the 6 most
important stories in a fast-paced ~8-minute video.

WEEK'S HIGHLIGHTS:
{week_block}

Return JSON with this exact structure:
{{
  "title":       "...",   // e.g. "This Week in AI: [2-3 word theme] | Week of {week_of}"
  "description": "...",   // 2-4 sentence summary + (TIMESTAMPS_AUTOFILL)
  "tags":        ["..."], // 10-15 tags
  "hook_variants": ["...", "...", "..."],  // 3 distinct 5-10s hooks:
                                           //   0 = most surprising stat/fact
                                           //   1 = provocative question
                                           //   2 = boldest implication
  "scenes": [
    {{
      "heading":      "...",  // short chapter title (max 8 words)
      "narration":    "...",  // ~100-130 words of spoken text
      "visual_prompt":"..."   // DALL-E 3 image prompt, 2-3 sentences
    }},
    ... (8 scenes total)
  ]
}}
"""


# ─────────────────── Collect daily scripts ───────────────────

def _week_dates(sunday: date) -> list[date]:
    """Return Mon–Sat dates preceding the given Sunday."""
    return [sunday - timedelta(days=d) for d in range(6, 0, -1)]


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
    """Return (found_dates, summary_blocks) for Mon-Sat preceding *sunday*."""
    sunday = sunday or date.today()
    data_dir = data_dir or (output_dir.parent / "data")
    found_dates: list[date] = []
    blocks: list[str] = []

    for day in _week_dates(sunday):
        data = _load_daily_script(output_dir, data_dir, day)
        if data:
            found_dates.append(day)
            blocks.append(_summarise_daily(day, data))
            logger.info(f"Digest: loaded {day.isoformat()} — {data.get('title', '?')}")
        else:
            logger.debug(f"Digest: no script found for {day.isoformat()}")

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

    client = OpenAI(api_key=settings.openai_api_key)
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
