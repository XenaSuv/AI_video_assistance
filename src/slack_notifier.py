"""Send pipeline status notifications to a Slack channel via Incoming Webhook.

Set SLACK_WEBHOOK_URL in your environment (or .env) to enable.
If the variable is absent or the webhook call fails, notifications are
silently skipped — the pipeline is never blocked by Slack being down.
"""
from __future__ import annotations

import datetime as dt

from loguru import logger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.log_filter import scrub_message
from src.retry_utils import http_post as _http_post


_PIPELINE_EMOJI = {
    "daily":    "📰",
    "weekly":   "🎓",
    "digest":   "📋",
    "breaking": "🚨",
}


def _post(payload: dict) -> None:
    url = settings.slack_webhook_url
    if not url:
        return
    try:
        _http_post(url, json=payload, timeout=10)
    except Exception as exc:
        logger.warning(f"Slack notification failed (non-fatal): {exc}")


def _duration_str(total_sec: int) -> str:
    m, s = divmod(total_sec, 60)
    return f"{m}m {s}s" if m else f"{s}s"


def _yt_url(video_id: str | None) -> str | None:
    return f"https://youtu.be/{video_id}" if video_id else None


def notify_success(summary: dict, pipeline: str, step_timings: list[dict] | None = None) -> None:
    """Post a green success card with key run stats."""
    emoji = _PIPELINE_EMOJI.get(pipeline, "🎬")
    date  = summary.get("date", dt.date.today().isoformat())
    title = summary.get("title", "")

    lines: list[str] = [f"*{emoji} {pipeline.capitalize()} pipeline — {date}*"]

    if title:
        lines.append(f"📺 *{title}*")

    dur_parts: list[str] = []
    if dur := summary.get("total_duration_sec"):
        dur_parts.append(_duration_str(int(dur)))
    if n := summary.get("num_scenes"):
        dur_parts.append(f"{n} scenes")
    if dur_parts:
        lines.append("⏱ " + " · ".join(dur_parts))

    if style := summary.get("thumbnail_style"):
        lines.append(f"🎨 Thumbnail: `{style}`")

    if tool := summary.get("tool"):
        lines.append(f"🛠 Tool: {tool}")

    links: list[str] = []
    if url := _yt_url(summary.get("video_id")):
        links.append(f"<{url}|YouTube>")
    if url := _yt_url(summary.get("short_id")):
        links.append(f"<{url}|Short>")
    if tid := summary.get("tiktok_id"):
        lines.append(f"📱 TikTok: `{tid}`")
    if links:
        lines.append("🔗 " + "  ·  ".join(links))

    if ru := summary.get("ru"):
        ru_status = ru.get("status", "?")
        ru_title  = ru.get("title", "")
        lines.append(f"🇷🇺 RU: {ru_status}" + (f" — {ru_title}" if ru_title else ""))

    if step_timings:
        timing_rows = "\n".join(
            f"  {'⚠️' if row['slow'] else '✅'} `{row['name']:<12}` {row['duration_str']}"
            for row in step_timings
        )
        lines.append(f"⏲ *Step timing:*\n{timing_rows}")

    _post({
        "attachments": [{
            "color": "#2eb886",
            "text": "\n".join(lines),
            "mrkdwn_in": ["text"],
        }]
    })


def notify_slow_steps(slow: list[dict], pipeline: str, date: str) -> None:
    """Post a yellow warning when one or more steps exceeded their latency threshold."""
    if not slow:
        return

    lines: list[str] = [f"*⚠️ {pipeline.capitalize()} pipeline — slow steps detected — {date}*"]
    for entry in slow:
        name     = entry["name"]
        dur      = entry["duration_sec"]
        limit    = entry["threshold_sec"]
        p95      = entry.get("p95_sec")
        label    = "total" if name == "_total" else name
        p95_note = f"  _(P95 baseline: {_duration_str(int(p95))})_" if p95 else ""
        lines.append(
            f"• `{label}` took *{_duration_str(int(dur))}* — threshold {_duration_str(int(limit))}{p95_note}"
        )

    _post({
        "attachments": [{
            "color": "#ffa500",
            "text": "\n".join(lines),
            "mrkdwn_in": ["text"],
        }]
    })


def notify_failure(
    exc: BaseException,
    pipeline: str,
    summary: dict | None = None,
    traceback_str: str | None = None,
) -> None:
    """Post a red failure alert with error message and traceback tail."""
    date = (summary or {}).get("date", dt.date.today().isoformat())
    err  = scrub_message(str(exc))[:400]

    lines: list[str] = [
        f"*🚨 {pipeline.capitalize()} pipeline FAILED — {date}*",
        f"```{err}```",
    ]

    if traceback_str:
        tb_lines = [l for l in scrub_message(traceback_str).strip().splitlines() if l.strip()]
        snippet  = "\n".join(tb_lines[-5:])
        lines.append(f"```{snippet}```")

    _post({
        "attachments": [{
            "color": "#e01e5a",
            "text": "\n".join(lines),
            "mrkdwn_in": ["text"],
        }]
    })
