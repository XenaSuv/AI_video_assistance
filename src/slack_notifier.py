"""Send pipeline status notifications to a Slack channel via Incoming Webhook.

Set SLACK_WEBHOOK_URL in your environment (or .env) to enable.
If the variable is absent or the webhook call fails, notifications are
silently skipped — the pipeline is never blocked by Slack being down.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any, Optional

from loguru import logger

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


def _post(payload: dict[str, Any]) -> None:
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


def notify_success(summary: dict[str, Any], pipeline: str, step_timings: list[dict[str, Any]] | None = None) -> None:
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


def notify_budget_alert(status: "Any", pipeline: str) -> None:
    """Post a red alert when daily or monthly API budget is exceeded."""
    from src.budget_guard import BudgetStatus  # local import to avoid circular
    assert isinstance(status, BudgetStatus)

    lines: list[str] = [
        f"*🚨 Budget limit exceeded — {pipeline.capitalize()} — {status.date}*"
    ]

    if status.day_exceeded:
        pct = f" ({status.day_pct}%)" if status.day_pct is not None else ""
        lines.append(
            f"• Today: *${status.day_usd:.2f}* of ${status.day_limit:.0f} limit{pct}"
        )
    if status.month_exceeded:
        pct = f" ({status.month_pct}%)" if status.month_pct is not None else ""
        lines.append(
            f"• This month: *${status.month_usd:.2f}* of ${status.month_limit:.0f} limit{pct}"
        )
    lines.append("_Review API usage and adjust thresholds via DAILY_BUDGET_USD / MONTHLY_BUDGET_USD._")

    _post({
        "attachments": [{
            "color": "#e01e5a",
            "text": "\n".join(lines),
            "mrkdwn_in": ["text"],
        }]
    })


def notify_budget_summary(status: "Any", pipeline: str) -> None:
    """Post a daily spend summary to Slack (informational, always sent)."""
    from src.budget_guard import BudgetStatus
    assert isinstance(status, BudgetStatus)

    day_bar   = _spend_bar(status.day_usd,   status.day_limit)
    month_bar = _spend_bar(status.month_usd, status.month_limit)

    day_pct   = f" {status.day_pct}%"   if status.day_pct   is not None else ""
    month_pct = f" {status.month_pct}%" if status.month_pct is not None else ""

    lines = [
        f"*💰 API spend — {pipeline.capitalize()} — {status.date}*",
        f"• Today:  ${status.day_usd:.2f} / ${status.day_limit:.0f}{day_pct}   {day_bar}",
        f"• Month:  ${status.month_usd:.2f} / ${status.month_limit:.0f}{month_pct}   {month_bar}",
    ]
    color = "#e01e5a" if status.any_exceeded else "#2eb886"
    _post({
        "attachments": [{
            "color": color,
            "text": "\n".join(lines),
            "mrkdwn_in": ["text"],
        }]
    })


def _spend_bar(spent: float, limit: float, width: int = 10) -> str:
    """Return a simple ASCII progress bar: ████░░░░░░ 42%"""
    if limit <= 0:
        return ""
    pct = min(spent / limit, 1.0)
    filled = round(pct * width)
    return "█" * filled + "░" * (width - filled)


def notify_slow_steps(slow: list[dict[str, Any]], pipeline: str, date: str) -> None:
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


def notify_quota_alert(used: int, limit: int, pipeline: str) -> None:
    """Post a warning when YouTube API quota is nearly or fully exhausted."""
    pct   = int(used / limit * 100) if limit else 0
    full  = used >= limit
    color = "#e01e5a" if full else "#ffa500"
    label = "exhausted" if full else "nearly exhausted"

    lines: list[str] = [
        f"*⚠️ YouTube quota {label} — {pipeline.capitalize()} — {dt.date.today().isoformat()}*",
        f"• Used: *{used:,}* / {limit:,} units ({pct}%)",
        "_Uploads will be skipped until midnight UTC when the quota resets._",
    ]

    _post({
        "attachments": [{
            "color": color,
            "text": "\n".join(lines),
            "mrkdwn_in": ["text"],
        }]
    })


def notify_watchdog_alert(
    last_run_at: Optional[dt.datetime],
    hours_since: float,
    pipeline: str,
) -> None:
    """Post a red alert when the pipeline hasn't produced a successful run recently."""
    if last_run_at is None:
        desc = "No successful run found in the output directory."
    else:
        h = int(hours_since)
        last_str = last_run_at.strftime("%Y-%m-%d %H:%M UTC")
        desc = f"Last success: *{last_str}* ({h}h ago)"

    lines: list[str] = [
        f"*🚨 Watchdog: {pipeline.capitalize()} pipeline stalled — {dt.date.today().isoformat()}*",
        desc,
        "_Check GitHub Actions logs and re-trigger manually if needed._",
    ]

    _post({
        "attachments": [{
            "color": "#e01e5a",
            "text": "\n".join(lines),
            "mrkdwn_in": ["text"],
        }]
    })


def notify_heartbeat(statuses: list[Any]) -> None:
    """Post a consolidated health status board for all pipelines.

    Each element of *statuses* must expose:  .pipeline, .last_run_at,
    .hours_since, .threshold_hours, .is_stale, .status_emoji
    (duck-typed so watchdog.PipelineHealth or any compatible object works).
    """
    now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [f"*🏥 Pipeline Health Report — {now_str}*", ""]

    for s in statuses:
        if s.last_run_at is None:
            last_str = "no run found"
        else:
            h = s.hours_since
            ts = s.last_run_at.strftime("%Y-%m-%d %H:%M UTC")
            last_str = f"{h:.1f}h ago  ({ts})"
        lines.append(f"{s.status_emoji}  `{s.pipeline:<8}`  {last_str}")

    any_stale = any(s.is_stale for s in statuses)
    color = "#e01e5a" if any_stale else "#2eb886"
    _post({
        "attachments": [{
            "color":    color,
            "text":     "\n".join(lines),
            "mrkdwn_in": ["text"],
        }]
    })


def notify_failure(
    exc: BaseException,
    pipeline: str,
    summary: dict[str, Any] | None = None,
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
