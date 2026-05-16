"""Loguru filter that redacts sensitive values before they reach any sink.

Install via the ``filter=`` parameter on every logger.add() call:

    from src.log_filter import scrub
    logger.add(sys.stderr, level="INFO", filter=scrub, ...)

``scrub_message(text)`` is also exported for use in places that build
strings before logging (e.g. Slack notifications).
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Redaction patterns (most-specific first to avoid double-substitution)
# ---------------------------------------------------------------------------

_RULES: list[tuple[re.Pattern[str], str]] = [
    # OpenAI / Anthropic API keys (sk-... or sk-proj-...)
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "sk-***"),

    # Slack Incoming Webhook — full URL or bare path (/services/T.../B.../secret)
    (re.compile(r"hooks\.slack\.com/services/[A-Za-z0-9/]+"), "hooks.slack.com/services/***"),
    # Bare path as it appears in requests.HTTPError connection pool messages
    (re.compile(r"/services/[A-Z][A-Za-z0-9]{2,}/[A-Z][A-Za-z0-9]{2,}/[A-Za-z0-9]{10,}"), "/services/***"),

    # Bearer tokens in Authorization headers
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}"), "Bearer ***"),

    # URL query params with sensitive key names:
    #   ?api_key=xxx  &key=xxx  &token=xxx  &secret=xxx  &access_token=xxx
    (
        re.compile(r"(?i)([?&](?:api_key|access_key|key|token|secret|access_token|client_secret|client_key)=)[^&\s\"']{6,}"),
        r"\1***",
    ),

    # JSON / dict literals:  "api_key": "xxx"  or  'token': 'xxx'
    (
        re.compile(r"""(?i)(['"](?:api_key|access_key|token|secret|client_secret|client_key)['"])\s*:\s*['"][^'"]{6,}['"]"""),
        r"\1: '***'",
    ),

    # ElevenLabs keys (32-char hex)
    (re.compile(r"\b[0-9a-f]{32}\b"), "***hex32***"),
]


def scrub_message(text: str) -> str:
    """Return *text* with all sensitive patterns replaced."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text


def scrub(record: Any) -> bool:
    """Loguru sink filter — redacts the message in place, always returns True."""
    record["message"] = scrub_message(record["message"])
    return True
