"""Validation helpers for LLM (OpenAI) JSON responses.

Two entry points:
  parse_llm_json(content, context)           — parse JSON, strip markdown fences
  validate_llm_response(data, keys, context) — assert required keys present

Both raise ScriptGenerationError with a human-readable message that names the
caller context and the missing/malformed fields, replacing the cryptic KeyError
/ JSONDecodeError that would otherwise surface from deep inside a pipeline step.
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.exceptions import ScriptGenerationError


def parse_llm_json(content: str | None, context: str = "") -> dict[str, Any]:
    """Parse JSON from an LLM response string.

    Handles the two common formatting quirks from OpenAI:
    - Empty / None response (raises ValueError)
    - Markdown code fences  (```json ... ```) wrapping the payload

    Args:
        content: the raw string from ``response.choices[0].message.content``.
        context: a short label shown in error messages (e.g. ``"generate_script"``).

    Returns:
        Parsed dict.

    Raises:
        ValueError: content is empty, or not valid JSON after fence stripping.
    """
    ctx = f" [{context}]" if context else ""
    if not content or not content.strip():
        raise ScriptGenerationError(f"LLM returned empty content{ctx}")

    text = content.strip()
    # Strip optional opening fence: ```json or ```
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    # Strip optional closing fence
    text = re.sub(r"```\s*$", "", text).strip()

    try:
        result: dict[str, Any] = json.loads(text)
        return result
    except json.JSONDecodeError as exc:
        snippet = text[:200] + ("…" if len(text) > 200 else "")
        raise ScriptGenerationError(
            f"LLM returned invalid JSON{ctx}: {exc}. "
            f"Content snippet: {snippet!r}"
        ) from exc


def validate_llm_response(
    data: dict[str, Any],
    required_keys: list[str],
    context: str = "",
) -> None:
    """Assert that *data* contains every key in *required_keys*.

    Args:
        data:          Parsed LLM response dict.
        required_keys: Keys that must be present (non-None).
        context:       Short label for error messages.

    Raises:
        ValueError: one or more required keys are absent.
    """
    ctx = f" [{context}]" if context else ""
    missing = [k for k in required_keys if k not in data or data[k] is None]
    if missing:
        raise ScriptGenerationError(
            f"LLM response{ctx} missing required keys: {missing}. "
            f"Present keys: {sorted(data.keys())}"
        )
