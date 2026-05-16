"""Tests for voice_generator — SSML conversion and retry predicate."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Stub elevenlabs before import
_elev_mock = MagicMock()
sys.modules.setdefault("elevenlabs", _elev_mock)
sys.modules.setdefault("elevenlabs.client", _elev_mock)

# Stub tenacity before import
_tenacity = MagicMock()
_tenacity.retry = lambda **kw: (lambda f: f)   # passthrough decorator
_tenacity.retry_if_exception = lambda f: f
_tenacity.stop_after_attempt = lambda n: n
_tenacity.wait_exponential = lambda **kw: None
sys.modules.setdefault("tenacity", _tenacity)

from src.voice_generator import annotated_to_ssml, _is_retryable_elevenlabs


# ── annotated_to_ssml ─────────────────────────────────────────────────────────

class TestAnnotatedToSsml:
    def test_wraps_in_speak_tag(self):
        result = annotated_to_ssml("Hello world")
        assert result.startswith("<speak>")
        assert result.endswith("</speak>")

    def test_pause_short_replaced(self):
        result = annotated_to_ssml("Wait[PAUSE_SHORT]now")
        assert '<break time="300ms"/>' in result
        assert "[PAUSE_SHORT]" not in result

    def test_pause_long_replaced(self):
        result = annotated_to_ssml("Think[PAUSE_LONG]deeply")
        assert '<break time="700ms"/>' in result
        assert "[PAUSE_LONG]" not in result

    def test_emphasis_tags_replaced(self):
        result = annotated_to_ssml("[EMPHASIS]important[/EMPHASIS]")
        assert '<emphasis level="strong">' in result
        assert "</emphasis>" in result
        assert "[EMPHASIS]" not in result
        assert "[/EMPHASIS]" not in result

    def test_multiple_annotations_in_sequence(self):
        text = "[EMPHASIS]AI[/EMPHASIS][PAUSE_SHORT]is here[PAUSE_LONG]"
        result = annotated_to_ssml(text)
        assert '<emphasis level="strong">' in result
        assert '<break time="300ms"/>' in result
        assert '<break time="700ms"/>' in result

    def test_plain_text_unchanged_inside_speak(self):
        result = annotated_to_ssml("No annotations here")
        assert result == "<speak>No annotations here</speak>"

    def test_empty_string(self):
        result = annotated_to_ssml("")
        assert result == "<speak></speak>"


# ── _is_retryable_elevenlabs ──────────────────────────────────────────────────

class TestIsRetryableElevenlabs:
    def test_429_is_retryable(self):
        exc = Exception("rate limit")
        exc.status_code = 429
        assert _is_retryable_elevenlabs(exc)

    def test_500_is_retryable(self):
        exc = Exception("server error")
        exc.status_code = 500
        assert _is_retryable_elevenlabs(exc)

    def test_503_is_retryable(self):
        exc = Exception("service unavailable")
        exc.status_code = 503
        assert _is_retryable_elevenlabs(exc)

    def test_400_is_not_retryable(self):
        exc = Exception("bad request")
        exc.status_code = 400
        assert not _is_retryable_elevenlabs(exc)

    def test_404_is_not_retryable(self):
        exc = Exception("not found")
        exc.status_code = 404
        assert not _is_retryable_elevenlabs(exc)

    def test_connection_error_is_retryable(self):
        assert _is_retryable_elevenlabs(ConnectionError("refused"))

    def test_timeout_error_is_retryable(self):
        assert _is_retryable_elevenlabs(TimeoutError("timed out"))

    def test_os_error_is_retryable(self):
        assert _is_retryable_elevenlabs(OSError("broken pipe"))

    def test_generic_exception_without_status_code_not_retryable(self):
        exc = Exception("unexpected")
        assert not _is_retryable_elevenlabs(exc)
