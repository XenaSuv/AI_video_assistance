"""Tests for src/log_filter.py — PII redaction before logs reach any sink."""
from __future__ import annotations

import pytest

from src.log_filter import scrub, scrub_message


# ── scrub_message: individual pattern coverage ────────────────────────────────

class TestScrubMessage:

    # OpenAI keys
    def test_redacts_openai_sk_key(self):
        msg = "OpenAI error: api_key=sk-abcdefghijklmnopqrstuvwxyz1234567890"
        result = scrub_message(msg)
        assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in result
        assert "sk-***" in result

    def test_redacts_openai_proj_key(self):
        msg = "Using key sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        result = scrub_message(msg)
        assert "sk-proj-" not in result or "***" in result

    # Slack webhook
    def test_redacts_slack_webhook_url(self):
        msg = "POST https://hooks.slack.com/services/T012AB3CD/B456EF7GH/abcXYZ123secret failed"
        result = scrub_message(msg)
        assert "T012AB3CD/B456EF7GH/abcXYZ123secret" not in result
        assert "***" in result

    # Bearer tokens
    def test_redacts_bearer_token(self):
        msg = "Authorization: Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.signature"
        result = scrub_message(msg)
        assert "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "Bearer ***" in result

    def test_bearer_case_insensitive(self):
        msg = "bearer ABCDEFGHIJKLMNOPQRSTUVWXYZ12345"
        result = scrub_message(msg)
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ12345" not in result

    # URL query params
    def test_redacts_api_key_query_param(self):
        msg = "GET https://pixabay.com/api/videos/?key=secretpixabaykey1234&q=AI"
        result = scrub_message(msg)
        assert "secretpixabaykey1234" not in result
        assert "***" in result

    def test_redacts_token_query_param(self):
        msg = "Request to https://api.example.com/data?token=supersecrettoken999&page=1"
        result = scrub_message(msg)
        assert "supersecrettoken999" not in result

    def test_redacts_secret_query_param(self):
        msg = "https://api.service.com/auth?client_secret=mysecretvalue123456&grant_type=code"
        result = scrub_message(msg)
        assert "mysecretvalue123456" not in result

    def test_redacts_access_token_query_param(self):
        msg = "Refreshing: https://auth.service.com/token?access_token=abcdef123456789xyz"
        result = scrub_message(msg)
        assert "abcdef123456789xyz" not in result

    # JSON / dict literals
    def test_redacts_json_api_key(self):
        msg = """Failed to parse {"api_key": "my-very-secret-api-key-here", "model": "gpt-4"}"""
        result = scrub_message(msg)
        assert "my-very-secret-api-key-here" not in result

    def test_redacts_json_token(self):
        msg = """Response: {"token": "sometoken1234567890abcdef", "expires": 3600}"""
        result = scrub_message(msg)
        assert "sometoken1234567890abcdef" not in result

    # 32-char hex (ElevenLabs keys)
    def test_redacts_32char_hex(self):
        msg = "ElevenLabs key: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4"
        result = scrub_message(msg)
        assert "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4" not in result
        assert "***hex32***" in result

    # Safe values preserved
    def test_does_not_redact_normal_text(self):
        msg = "Pipeline completed successfully in 42 seconds"
        assert scrub_message(msg) == msg

    def test_does_not_redact_short_values(self):
        # Short tokens (< 6 chars) are not redacted — avoids false positives
        msg = "?key=short"
        result = scrub_message(msg)
        assert result == msg

    def test_does_not_redact_video_id(self):
        msg = "Uploaded: https://youtu.be/dQw4w9WgXcQ"
        result = scrub_message(msg)
        assert "https://youtu.be/dQw4w9WgXcQ" in result

    def test_empty_string(self):
        assert scrub_message("") == ""


# ── scrub: loguru filter interface ───────────────────────────────────────────

class TestScrubFilter:
    def _record(self, message: str) -> dict:
        return {"message": message}

    def test_returns_true_always(self):
        record = self._record("normal message")
        assert scrub(record) is True

    def test_modifies_record_in_place(self):
        record = self._record("key sk-abcdefghijklmnopqrstuvwxyz1234567890 used")
        scrub(record)
        assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in record["message"]
        assert "sk-***" in record["message"]

    def test_clean_record_unchanged(self):
        record = self._record("Scene 3: audio generated in 1.2s")
        scrub(record)
        assert record["message"] == "Scene 3: audio generated in 1.2s"

    def test_multiple_secrets_in_one_message(self):
        msg = (
            "Failed: Bearer ABCDEFGHIJKLMNOPQRSTUVWXYZ12345 "
            "at https://hooks.slack.com/services/T000/B000/abcXYZsecret123"
        )
        record = self._record(msg)
        scrub(record)
        result = record["message"]
        assert "ABCDEFGHIJKLMNOPQRSTUVWXYZ12345" not in result
        assert "abcXYZsecret123" not in result


# ── real-world exception message patterns ─────────────────────────────────────

class TestRealWorldPatterns:
    """Patterns that actually appear in production exception messages."""

    def test_requests_httperror_with_pixabay_key(self):
        exc_msg = (
            "400 Client Error: Bad Request for url: "
            "https://pixabay.com/api/videos/?key=1234567890abcdef1234&q=AI+robot"
        )
        result = scrub_message(exc_msg)
        assert "1234567890abcdef1234" not in result
        assert "pixabay.com" in result  # domain preserved

    def test_slack_webhook_in_connection_error(self):
        exc_msg = (
            "HTTPSConnectionPool(host='hooks.slack.com', port=443): "
            "Max retries exceeded with url: /services/TABC123/BDEF456/xyzSecretTokenHere"
        )
        result = scrub_message(exc_msg)
        assert "xyzSecretTokenHere" not in result

    def test_openai_key_in_auth_error(self):
        exc_msg = "AuthenticationError: Invalid API key: sk-real-openai-key-here-abcdefghij"
        result = scrub_message(exc_msg)
        assert "sk-real-openai-key-here-abcdefghij" not in result
        assert "AuthenticationError" in result  # context preserved
