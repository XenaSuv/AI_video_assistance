"""Tests for response_validator — parse_llm_json and validate_llm_response."""
from __future__ import annotations

import json

import pytest

from src.response_validator import parse_llm_json, validate_llm_response
from src.exceptions import ScriptGenerationError, AIVideoError


# ── parse_llm_json ────────────────────────────────────────────────────────────

class TestParseLlmJson:
    def test_parses_plain_json(self):
        result = parse_llm_json('{"title": "GPT-5 launch"}')
        assert result == {"title": "GPT-5 launch"}

    def test_strips_json_code_fence(self):
        content = '```json\n{"title": "test"}\n```'
        result = parse_llm_json(content)
        assert result["title"] == "test"

    def test_strips_plain_code_fence(self):
        content = '```\n{"title": "test"}\n```'
        result = parse_llm_json(content)
        assert result["title"] == "test"

    def test_strips_fence_case_insensitive(self):
        content = '```JSON\n{"title": "test"}\n```'
        result = parse_llm_json(content)
        assert result["title"] == "test"

    def test_raises_on_none_content(self):
        with pytest.raises(ScriptGenerationError, match="empty content"):
            parse_llm_json(None)

    def test_raises_on_empty_string(self):
        with pytest.raises(ScriptGenerationError, match="empty content"):
            parse_llm_json("")

    def test_raises_on_whitespace_only(self):
        with pytest.raises(ScriptGenerationError, match="empty content"):
            parse_llm_json("   ")

    def test_raises_on_invalid_json(self):
        with pytest.raises(ScriptGenerationError, match="invalid JSON"):
            parse_llm_json('{"title": missing_quotes}')

    def test_context_appears_in_error_message(self):
        with pytest.raises(ScriptGenerationError, match="generate_script"):
            parse_llm_json(None, context="generate_script")

    def test_invalid_json_shows_snippet(self):
        with pytest.raises(ScriptGenerationError, match="not-json"):
            parse_llm_json("not-json")

    def test_error_is_ai_video_error_subclass(self):
        with pytest.raises(AIVideoError):
            parse_llm_json(None)

    def test_nested_dict_parsed_correctly(self):
        payload = {"scenes": [{"idx": 0, "heading": "Intro"}], "title": "T"}
        result = parse_llm_json(json.dumps(payload))
        assert result["scenes"][0]["heading"] == "Intro"

    def test_returns_dict_type(self):
        result = parse_llm_json('{"key": "value"}')
        assert isinstance(result, dict)

    def test_strips_leading_trailing_whitespace(self):
        result = parse_llm_json('  \n  {"k": "v"}  \n  ')
        assert result == {"k": "v"}


# ── validate_llm_response ─────────────────────────────────────────────────────

class TestValidateLlmResponse:
    def test_passes_when_all_keys_present(self):
        data = {"title": "T", "scenes": [], "tags": ["ai"]}
        validate_llm_response(data, ["title", "scenes"])  # no raise

    def test_raises_on_missing_key(self):
        with pytest.raises(ScriptGenerationError, match="missing required keys"):
            validate_llm_response({"title": "T"}, ["title", "scenes"])

    def test_raises_listing_all_missing_keys(self):
        with pytest.raises(ScriptGenerationError) as exc_info:
            validate_llm_response({}, ["title", "description", "scenes"])
        msg = str(exc_info.value)
        assert "title" in msg
        assert "description" in msg
        assert "scenes" in msg

    def test_none_value_treated_as_missing(self):
        with pytest.raises(ScriptGenerationError, match="missing required keys"):
            validate_llm_response({"title": None}, ["title"])

    def test_error_is_ai_video_error_subclass(self):
        with pytest.raises(AIVideoError):
            validate_llm_response({}, ["title"])

    def test_empty_string_is_not_missing(self):
        # Empty string is a valid value — only None / absent counts as missing
        validate_llm_response({"title": ""}, ["title"])  # no raise

    def test_zero_is_not_missing(self):
        validate_llm_response({"count": 0}, ["count"])  # no raise

    def test_false_is_not_missing(self):
        validate_llm_response({"enabled": False}, ["enabled"])  # no raise

    def test_context_appears_in_error_message(self):
        with pytest.raises(ScriptGenerationError, match="generate_digest_script"):
            validate_llm_response({}, ["title"], context="generate_digest_script")

    def test_error_shows_present_keys(self):
        with pytest.raises(ScriptGenerationError, match="existing_key"):
            validate_llm_response({"existing_key": "val"}, ["missing_key"])

    def test_empty_required_keys_always_passes(self):
        validate_llm_response({}, [])  # no raise

    def test_passes_with_extra_unexpected_keys(self):
        # Extra keys in data should not cause failure
        data = {"title": "T", "extra": "ok", "scenes": []}
        validate_llm_response(data, ["title"])  # no raise
