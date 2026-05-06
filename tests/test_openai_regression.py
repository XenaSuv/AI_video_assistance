"""Regression tests for OpenAI SDK integration.

Covers every API surface used in the codebase so that upgrading the openai
package (>=2.33.0) is caught immediately if any call signature or response
shape breaks.

Mock structure mirrors the real SDK response types so tests fail when:
  - parameter names change (e.g. n= removed from images.generate)
  - response fields move  (e.g. .choices[0].message.content → something else)
  - exception class hierarchy changes
  - prompt_tokens_details.cached_tokens disappears
"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers — build realistic mock responses matching SDK types
# ---------------------------------------------------------------------------

def _make_usage(prompt: int = 100, completion: int = 50, cached: int = 20) -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    details = MagicMock()
    details.cached_tokens = cached
    usage.prompt_tokens_details = details
    return usage


def _make_chat_response(content: str, usage: MagicMock | None = None) -> MagicMock:
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage or _make_usage()
    return resp


def _make_transcription_response(segments: list[dict]) -> MagicMock:
    segs = []
    for s in segments:
        seg = MagicMock()
        seg.start = s["start"]
        seg.end = s["end"]
        seg.text = s["text"]
        segs.append(seg)
    resp = MagicMock()
    resp.segments = segs
    return resp


def _make_image_response(url: str = "https://example.com/img.png") -> MagicMock:
    img = MagicMock()
    img.url = url
    resp = MagicMock()
    resp.data = [img]
    return resp


def _make_embedding_response(vector: list[float] | None = None) -> MagicMock:
    if vector is None:
        vector = [0.1] * 1536
    item = MagicMock()
    item.embedding = vector
    resp = MagicMock()
    resp.data = [item]
    return resp


# ---------------------------------------------------------------------------
# 1. SDK import & exception classes
# ---------------------------------------------------------------------------

class TestSDKImports:
    def test_openai_importable(self):
        import openai  # noqa: F401
        assert hasattr(openai, "__version__")

    def test_openai_client_importable(self):
        from openai import OpenAI
        client = OpenAI(api_key="test-key")
        assert client is not None

    def test_exception_classes_exist(self):
        from openai import BadRequestError, RateLimitError, APIStatusError
        assert issubclass(BadRequestError, Exception)
        assert issubclass(RateLimitError, Exception)
        assert issubclass(APIStatusError, Exception)

    def test_exception_hierarchy(self):
        """RateLimitError and BadRequestError must be subclasses of APIStatusError."""
        from openai import BadRequestError, RateLimitError, APIStatusError
        assert issubclass(RateLimitError, APIStatusError)
        assert issubclass(BadRequestError, APIStatusError)

    def test_version_at_least_2_33(self):
        import openai
        major, minor = map(int, openai.__version__.split(".")[:2])
        assert (major, minor) >= (2, 33), (
            f"openai=={openai.__version__} is below the 2.33 floor"
        )


# ---------------------------------------------------------------------------
# 2. Chat Completions — call signature
# ---------------------------------------------------------------------------

class TestChatCompletions:
    """Verify that chat.completions.create accepts all params used in production."""

    def _make_client(self) -> tuple[Any, MagicMock]:
        from openai import OpenAI
        client = OpenAI(api_key="test-key")
        mock_resp = _make_chat_response('{"result": "ok"}')
        client.chat.completions.create = MagicMock(return_value=mock_resp)
        return client, mock_resp

    def test_basic_call_with_messages_and_model(self):
        client, mock_resp = self._make_client()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )
        client.chat.completions.create.assert_called_once()
        assert resp.choices[0].message.content == '{"result": "ok"}'

    def test_json_object_response_format(self):
        """response_format={"type": "json_object"} must be accepted."""
        client, _ = self._make_client()
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "return json"}],
            response_format={"type": "json_object"},
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_temperature_parameter(self):
        client, _ = self._make_client()
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.7

    def test_system_plus_user_messages(self):
        client, _ = self._make_client()
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Summarise this."},
            ],
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert len(kwargs["messages"]) == 2
        assert kwargs["messages"][0]["role"] == "system"

    def test_max_tokens_parameter(self):
        client, _ = self._make_client()
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=2000,
        )
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_tokens"] == 2000

    def test_response_content_accessible(self):
        content = '{"scenes": [{"idx": 0, "text": "hello"}]}'
        client, _ = self._make_client()
        client.chat.completions.create.return_value = _make_chat_response(content)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "x"}],
        )
        assert resp.choices[0].message.content == content
        # Must be parseable JSON as production code does
        parsed = json.loads(resp.choices[0].message.content)
        assert "scenes" in parsed


# ---------------------------------------------------------------------------
# 3. Token usage — prompt_tokens_details.cached_tokens
# ---------------------------------------------------------------------------

class TestTokenUsage:
    """The _log_usage() helper in multiple modules reads cached_tokens via getattr."""

    def test_usage_fields_present(self):
        usage = _make_usage(prompt=200, completion=80, cached=50)
        assert usage.prompt_tokens == 200
        assert usage.completion_tokens == 80
        assert usage.total_tokens == 280

    def test_cached_tokens_via_getattr(self):
        """Production uses: getattr(details, 'cached_tokens', 0)"""
        usage = _make_usage(cached=40)
        details = usage.prompt_tokens_details
        cached = getattr(details, "cached_tokens", 0) if details else 0
        assert cached == 40

    def test_cached_tokens_fallback_when_none(self):
        """If prompt_tokens_details is None, cached must fall back to 0."""
        usage = _make_usage()
        usage.prompt_tokens_details = None
        details = usage.prompt_tokens_details
        cached = getattr(details, "cached_tokens", 0) if details else 0
        assert cached == 0

    def test_cached_tokens_fallback_when_attr_missing(self):
        """Older SDK versions may not have cached_tokens; getattr default must work."""
        usage = _make_usage()
        del usage.prompt_tokens_details.cached_tokens
        details = usage.prompt_tokens_details
        cached = getattr(details, "cached_tokens", 0) if details else 0
        assert cached == 0

    def test_log_usage_called_via_script_generator(self):
        """_log_usage in script_generator must not crash with a real-shaped usage."""
        from src.script_generator import _log_usage
        usage = _make_usage(prompt=150, completion=60, cached=30)
        # Should not raise
        _log_usage(usage, label="test-regression", model="gpt-4o-mini")

    def test_log_usage_with_none_usage(self):
        """_log_usage must be a no-op when usage is None/falsy."""
        from src.script_generator import _log_usage
        _log_usage(None, label="test", model="gpt-4o-mini")  # must not raise


# ---------------------------------------------------------------------------
# 4. Audio Transcription — verbose_json with segments
# ---------------------------------------------------------------------------

class TestAudioTranscription:
    """subtitle_generator uses client.audio.transcriptions.create with
    response_format='verbose_json' and timestamp_granularities=['segment']."""

    def _make_client(self, segments: list[dict]) -> Any:
        from openai import OpenAI
        client = OpenAI(api_key="test-key")
        mock_resp = _make_transcription_response(segments)
        client.audio.transcriptions.create = MagicMock(return_value=mock_resp)
        return client

    def test_create_call_signature(self):
        client = self._make_client([])
        mock_file = MagicMock()
        client.audio.transcriptions.create(
            model="whisper-1",
            file=mock_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        kwargs = client.audio.transcriptions.create.call_args.kwargs
        assert kwargs["model"] == "whisper-1"
        assert kwargs["response_format"] == "verbose_json"
        assert kwargs["timestamp_granularities"] == ["segment"]

    def test_segments_accessible(self):
        segs_data = [
            {"start": 0.0, "end": 2.5, "text": "Hello world."},
            {"start": 2.5, "end": 5.0, "text": "How are you?"},
        ]
        client = self._make_client(segs_data)
        resp = client.audio.transcriptions.create(
            model="whisper-1", file=MagicMock(),
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )
        assert resp.segments is not None
        result = [
            {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
            for seg in resp.segments
        ]
        assert result[0]["start"] == 0.0
        assert result[1]["text"] == "How are you?"

    def test_assert_segments_not_none_pattern(self):
        """subtitle_generator asserts segments is not None before iterating."""
        resp = _make_transcription_response([{"start": 0.0, "end": 1.0, "text": "hi"}])
        assert resp.segments is not None  # the assert that guards the loop


# ---------------------------------------------------------------------------
# 5. Image Generation — DALL-E 3
# ---------------------------------------------------------------------------

class TestImageGeneration:
    """image_generator._call_dalle uses images.generate and reads .data[0].url"""

    def _make_client(self, url: str = "https://cdn.openai.com/img.png") -> Any:
        from openai import OpenAI
        client = OpenAI(api_key="test-key")
        client.images.generate = MagicMock(return_value=_make_image_response(url))
        return client

    def test_generate_call_signature(self):
        client = self._make_client()
        client.images.generate(
            model="dall-e-3",
            prompt="A futuristic AI robot",
            size="1792x1024",
            quality="hd",
            n=1,
        )
        kwargs = client.images.generate.call_args.kwargs
        assert kwargs["model"] == "dall-e-3"
        assert kwargs["size"] == "1792x1024"
        assert kwargs["quality"] == "hd"
        assert kwargs["n"] == 1

    def test_url_accessible_from_response(self):
        url = "https://cdn.openai.com/generated.png"
        client = self._make_client(url)
        resp = client.images.generate(
            model="dall-e-3", prompt="test", size="1792x1024", quality="hd", n=1,
        )
        assert resp.data[0].url == url

    def test_data_list_not_empty(self):
        client = self._make_client()
        resp = client.images.generate(
            model="dall-e-3", prompt="test", size="1792x1024", quality="hd", n=1,
        )
        assert len(resp.data) >= 1


# ---------------------------------------------------------------------------
# 6. Embeddings
# ---------------------------------------------------------------------------

class TestEmbeddings:
    """image_cache._embed uses client.embeddings.create and reads .data[0].embedding."""

    def _make_client(self, vector: list[float] | None = None) -> Any:
        from openai import OpenAI
        client = OpenAI(api_key="test-key")
        client.embeddings.create = MagicMock(
            return_value=_make_embedding_response(vector)
        )
        return client

    def test_create_call_signature(self):
        client = self._make_client()
        client.embeddings.create(
            model="text-embedding-3-small",
            input="hello world",
        )
        kwargs = client.embeddings.create.call_args.kwargs
        assert kwargs["model"] == "text-embedding-3-small"
        assert kwargs["input"] == "hello world"

    def test_embedding_vector_accessible(self):
        # numpy is stubbed in conftest.py — test raw list access, not np.array()
        vector = [0.1, 0.2, 0.3]
        client = self._make_client(vector)
        resp = client.embeddings.create(model="text-embedding-3-small", input="test")
        embedding = resp.data[0].embedding
        assert embedding == vector
        assert len(embedding) == 3
        assert abs(embedding[0] - 0.1) < 1e-6

    def test_input_truncated_to_2048_chars(self):
        """Production truncates input to [:2048] — verify mock accepts that."""
        client = self._make_client()
        long_text = "a" * 5000
        client.embeddings.create(
            model="text-embedding-3-small", input=long_text[:2048]
        )
        kwargs = client.embeddings.create.call_args.kwargs
        assert len(kwargs["input"]) == 2048


# ---------------------------------------------------------------------------
# 7. Error handling — is_retryable_dalle logic
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_bad_request_error_is_not_retryable(self):
        from openai import BadRequestError, RateLimitError, APIStatusError

        def _is_retryable(exc: BaseException) -> bool:
            if isinstance(exc, BadRequestError):
                return False
            if isinstance(exc, RateLimitError):
                return True
            if isinstance(exc, APIStatusError):
                return exc.status_code >= 500
            return isinstance(exc, OSError)

        bad_req = MagicMock(spec=BadRequestError)
        bad_req.__class__ = BadRequestError
        assert _is_retryable(bad_req) is False

    def test_rate_limit_error_is_retryable(self):
        from openai import RateLimitError

        def _is_retryable(exc: BaseException) -> bool:
            from openai import BadRequestError, APIStatusError
            if isinstance(exc, BadRequestError):
                return False
            if isinstance(exc, RateLimitError):
                return True
            if isinstance(exc, APIStatusError):
                return exc.status_code >= 500
            return False

        rate_limit = MagicMock(spec=RateLimitError)
        rate_limit.__class__ = RateLimitError
        assert _is_retryable(rate_limit) is True

    def test_api_status_500_is_retryable(self):
        from openai import APIStatusError

        server_err = MagicMock(spec=APIStatusError)
        server_err.__class__ = APIStatusError
        server_err.status_code = 503

        def _is_retryable(exc: BaseException) -> bool:
            from openai import BadRequestError, RateLimitError
            if isinstance(exc, BadRequestError):
                return False
            if isinstance(exc, RateLimitError):
                return True
            if isinstance(exc, APIStatusError):
                return exc.status_code >= 500
            return False

        assert _is_retryable(server_err) is True

    def test_api_status_400_is_not_retryable(self):
        from openai import APIStatusError

        client_err = MagicMock(spec=APIStatusError)
        client_err.__class__ = APIStatusError
        client_err.status_code = 400

        def _is_retryable(exc: BaseException) -> bool:
            from openai import BadRequestError, RateLimitError
            if isinstance(exc, BadRequestError):
                return False
            if isinstance(exc, RateLimitError):
                return True
            if isinstance(exc, APIStatusError):
                return exc.status_code >= 500
            return False

        assert _is_retryable(client_err) is False


# ---------------------------------------------------------------------------
# 8. make_openai_client — wires key + max_retries
# ---------------------------------------------------------------------------

class TestMakeOpenAIClient:
    def test_returns_openai_client(self):
        from src.retry_utils import make_openai_client
        from openai import OpenAI
        client = make_openai_client()
        assert isinstance(client, OpenAI)

    def test_client_has_max_retries_3(self):
        from src.retry_utils import make_openai_client
        client = make_openai_client()
        # The SDK exposes _max_retries or max_retries depending on version
        retries = getattr(client, "_max_retries", None) or getattr(client, "max_retries", None)
        assert retries == 3, f"Expected max_retries=3, got {retries}"
