"""Tests for the custom exception hierarchy in src/exceptions.py."""
from __future__ import annotations

import pytest

from src.exceptions import (
    AIVideoError,
    AudioMissingError,
    AuthenticationError,
    ConfigurationError,
    MediaProcessingError,
    NetworkError,
    QualityCheckError,
    ScriptGenerationError,
    ServiceUnavailableError,
    UploadError,
)

# ── Inheritance tree ──────────────────────────────────────────────────────────

class TestInheritanceTree:
    """Every domain error must be catchable as AIVideoError."""

    def _raises(self, exc_class, *args):
        msg = args[0] if args else "test"
        with pytest.raises(exc_class):
            raise exc_class(msg)

    def test_configuration_error_is_ai_video_error(self):
        with pytest.raises(AIVideoError):
            raise ConfigurationError("missing key")

    def test_script_generation_error_is_ai_video_error(self):
        with pytest.raises(AIVideoError):
            raise ScriptGenerationError("bad json")

    def test_media_processing_error_is_ai_video_error(self):
        with pytest.raises(AIVideoError):
            raise MediaProcessingError("ffmpeg failed")

    def test_audio_missing_error_is_media_processing_error(self):
        with pytest.raises(MediaProcessingError):
            raise AudioMissingError("no audio")

    def test_audio_missing_error_is_ai_video_error(self):
        with pytest.raises(AIVideoError):
            raise AudioMissingError("no audio")

    def test_network_error_is_ai_video_error(self):
        with pytest.raises(AIVideoError):
            raise NetworkError("timeout")

    def test_service_unavailable_is_network_error(self):
        with pytest.raises(NetworkError):
            raise ServiceUnavailableError("circuit open")

    def test_service_unavailable_is_ai_video_error(self):
        with pytest.raises(AIVideoError):
            raise ServiceUnavailableError("circuit open")

    def test_upload_error_is_ai_video_error(self):
        with pytest.raises(AIVideoError):
            raise UploadError("upload failed")

    def test_authentication_error_is_upload_error(self):
        with pytest.raises(UploadError):
            raise AuthenticationError("token expired")

    def test_authentication_error_is_ai_video_error(self):
        with pytest.raises(AIVideoError):
            raise AuthenticationError("token expired")

    def test_quality_check_error_is_ai_video_error(self):
        with pytest.raises(AIVideoError):
            raise QualityCheckError("hook too short")


# ── AIVideoError is NOT Exception subclass hierarchy pollution ────────────────

class TestBaseClass:
    def test_ai_video_error_is_exception(self):
        assert issubclass(AIVideoError, Exception)

    def test_ai_video_error_is_not_runtime_error(self):
        assert not issubclass(AIVideoError, RuntimeError)

    def test_ai_video_error_is_not_value_error(self):
        assert not issubclass(AIVideoError, ValueError)


# ── Error messages preserved ──────────────────────────────────────────────────

class TestErrorMessages:
    def test_configuration_error_message(self):
        exc = ConfigurationError("OPENAI_API_KEY not set")
        assert "OPENAI_API_KEY" in str(exc)

    def test_script_generation_error_message(self):
        exc = ScriptGenerationError("missing key: title")
        assert "title" in str(exc)

    def test_audio_missing_error_message(self):
        exc = AudioMissingError("scene_01.mp3 not produced")
        assert "scene_01" in str(exc)

    def test_service_unavailable_message(self):
        exc = ServiceUnavailableError("OpenAI circuit open")
        assert "OpenAI" in str(exc)


# ── Backward-compatible aliases (QualityGateError, CircuitOpenError) ──────────

class TestBackwardCompatibility:
    def test_quality_gate_error_is_quality_check_error(self):
        from src.quality_gate import QualityGateError
        assert issubclass(QualityGateError, QualityCheckError)

    def test_quality_gate_error_is_ai_video_error(self):
        from src.quality_gate import QualityGateError
        with pytest.raises(AIVideoError):
            raise QualityGateError("hook too short")

    def test_catching_quality_check_catches_quality_gate(self):
        from src.quality_gate import QualityGateError
        with pytest.raises(QualityCheckError):
            raise QualityGateError("scene count < 3")

    def test_circuit_open_error_is_service_unavailable(self):
        from src.circuit_breaker import CircuitOpenError
        assert issubclass(CircuitOpenError, ServiceUnavailableError)

    def test_circuit_open_error_is_ai_video_error(self):
        from src.circuit_breaker import CircuitOpenError
        with pytest.raises(AIVideoError):
            raise CircuitOpenError("openai", retry_after=60.0)

    def test_catching_network_error_catches_circuit_open(self):
        from src.circuit_breaker import CircuitOpenError
        with pytest.raises(NetworkError):
            raise CircuitOpenError("elevenlabs", retry_after=30.0)

    def test_circuit_open_error_exposes_service_name(self):
        from src.circuit_breaker import CircuitOpenError
        exc = CircuitOpenError("youtube", retry_after=120.0)
        assert exc.service == "youtube"
        assert exc.retry_after == 120.0

    def test_script_generation_error_from_validator(self):
        from src.response_validator import parse_llm_json
        with pytest.raises(ScriptGenerationError):
            parse_llm_json(None, context="test")

    def test_validator_error_catchable_as_ai_video_error(self):
        from src.response_validator import validate_llm_response
        with pytest.raises(AIVideoError):
            validate_llm_response({}, ["title"])
