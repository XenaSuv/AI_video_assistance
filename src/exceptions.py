"""Domain exception hierarchy for the AI video pipeline.

Hierarchy
---------
AIVideoError                        ← base; catch this to handle any pipeline error
├── ConfigurationError              ← missing / invalid env vars or settings
├── ScriptGenerationError           ← LLM returned bad, empty, or malformed content
├── MediaProcessingError            ← ffmpeg / audio / video file failures
│   └── AudioMissingError           ← TTS produced no audio file
├── NetworkError                    ← external API call failed (non-auth)
│   └── ServiceUnavailableError     ← circuit open, rate-limited, temporary outage
├── UploadError                     ← YouTube / TikTok publish failed
│   └── AuthenticationError         ← OAuth token expired or invalid
└── QualityCheckError               ← video or script failed quality gate

Backward-compatible aliases
---------------------------
QualityGateError    → QualityCheckError  (imported from quality_gate still works)
CircuitOpenError    → ServiceUnavailableError (imported from circuit_breaker still works)
"""
from __future__ import annotations


class AIVideoError(Exception):
    """Base class for all pipeline-specific exceptions.

    Catch this to handle any domain error without catching system errors
    (MemoryError, KeyboardInterrupt, etc.).
    """


# ── Configuration ─────────────────────────────────────────────────────────────

class ConfigurationError(AIVideoError):
    """A required setting or credential is missing or invalid.

    Examples: OPENAI_API_KEY not set, YouTube client_secrets.json missing.
    """


# ── Script / LLM ──────────────────────────────────────────────────────────────

class ScriptGenerationError(AIVideoError):
    """The LLM returned content that cannot produce a valid VideoScript.

    Examples: empty response, missing required JSON keys, parse failure.
    """


# ── Media processing ──────────────────────────────────────────────────────────

class MediaProcessingError(AIVideoError):
    """An ffmpeg operation or media file is broken or missing."""


class AudioMissingError(MediaProcessingError):
    """TTS synthesis completed but produced no usable audio file."""


# ── Network / external services ───────────────────────────────────────────────

class NetworkError(AIVideoError):
    """An external API call failed for a non-authentication reason."""


class ServiceUnavailableError(NetworkError):
    """The target service is temporarily unavailable.

    Raised by the circuit breaker when the failure threshold is exceeded,
    and by callers that detect rate-limiting or gateway errors.
    """


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadError(AIVideoError):
    """Publishing a video to YouTube or TikTok failed."""


class AuthenticationError(UploadError):
    """OAuth token has expired or credentials are invalid.

    Distinct from NetworkError so callers can prompt for re-authentication
    rather than retrying the same request.
    """


# ── Quality gate ──────────────────────────────────────────────────────────────

class QualityCheckError(AIVideoError):
    """The video or script failed one or more quality gate checks.

    Hard failures block publishing; the pipeline is expected to re-raise
    this exception rather than swallowing it.
    """
