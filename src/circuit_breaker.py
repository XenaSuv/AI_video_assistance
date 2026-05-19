"""Thread-safe circuit breaker for external API calls.

States:
  CLOSED    — normal; all calls go through and failures are counted
  OPEN      — service appears down; calls fail immediately with CircuitOpenError
              without touching the network
  HALF_OPEN — recovery probe; one call allowed through per window to test
              whether the service has recovered

Typical usage — wrap around an existing tenacity-decorated function so the
circuit breaker sees the failure only after all retry attempts are exhausted:

    from src.circuit_breaker import openai_breaker

    @openai_breaker          # outer: trips after N total tenacity defeats
    @retry(...)              # inner: retries transient errors first
    def _call_dalle(...):
        ...

Per-service singletons (openai_breaker, elevenlabs_breaker, youtube_breaker)
live at the bottom of this module so every import gets the same instance.
"""
from __future__ import annotations

import functools
import threading
import time
from enum import Enum
from typing import Any, Callable

from src.exceptions import ServiceUnavailableError

# ── Public exception ──────────────────────────────────────────────────────────

class CircuitOpenError(ServiceUnavailableError):
    """Raised when a call is rejected because the circuit is OPEN.

    Extends ServiceUnavailableError so callers can catch either name.
    The name CircuitOpenError is kept for backward compatibility.
    """

    def __init__(self, name: str, retry_after: float) -> None:
        self.service = name
        self.retry_after = max(0.0, retry_after)
        super().__init__(
            f"Circuit '{name}' is OPEN — service appears unavailable. "
            f"Retry after {self.retry_after:.0f}s."
        )


# ── State enum ────────────────────────────────────────────────────────────────

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ── Core class ────────────────────────────────────────────────────────────────

class CircuitBreaker:
    """Thread-safe circuit breaker.

    Args:
        name:               Human-readable label used in log messages and errors.
        failure_threshold:  Consecutive failures that trip the circuit (→ OPEN).
        recovery_timeout:   Seconds to wait in OPEN before trying HALF_OPEN.
        half_open_max_calls: Probe calls allowed in HALF_OPEN before closing.
        expected_exceptions: Exception types that count as failures; others
                             propagate but do *not* increment the counter.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 1,
        expected_exceptions: tuple[type[Exception], ...] = (Exception,),
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.expected_exceptions = expected_exceptions

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._effective_state()

    @property
    def failure_count(self) -> int:
        with self._lock:
            return self._failure_count

    # ── Internal state machine ────────────────────────────────────────────────

    def _effective_state(self) -> CircuitState:
        """Return current state, auto-transitioning OPEN → HALF_OPEN on timeout."""
        if (
            self._state is CircuitState.OPEN
            and time.monotonic() - self._last_failure_time >= self.recovery_timeout
        ):
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 0
            self._success_count = 0
        return self._state

    def _record_success(self) -> None:
        state = self._effective_state()
        if state is CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        elif state is CircuitState.CLOSED:
            self._failure_count = 0  # reset streak on any success

    def _record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._success_count = 0

    # ── Call interface ────────────────────────────────────────────────────────

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute *func* unless the circuit is OPEN, then raise CircuitOpenError."""
        with self._lock:
            state = self._effective_state()
            if state is CircuitState.OPEN:
                raise CircuitOpenError(
                    self.name,
                    self.recovery_timeout - (time.monotonic() - self._last_failure_time),
                )
            if state is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitOpenError(self.name, 0.0)
                self._half_open_calls += 1

        try:
            result = func(*args, **kwargs)
        except self.expected_exceptions:
            with self._lock:
                self._record_failure()
            raise
        else:
            with self._lock:
                self._record_success()
            return result

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Use as a decorator: @breaker_instance"""
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper

    def reset(self) -> None:
        """Manually close the circuit (useful in tests and after maintenance)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
            self._last_failure_time = 0.0

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, state={self._state.value}, "
            f"failures={self._failure_count}/{self.failure_threshold})"
        )


# ── Per-service singletons ────────────────────────────────────────────────────
# Import these in the modules that make outbound calls.

openai_breaker = CircuitBreaker(
    name="openai",
    failure_threshold=5,    # open after 5 tenacity-exhausted failures
    recovery_timeout=60.0,  # try again after 60 s
)

elevenlabs_breaker = CircuitBreaker(
    name="elevenlabs",
    failure_threshold=5,
    recovery_timeout=60.0,
)

youtube_breaker = CircuitBreaker(
    name="youtube",
    failure_threshold=3,     # fewer retries; upload failures are expensive
    recovery_timeout=120.0,  # longer pause before retrying YouTube
)
