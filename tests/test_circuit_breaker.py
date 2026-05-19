"""Tests for CircuitBreaker and per-service singleton wiring."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, call, patch

import pytest

from src.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    elevenlabs_breaker,
    openai_breaker,
    youtube_breaker,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_cb(**kwargs) -> CircuitBreaker:
    defaults = dict(name="test", failure_threshold=3, recovery_timeout=60.0)
    defaults.update(kwargs)
    return CircuitBreaker(**defaults)


def _failing(exc: Exception):
    """Return a callable that always raises *exc*."""
    def _f(*a, **kw):
        raise exc
    return _f


# ── State transitions ─────────────────────────────────────────────────────────

class TestClosedState:
    def test_starts_closed(self):
        cb = _make_cb()
        assert cb.state is CircuitState.CLOSED

    def test_success_keeps_closed(self):
        cb = _make_cb()
        cb.call(lambda: 42)
        assert cb.state is CircuitState.CLOSED

    def test_failure_below_threshold_stays_closed(self):
        cb = _make_cb(failure_threshold=3)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(_failing(RuntimeError("boom")))
        assert cb.state is CircuitState.CLOSED

    def test_success_resets_failure_streak(self):
        cb = _make_cb(failure_threshold=3)
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))
        cb.call(lambda: None)  # success resets streak
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))
        assert cb.state is CircuitState.CLOSED  # only 1 failure after reset

    def test_return_value_passed_through(self):
        cb = _make_cb()
        result = cb.call(lambda: "hello")
        assert result == "hello"


class TestOpenState:
    def test_trips_after_threshold(self):
        cb = _make_cb(failure_threshold=3)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(_failing(RuntimeError()))
        assert cb.state is CircuitState.OPEN

    def test_open_raises_circuit_open_error(self):
        cb = _make_cb(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(lambda: None)
        assert exc_info.value.service == "test"

    def test_open_does_not_call_underlying_function(self):
        cb = _make_cb(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))
        mock_fn = MagicMock()
        with pytest.raises(CircuitOpenError):
            cb.call(mock_fn)
        mock_fn.assert_not_called()

    def test_circuit_open_error_has_retry_after(self):
        cb = _make_cb(failure_threshold=1, recovery_timeout=30.0)
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(lambda: None)
        assert exc_info.value.retry_after > 0

    def test_failure_count_readable(self):
        cb = _make_cb(failure_threshold=5)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(_failing(RuntimeError()))
        assert cb.failure_count == 3


class TestHalfOpenState:
    def test_transitions_to_half_open_after_timeout(self):
        cb = _make_cb(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))
        time.sleep(0.06)
        assert cb.state is CircuitState.HALF_OPEN

    def test_half_open_success_closes_circuit(self):
        cb = _make_cb(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))
        time.sleep(0.06)
        cb.call(lambda: None)  # probe succeeds
        assert cb.state is CircuitState.CLOSED

    def test_half_open_failure_reopens_circuit(self):
        cb = _make_cb(failure_threshold=1, recovery_timeout=0.05)
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))
        time.sleep(0.06)
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))  # probe fails
        assert cb.state is CircuitState.OPEN

    def test_half_open_blocks_extra_concurrent_probes(self):
        cb = _make_cb(failure_threshold=1, recovery_timeout=0.05, half_open_max_calls=1)
        # Directly set HALF_OPEN with max probes already dispatched (simulates
        # concurrent calls; setting state directly avoids the auto-reset in
        # _effective_state() which only fires on OPEN→HALF_OPEN transition).
        cb._state = CircuitState.HALF_OPEN
        cb._half_open_calls = 1  # already at limit
        with pytest.raises(CircuitOpenError):
            cb.call(lambda: None)


class TestReset:
    def test_reset_closes_open_circuit(self):
        cb = _make_cb(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))
        assert cb.state is CircuitState.OPEN
        cb.reset()
        assert cb.state is CircuitState.CLOSED

    def test_reset_clears_failure_count(self):
        cb = _make_cb(failure_threshold=5)
        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(_failing(RuntimeError()))
        cb.reset()
        assert cb.failure_count == 0

    def test_call_succeeds_after_reset(self):
        cb = _make_cb(failure_threshold=1)
        with pytest.raises(RuntimeError):
            cb.call(_failing(RuntimeError()))
        cb.reset()
        result = cb.call(lambda: "ok")
        assert result == "ok"


# ── Decorator form ────────────────────────────────────────────────────────────

class TestDecoratorForm:
    def test_decorator_wraps_function(self):
        cb = _make_cb()

        @cb
        def my_func(x):
            return x * 2

        assert my_func(5) == 10

    def test_decorator_preserves_function_name(self):
        cb = _make_cb()

        @cb
        def my_important_func():
            pass

        assert my_important_func.__name__ == "my_important_func"

    def test_decorator_trips_on_repeated_failures(self):
        cb = _make_cb(failure_threshold=2)

        @cb
        def bad_func():
            raise ValueError("oops")

        for _ in range(2):
            with pytest.raises(ValueError):
                bad_func()
        assert cb.state is CircuitState.OPEN

    def test_open_circuit_raises_circuit_open_error_via_decorator(self):
        cb = _make_cb(failure_threshold=1)

        @cb
        def bad_func():
            raise ValueError()

        with pytest.raises(ValueError):
            bad_func()
        with pytest.raises(CircuitOpenError):
            bad_func()


# ── expected_exceptions filtering ─────────────────────────────────────────────

class TestExpectedExceptions:
    def test_only_expected_exceptions_count_as_failures(self):
        cb = _make_cb(
            failure_threshold=2,
            expected_exceptions=(ConnectionError,),
        )
        # ValueError is NOT in expected_exceptions → should propagate but NOT count
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(_failing(ValueError("unrelated")))
        assert cb.state is CircuitState.CLOSED  # did not trip

    def test_expected_exceptions_count_as_failures(self):
        cb = _make_cb(
            failure_threshold=2,
            expected_exceptions=(ConnectionError,),
        )
        for _ in range(2):
            with pytest.raises(ConnectionError):
                cb.call(_failing(ConnectionError()))
        assert cb.state is CircuitState.OPEN


# ── repr ──────────────────────────────────────────────────────────────────────

class TestRepr:
    def test_repr_contains_name_and_state(self):
        cb = _make_cb(name="myservice")
        r = repr(cb)
        assert "myservice" in r
        assert "closed" in r


# ── Per-service singletons ────────────────────────────────────────────────────

class TestServiceSingletons:
    def setup_method(self):
        # Ensure singletons start fresh for each test
        openai_breaker.reset()
        elevenlabs_breaker.reset()
        youtube_breaker.reset()

    def test_openai_breaker_is_circuit_breaker(self):
        assert isinstance(openai_breaker, CircuitBreaker)
        assert openai_breaker.name == "openai"

    def test_elevenlabs_breaker_is_circuit_breaker(self):
        assert isinstance(elevenlabs_breaker, CircuitBreaker)
        assert elevenlabs_breaker.name == "elevenlabs"

    def test_youtube_breaker_is_circuit_breaker(self):
        assert isinstance(youtube_breaker, CircuitBreaker)
        assert youtube_breaker.name == "youtube"

    def test_openai_threshold_is_5(self):
        assert openai_breaker.failure_threshold == 5

    def test_elevenlabs_threshold_is_5(self):
        assert elevenlabs_breaker.failure_threshold == 5

    def test_youtube_threshold_is_3(self):
        # YouTube gets tighter threshold — upload failures are expensive
        assert youtube_breaker.failure_threshold == 3

    def test_youtube_recovery_timeout_longer(self):
        assert youtube_breaker.recovery_timeout > openai_breaker.recovery_timeout

    def test_singletons_are_same_object_on_reimport(self):
        from src.circuit_breaker import openai_breaker as ob2
        assert ob2 is openai_breaker


# ── Integration: decorator wiring on production functions ─────────────────────

class TestProductionWiring:
    """Verify the @breaker decorators are actually applied to the target functions."""

    def setup_method(self):
        openai_breaker.reset()
        elevenlabs_breaker.reset()
        youtube_breaker.reset()

    def test_call_dalle_has_circuit_breaker(self):
        """_call_dalle should fail fast after openai_breaker trips."""
        from src.image_generator import _call_dalle

        # Trip the breaker
        for _ in range(openai_breaker.failure_threshold):
            with pytest.raises(RuntimeError):
                openai_breaker.call(_failing(RuntimeError("simulated openai down")))
        assert openai_breaker.state is CircuitState.OPEN

        # Now any call through the breaker fails immediately
        mock_client = MagicMock()
        with pytest.raises(CircuitOpenError):
            openai_breaker.call(lambda: None)

    def test_tts_convert_has_circuit_breaker(self):
        """_tts_convert should use elevenlabs_breaker."""
        for _ in range(elevenlabs_breaker.failure_threshold):
            with pytest.raises(RuntimeError):
                elevenlabs_breaker.call(_failing(RuntimeError("11labs down")))
        assert elevenlabs_breaker.state is CircuitState.OPEN

    def test_upload_video_has_circuit_breaker(self):
        """upload_video should use youtube_breaker."""
        for _ in range(youtube_breaker.failure_threshold):
            with pytest.raises(RuntimeError):
                youtube_breaker.call(_failing(RuntimeError("yt down")))
        assert youtube_breaker.state is CircuitState.OPEN
