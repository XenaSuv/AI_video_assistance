"""Shared constants and mixins used across bandit and A/B testing modules."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from loguru import logger

# hook_aggressiveness adjustment per winning variant type
# conflict  → viewers respond to explicit tension — push harder
# curiosity → open loops work — moderate increase
# simple    → clarity wins — dial back aggression
VARIANT_TYPE_DELTAS: dict[str, float] = {
    "conflict":  +0.10,
    "curiosity": +0.05,
    "simple":    -0.05,
}


# ── Rate-limiting mixin ───────────────────────────────────────────────────────

class _HasSwitchState(Protocol):
    """Protocol for bandit state objects that track last_switch_at."""
    last_switch_at: str | None


class _HasStore(Protocol):
    """Protocol for bandit store objects."""
    def save(self, state: Any) -> None: ...


class RateLimitMixin:
    """Mixin that adds can_switch / record_switch / time_until_switch to any bandit.

    Concrete class must expose:
        self._state           — object with .last_switch_at: str | None
        self._store           — object with .save(state)
        self.min_switch_hours — float
        self._label           — str  (for log messages, e.g. "ThompsonBandit")
    """

    _label: str = "Bandit"

    def can_switch(self) -> bool:
        """True if enough time has elapsed since the last variant/policy switch."""
        if self._state.last_switch_at is None:  # type: ignore[attr-defined]
            return True
        last    = datetime.fromisoformat(self._state.last_switch_at)  # type: ignore[attr-defined]
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        ok: bool = elapsed >= self.min_switch_hours  # type: ignore[attr-defined]
        if not ok:
            remaining = self.min_switch_hours - elapsed  # type: ignore[attr-defined]
            logger.debug(f"{self._label}: rate-limited — {remaining:.1f} h remaining")
        return ok

    def record_switch(self) -> None:
        """Mark now as the last switch time (call after applying a variant/policy)."""
        self._state.last_switch_at = datetime.now(timezone.utc).isoformat()  # type: ignore[attr-defined]
        self._store.save(self._state)  # type: ignore[attr-defined]

    def time_until_switch(self) -> float:
        """Hours remaining until can_switch() returns True (0.0 if ready)."""
        if self._state.last_switch_at is None:  # type: ignore[attr-defined]
            return 0.0
        last    = datetime.fromisoformat(self._state.last_switch_at)  # type: ignore[attr-defined]
        elapsed = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        return float(max(0.0, self.min_switch_hours - elapsed))  # type: ignore[attr-defined]
