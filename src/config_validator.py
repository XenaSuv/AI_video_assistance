"""Validate Settings at startup — collects all errors before raising.

Usage (already wired in config/settings.py):
    from src.config_validator import validate
    validate(settings)
"""
from __future__ import annotations

from typing import Any


class ConfigurationError(ValueError):
    """Raised when one or more configuration values are invalid.

    The message lists every problem found, not just the first.
    """


_YOUTUBE_PRIVACY = frozenset({"public", "unlisted", "private"})
_TIKTOK_PRIVACY  = frozenset({"PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"})
_HEYGEN_ASPECT   = frozenset({"16:9", "9:16"})


def validate(cfg: Any) -> None:
    """Check all settings fields.

    Raises ConfigurationError listing every problem found.
    Does nothing when the configuration is valid.
    """
    errors: list[str] = []

    # ── Numeric range checks ──────────────────────────────────────────────────

    if not (0 <= cfg.daily_run_hour_utc <= 23):
        errors.append(
            f"DAILY_RUN_HOUR_UTC must be 0–23, got {cfg.daily_run_hour_utc}"
        )
    if not (0 <= cfg.topic_run_hour_utc <= 23):
        errors.append(
            f"TOPIC_RUN_HOUR_UTC must be 0–23, got {cfg.topic_run_hour_utc}"
        )
    if not (0.0 <= cfg.background_music_volume <= 1.0):
        errors.append(
            f"BACKGROUND_MUSIC_VOLUME must be 0.0–1.0, "
            f"got {cfg.background_music_volume}"
        )
    if not (0.0 <= cfg.shorts_music_volume <= 1.0):
        errors.append(
            f"SHORTS_MUSIC_VOLUME must be 0.0–1.0, got {cfg.shorts_music_volume}"
        )
    if cfg.script_target_words <= 0:
        errors.append(
            f"SCRIPT_TARGET_WORDS must be > 0, got {cfg.script_target_words}"
        )
    if cfg.dedup_ttl_days <= 0:
        errors.append(
            f"DEDUP_TTL_DAYS must be > 0, got {cfg.dedup_ttl_days}"
        )
    if cfg.daily_budget_usd < 0:
        errors.append(
            f"DAILY_BUDGET_USD must be >= 0, got {cfg.daily_budget_usd}"
        )
    if cfg.monthly_budget_usd < 0:
        errors.append(
            f"MONTHLY_BUDGET_USD must be >= 0, got {cfg.monthly_budget_usd}"
        )
    if cfg.end_card_duration < 0:
        errors.append(
            f"END_CARD_DURATION must be >= 0, got {cfg.end_card_duration}"
        )

    # ── Choice checks ─────────────────────────────────────────────────────────

    if cfg.youtube_privacy not in _YOUTUBE_PRIVACY:
        errors.append(
            f"YOUTUBE_PRIVACY must be one of {sorted(_YOUTUBE_PRIVACY)}, "
            f"got {cfg.youtube_privacy!r}"
        )
    if cfg.tiktok_privacy not in _TIKTOK_PRIVACY:
        errors.append(
            f"TIKTOK_PRIVACY must be one of {sorted(_TIKTOK_PRIVACY)}, "
            f"got {cfg.tiktok_privacy!r}"
        )
    if cfg.heygen_aspect not in _HEYGEN_ASPECT:
        errors.append(
            f"HEYGEN_ASPECT must be one of {sorted(_HEYGEN_ASPECT)}, "
            f"got {cfg.heygen_aspect!r}"
        )

    # ── Cross-field dependency checks ─────────────────────────────────────────

    if cfg.tiktok_enabled:
        if not cfg.tiktok_client_key:
            errors.append(
                "TIKTOK_CLIENT_KEY is required when TIKTOK_ENABLED=true"
            )
        if not cfg.tiktok_client_secret:
            errors.append(
                "TIKTOK_CLIENT_SECRET is required when TIKTOK_ENABLED=true"
            )
    if cfg.presenter_enabled and not cfg.did_api_key:
        errors.append("DID_API_KEY is required when PRESENTER_ENABLED=true")
    if cfg.heygen_enabled:
        if not cfg.heygen_api_key:
            errors.append("HEYGEN_API_KEY is required when HEYGEN_ENABLED=true")
        _persona_ids = [
            cfg.heygen_avatar_id_direct,
            cfg.heygen_avatar_id_serious,
            cfg.heygen_avatar_id_curious,
            cfg.heygen_avatar_id_professional,
            cfg.heygen_avatar_id_casual,
        ]
        if not cfg.heygen_avatar_id and not any(_persona_ids):
            errors.append(
                "HEYGEN_AVATAR_ID (or at least one HEYGEN_AVATAR_ID_<PERSONA>) "
                "is required when HEYGEN_ENABLED=true"
            )

    if errors:
        bullet_list = "\n".join(f"  • {e}" for e in errors)
        raise ConfigurationError(
            f"Configuration errors ({len(errors)} found):\n{bullet_list}"
        )
