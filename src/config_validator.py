"""Validate Settings at startup — collects all errors before raising.

Usage (already wired in config/settings.py):
    from src.config_validator import validate
    validate(settings)

For runtime path checks (call at pipeline start, not at import time):
    from src.config_validator import validate_runtime_paths
    validate_runtime_paths(settings)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when one or more configuration values are invalid.

    The message lists every problem found, not just the first.
    """


_YOUTUBE_PRIVACY    = frozenset({"public", "unlisted", "private"})
_TIKTOK_PRIVACY     = frozenset({"PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"})
_HEYGEN_ASPECT      = frozenset({"16:9", "9:16"})
_OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "o4", "claude-")


def validate(cfg: Any) -> None:
    """Check all settings fields.

    Raises ConfigurationError listing every problem found.
    Does nothing when the configuration is valid.
    """
    errors: list[str] = []

    # ── Numeric lower bounds ──────────────────────────────────────────────────

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
    if cfg.ffmpeg_probe_timeout <= 0:
        errors.append(
            f"FFMPEG_PROBE_TIMEOUT must be > 0, got {cfg.ffmpeg_probe_timeout}"
        )
    if cfg.ffmpeg_clip_timeout <= 0:
        errors.append(
            f"FFMPEG_CLIP_TIMEOUT must be > 0, got {cfg.ffmpeg_clip_timeout}"
        )
    if cfg.ffmpeg_long_timeout <= 0:
        errors.append(
            f"FFMPEG_LONG_TIMEOUT must be > 0, got {cfg.ffmpeg_long_timeout}"
        )

    # ── Numeric upper bounds ──────────────────────────────────────────────────

    if cfg.script_target_words > 10_000:
        errors.append(
            f"SCRIPT_TARGET_WORDS must be <= 10000, got {cfg.script_target_words}"
        )
    if cfg.dedup_ttl_days > 365:
        errors.append(
            f"DEDUP_TTL_DAYS must be <= 365, got {cfg.dedup_ttl_days}"
        )
    if cfg.end_card_duration > 60:
        errors.append(
            f"END_CARD_DURATION must be <= 60, got {cfg.end_card_duration}"
        )

    # ── Numeric ordering ──────────────────────────────────────────────────────

    if cfg.ffmpeg_clip_timeout < cfg.ffmpeg_probe_timeout:
        errors.append(
            f"FFMPEG_CLIP_TIMEOUT ({cfg.ffmpeg_clip_timeout}) must be "
            f">= FFMPEG_PROBE_TIMEOUT ({cfg.ffmpeg_probe_timeout})"
        )
    if cfg.ffmpeg_long_timeout < cfg.ffmpeg_clip_timeout:
        errors.append(
            f"FFMPEG_LONG_TIMEOUT ({cfg.ffmpeg_long_timeout}) must be "
            f">= FFMPEG_CLIP_TIMEOUT ({cfg.ffmpeg_clip_timeout})"
        )
    if cfg.monthly_budget_usd > 0 and cfg.daily_budget_usd > cfg.monthly_budget_usd:
        errors.append(
            f"DAILY_BUDGET_USD ({cfg.daily_budget_usd}) must be "
            f"<= MONTHLY_BUDGET_USD ({cfg.monthly_budget_usd})"
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

    # ── Non-empty and format checks ───────────────────────────────────────────

    if not cfg.openai_model:
        errors.append("OPENAI_MODEL must not be empty")
    elif not any(cfg.openai_model.startswith(p) for p in _OPENAI_MODEL_PREFIXES):
        errors.append(
            f"OPENAI_MODEL {cfg.openai_model!r} does not match a known prefix "
            f"{_OPENAI_MODEL_PREFIXES}"
        )
    if not cfg.elevenlabs_voice_id:
        errors.append("ELEVENLABS_VOICE_ID must not be empty")
    if not cfg.elevenlabs_model:
        errors.append("ELEVENLABS_MODEL must not be empty")
    if not cfg.channel_name:
        errors.append("CHANNEL_NAME must not be empty")
    if not str(cfg.channel_handle).startswith("@"):
        errors.append(
            f"CHANNEL_HANDLE must start with '@', got {cfg.channel_handle!r}"
        )
    _cat_id = str(cfg.youtube_category_id).strip()
    if not _cat_id or not _cat_id.isdigit():
        errors.append(
            f"YOUTUBE_CATEGORY_ID must be a non-empty integer string, "
            f"got {cfg.youtube_category_id!r}"
        )

    # ── URL format checks ─────────────────────────────────────────────────────

    if cfg.slack_webhook_url and not cfg.slack_webhook_url.startswith("https://"):
        errors.append(
            f"SLACK_WEBHOOK_URL must start with 'https://', "
            f"got {cfg.slack_webhook_url!r}"
        )
    if cfg.telegram_rss_urls:
        for _url in cfg.telegram_rss_urls.split(","):
            _url = _url.strip()
            if _url and not _url.startswith("http"):
                errors.append(
                    f"TELEGRAM_RSS_URLS entry must start with 'http', "
                    f"got {_url!r}"
                )
                break  # one error per field is enough

    # ── Path suffix checks ────────────────────────────────────────────────────

    if not str(cfg.youtube_client_secrets).endswith(".json"):
        errors.append(
            f"YOUTUBE_CLIENT_SECRETS must point to a .json file, "
            f"got {cfg.youtube_client_secrets!r}"
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
        if not str(cfg.tiktok_token_file).endswith(".json"):
            errors.append(
                f"TIKTOK_TOKEN_FILE must point to a .json file, "
                f"got {cfg.tiktok_token_file!r}"
            )

    if cfg.presenter_enabled and not cfg.heygen_enabled and not cfg.did_api_key:
        errors.append("DID_API_KEY is required when PRESENTER_ENABLED=true and HEYGEN_ENABLED=false")
    if cfg.presenter_enabled and not cfg.heygen_enabled and not cfg.presenter_avatar_path:
        errors.append(
            "PRESENTER_AVATAR_PATH must not be empty when PRESENTER_ENABLED=true"
        )

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
        if not cfg.heygen_voice_id:
            errors.append(
                "HEYGEN_VOICE_ID is required when HEYGEN_ENABLED=true "
                "(used as TTS fallback when audio upload fails)"
            )

    if cfg.ru_enabled:
        if not cfg.ru_elevenlabs_voice_id:
            errors.append(
                "RU_ELEVENLABS_VOICE_ID is required when RU_ENABLED=true"
            )
        if not cfg.ru_elevenlabs_model:
            errors.append(
                "RU_ELEVENLABS_MODEL is required when RU_ENABLED=true"
            )
        if not str(cfg.ru_youtube_client_secrets).endswith(".json"):
            errors.append(
                f"RU_YOUTUBE_CLIENT_SECRETS must point to a .json file, "
                f"got {cfg.ru_youtube_client_secrets!r}"
            )
        if not str(cfg.ru_youtube_token_file).endswith(".json"):
            errors.append(
                f"RU_YOUTUBE_TOKEN_FILE must point to a .json file, "
                f"got {cfg.ru_youtube_token_file!r}"
            )

    if cfg.background_music_path and cfg.background_music_volume <= 0:
        errors.append(
            f"BACKGROUND_MUSIC_VOLUME must be > 0 when BACKGROUND_MUSIC_PATH "
            f"is set, got {cfg.background_music_volume}"
        )

    if cfg.end_card_enabled and cfg.end_card_duration <= 0:
        errors.append(
            f"END_CARD_DURATION must be > 0 when END_CARD_ENABLED=true, "
            f"got {cfg.end_card_duration}"
        )

    if errors:
        bullet_list = "\n".join(f"  • {e}" for e in errors)
        raise ConfigurationError(
            f"Configuration errors ({len(errors)} found):\n{bullet_list}"
        )


# ── Runtime path validation ───────────────────────────────────────────────────

def _missing(path: Path, env_var: str, errors: list[str]) -> None:
    """Append an error if *path* does not point to an existing file."""
    if not path.is_file():
        errors.append(f"{env_var} does not exist: {path}")


def validate_runtime_paths(cfg: Any) -> None:
    """Check that critical filesystem paths exist.

    Call explicitly at the beginning of each pipeline entry point — NOT at
    import time — so that the test suite (which never touches real files) is
    unaffected.  Raises ConfigurationError listing every missing path.
    """
    errors: list[str] = []

    # ── Source assets directory ───────────────────────────────────────────────
    source_dir = Path(cfg.source_dir)
    if not source_dir.is_dir():
        errors.append(f"SOURCE_DIR does not exist: {source_dir}")

    # ── Background music file (optional — only when path is configured) ───────
    if cfg.background_music_path:
        raw = str(cfg.background_music_path).strip()
        music_path = Path(raw) if Path(raw).is_absolute() else source_dir / raw
        if not music_path.is_file():
            errors.append(f"BACKGROUND_MUSIC_PATH does not exist: {music_path}")

    # ── D-ID presenter avatar (only when presenter is active without HeyGen) ──
    if cfg.presenter_enabled and not cfg.heygen_enabled and cfg.presenter_avatar_path:
        avatar_raw = Path(cfg.presenter_avatar_path)
        avatar = avatar_raw if avatar_raw.is_absolute() else source_dir.parent / avatar_raw
        if not avatar.is_file():
            errors.append(f"PRESENTER_AVATAR_PATH does not exist: {avatar}")

    # ── YouTube OAuth credentials ─────────────────────────────────────────────
    _missing(Path(cfg.youtube_client_secrets), "YOUTUBE_CLIENT_SECRETS", errors)
    _missing(Path(cfg.youtube_token_file),     "YOUTUBE_TOKEN_FILE",     errors)

    # ── Russian variant credentials ───────────────────────────────────────────
    if cfg.ru_enabled:
        _missing(Path(cfg.ru_youtube_client_secrets), "RU_YOUTUBE_CLIENT_SECRETS", errors)
        _missing(Path(cfg.ru_youtube_token_file),     "RU_YOUTUBE_TOKEN_FILE",     errors)

    if errors:
        bullet_list = "\n".join(f"  • {e}" for e in errors)
        raise ConfigurationError(
            f"Runtime path errors ({len(errors)} found):\n{bullet_list}"
        )
