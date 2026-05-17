"""Tests for src/config_validator.py — validate() and ConfigurationError."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config_validator import ConfigurationError, validate


def _cfg(**overrides) -> SimpleNamespace:
    """Minimal valid config with optional overrides."""
    base = dict(
        daily_run_hour_utc=8,
        topic_run_hour_utc=10,
        background_music_volume=0.10,
        shorts_music_volume=0.13,
        script_target_words=2200,
        dedup_ttl_days=30,
        daily_budget_usd=50.0,
        monthly_budget_usd=500.0,
        end_card_duration=10.0,
        youtube_privacy="public",
        tiktok_privacy="PUBLIC_TO_EVERYONE",
        heygen_aspect="16:9",
        tiktok_enabled=False,
        tiktok_client_key="",
        tiktok_client_secret="",
        presenter_enabled=False,
        did_api_key="",
        heygen_enabled=False,
        heygen_api_key="",
        heygen_avatar_id="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── Happy path ────────────────────────────────────────────────────────────────

class TestValidateHappyPath:
    def test_valid_defaults_do_not_raise(self):
        validate(_cfg())

    def test_boundary_values_are_accepted(self):
        validate(_cfg(
            daily_run_hour_utc=0,
            topic_run_hour_utc=23,
            background_music_volume=0.0,
            shorts_music_volume=1.0,
            daily_budget_usd=0.0,
            monthly_budget_usd=0.0,
            end_card_duration=0.0,
        ))

    def test_all_youtube_privacy_values(self):
        for v in ("public", "unlisted", "private"):
            validate(_cfg(youtube_privacy=v))

    def test_all_tiktok_privacy_values(self):
        for v in ("PUBLIC_TO_EVERYONE", "MUTUAL_FOLLOW_FRIENDS", "SELF_ONLY"):
            validate(_cfg(tiktok_privacy=v))

    def test_both_heygen_aspect_values(self):
        for v in ("16:9", "9:16"):
            validate(_cfg(heygen_aspect=v))


# ── Range checks ──────────────────────────────────────────────────────────────

class TestRangeChecks:
    def test_daily_run_hour_below_0(self):
        with pytest.raises(ConfigurationError, match="DAILY_RUN_HOUR_UTC"):
            validate(_cfg(daily_run_hour_utc=-1))

    def test_daily_run_hour_above_23(self):
        with pytest.raises(ConfigurationError, match="DAILY_RUN_HOUR_UTC"):
            validate(_cfg(daily_run_hour_utc=24))

    def test_topic_run_hour_below_0(self):
        with pytest.raises(ConfigurationError, match="TOPIC_RUN_HOUR_UTC"):
            validate(_cfg(topic_run_hour_utc=-1))

    def test_topic_run_hour_above_23(self):
        with pytest.raises(ConfigurationError, match="TOPIC_RUN_HOUR_UTC"):
            validate(_cfg(topic_run_hour_utc=24))

    def test_background_music_volume_below_0(self):
        with pytest.raises(ConfigurationError, match="BACKGROUND_MUSIC_VOLUME"):
            validate(_cfg(background_music_volume=-0.01))

    def test_background_music_volume_above_1(self):
        with pytest.raises(ConfigurationError, match="BACKGROUND_MUSIC_VOLUME"):
            validate(_cfg(background_music_volume=1.01))

    def test_shorts_music_volume_below_0(self):
        with pytest.raises(ConfigurationError, match="SHORTS_MUSIC_VOLUME"):
            validate(_cfg(shorts_music_volume=-0.01))

    def test_shorts_music_volume_above_1(self):
        with pytest.raises(ConfigurationError, match="SHORTS_MUSIC_VOLUME"):
            validate(_cfg(shorts_music_volume=1.01))

    def test_script_target_words_zero(self):
        with pytest.raises(ConfigurationError, match="SCRIPT_TARGET_WORDS"):
            validate(_cfg(script_target_words=0))

    def test_script_target_words_negative(self):
        with pytest.raises(ConfigurationError, match="SCRIPT_TARGET_WORDS"):
            validate(_cfg(script_target_words=-1))

    def test_dedup_ttl_days_zero(self):
        with pytest.raises(ConfigurationError, match="DEDUP_TTL_DAYS"):
            validate(_cfg(dedup_ttl_days=0))

    def test_dedup_ttl_days_negative(self):
        with pytest.raises(ConfigurationError, match="DEDUP_TTL_DAYS"):
            validate(_cfg(dedup_ttl_days=-5))

    def test_daily_budget_negative(self):
        with pytest.raises(ConfigurationError, match="DAILY_BUDGET_USD"):
            validate(_cfg(daily_budget_usd=-0.01))

    def test_monthly_budget_negative(self):
        with pytest.raises(ConfigurationError, match="MONTHLY_BUDGET_USD"):
            validate(_cfg(monthly_budget_usd=-1.0))

    def test_end_card_duration_negative(self):
        with pytest.raises(ConfigurationError, match="END_CARD_DURATION"):
            validate(_cfg(end_card_duration=-1.0))


# ── Choice checks ─────────────────────────────────────────────────────────────

class TestChoiceChecks:
    def test_invalid_youtube_privacy(self):
        with pytest.raises(ConfigurationError, match="YOUTUBE_PRIVACY"):
            validate(_cfg(youtube_privacy="friends_only"))

    def test_case_sensitive_youtube_privacy(self):
        with pytest.raises(ConfigurationError, match="YOUTUBE_PRIVACY"):
            validate(_cfg(youtube_privacy="Public"))

    def test_invalid_tiktok_privacy(self):
        with pytest.raises(ConfigurationError, match="TIKTOK_PRIVACY"):
            validate(_cfg(tiktok_privacy="friends"))

    def test_invalid_heygen_aspect(self):
        with pytest.raises(ConfigurationError, match="HEYGEN_ASPECT"):
            validate(_cfg(heygen_aspect="4:3"))

    def test_heygen_aspect_empty_string(self):
        with pytest.raises(ConfigurationError, match="HEYGEN_ASPECT"):
            validate(_cfg(heygen_aspect=""))


# ── Cross-field dependency checks ─────────────────────────────────────────────

class TestCrossFieldDependencies:
    def test_tiktok_enabled_requires_client_key(self):
        with pytest.raises(ConfigurationError, match="TIKTOK_CLIENT_KEY"):
            validate(_cfg(tiktok_enabled=True, tiktok_client_key="", tiktok_client_secret="secret"))

    def test_tiktok_enabled_requires_client_secret(self):
        with pytest.raises(ConfigurationError, match="TIKTOK_CLIENT_SECRET"):
            validate(_cfg(tiktok_enabled=True, tiktok_client_key="key", tiktok_client_secret=""))

    def test_tiktok_enabled_with_both_credentials_valid(self):
        validate(_cfg(tiktok_enabled=True, tiktok_client_key="key", tiktok_client_secret="secret"))

    def test_tiktok_disabled_no_credentials_needed(self):
        validate(_cfg(tiktok_enabled=False, tiktok_client_key="", tiktok_client_secret=""))

    def test_presenter_enabled_requires_did_api_key(self):
        with pytest.raises(ConfigurationError, match="DID_API_KEY"):
            validate(_cfg(presenter_enabled=True, did_api_key=""))

    def test_presenter_enabled_with_key_valid(self):
        validate(_cfg(presenter_enabled=True, did_api_key="mykey"))

    def test_presenter_disabled_no_key_required(self):
        validate(_cfg(presenter_enabled=False, did_api_key=""))

    def test_heygen_enabled_requires_api_key(self):
        with pytest.raises(ConfigurationError, match="HEYGEN_API_KEY"):
            validate(_cfg(heygen_enabled=True, heygen_api_key="", heygen_avatar_id="avatar123"))

    def test_heygen_enabled_requires_avatar_id(self):
        with pytest.raises(ConfigurationError, match="HEYGEN_AVATAR_ID"):
            validate(_cfg(heygen_enabled=True, heygen_api_key="key", heygen_avatar_id=""))

    def test_heygen_enabled_with_all_values_valid(self):
        validate(_cfg(heygen_enabled=True, heygen_api_key="key", heygen_avatar_id="avatar123"))

    def test_heygen_disabled_no_fields_required(self):
        validate(_cfg(heygen_enabled=False, heygen_api_key="", heygen_avatar_id=""))


# ── All-errors-collected behaviour ────────────────────────────────────────────

class TestMultipleErrors:
    def test_collects_all_errors_before_raising(self):
        cfg = _cfg(
            daily_run_hour_utc=99,
            script_target_words=-5,
            youtube_privacy="bad",
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate(cfg)
        msg = str(exc_info.value)
        assert "DAILY_RUN_HOUR_UTC" in msg
        assert "SCRIPT_TARGET_WORDS" in msg
        assert "YOUTUBE_PRIVACY" in msg
        assert "3 found" in msg

    def test_error_message_uses_bullet_format(self):
        with pytest.raises(ConfigurationError) as exc_info:
            validate(_cfg(daily_run_hour_utc=99))
        assert "  •" in str(exc_info.value)

    def test_tiktok_enabled_reports_both_missing_keys(self):
        with pytest.raises(ConfigurationError) as exc_info:
            validate(_cfg(tiktok_enabled=True, tiktok_client_key="", tiktok_client_secret=""))
        msg = str(exc_info.value)
        assert "TIKTOK_CLIENT_KEY" in msg
        assert "TIKTOK_CLIENT_SECRET" in msg


# ── ConfigurationError type ───────────────────────────────────────────────────

class TestConfigurationError:
    def test_is_value_error_subclass(self):
        assert issubclass(ConfigurationError, ValueError)

    def test_can_be_caught_as_value_error(self):
        with pytest.raises(ValueError):
            raise ConfigurationError("bad config")

    def test_message_preserved(self):
        err = ConfigurationError("my message")
        assert "my message" in str(err)
