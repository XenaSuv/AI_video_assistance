"""Tests for src/config_validator.py — validate() and ConfigurationError."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config_validator import ConfigurationError, validate


def _cfg(**overrides) -> SimpleNamespace:
    """Minimal valid config with optional overrides."""
    base = dict(
        # Numeric — lower bounds
        daily_run_hour_utc=8,
        topic_run_hour_utc=10,
        background_music_volume=0.10,
        shorts_music_volume=0.13,
        script_target_words=2200,
        dedup_ttl_days=30,
        daily_budget_usd=50.0,
        monthly_budget_usd=500.0,
        end_card_duration=10.0,
        ffmpeg_probe_timeout=30,
        ffmpeg_clip_timeout=300,
        ffmpeg_long_timeout=600,
        # Choice
        youtube_privacy="public",
        tiktok_privacy="PUBLIC_TO_EVERYONE",
        heygen_aspect="16:9",
        # Non-empty / format
        openai_model="gpt-4o-mini",
        elevenlabs_voice_id="Gfpl8Yo74Is0W6cPUWWT",
        elevenlabs_model="eleven_turbo_v2_5",
        channel_name="AI Today",
        channel_handle="@AIToday",
        youtube_category_id="28",
        # URL
        slack_webhook_url="",
        telegram_rss_urls="https://rsshub.app/telegram/channel/ai_for_devs",
        # Paths
        youtube_client_secrets=Path("config/client_secrets.json"),
        # Feature flags — all off so cross-field checks don't trigger
        tiktok_enabled=False,
        tiktok_client_key="",
        tiktok_client_secret="",
        tiktok_token_file=Path("config/tiktok_token.json"),
        presenter_enabled=False,
        did_api_key="",
        presenter_avatar_path="assets/avatar.png",
        heygen_enabled=False,
        heygen_api_key="",
        heygen_avatar_id="",
        heygen_avatar_id_direct="",
        heygen_avatar_id_serious="",
        heygen_avatar_id_curious="",
        heygen_avatar_id_professional="",
        heygen_avatar_id_casual="",
        heygen_voice_id="",
        ru_enabled=False,
        ru_elevenlabs_voice_id="TUQNWEvVPBLzMBSVDPUA",
        ru_elevenlabs_model="eleven_multilingual_v2",
        ru_youtube_client_secrets=Path("config/client_secrets_ru.json"),
        ru_youtube_token_file=Path("config/token_ru.json"),
        # Music
        background_music_path="",
        # End card
        end_card_enabled=True,
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
            end_card_enabled=False,   # duration=0 valid when card disabled
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

    def test_equal_daily_and_monthly_budget_ok(self):
        validate(_cfg(daily_budget_usd=100.0, monthly_budget_usd=100.0))

    def test_equal_ffmpeg_timeouts_ok(self):
        validate(_cfg(ffmpeg_probe_timeout=60, ffmpeg_clip_timeout=60, ffmpeg_long_timeout=60))

    def test_openai_model_o1_prefix_accepted(self):
        validate(_cfg(openai_model="o1-mini"))

    def test_openai_model_o3_prefix_accepted(self):
        validate(_cfg(openai_model="o3"))

    def test_slack_webhook_url_empty_ok(self):
        validate(_cfg(slack_webhook_url=""))

    def test_background_music_path_empty_any_volume_ok(self):
        validate(_cfg(background_music_path="", background_music_volume=0.0))


# ── Numeric lower-bound checks ────────────────────────────────────────────────

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


# ── Numeric upper-bound checks ────────────────────────────────────────────────

class TestUpperBoundChecks:
    def test_script_target_words_at_limit_ok(self):
        validate(_cfg(script_target_words=10_000))

    def test_script_target_words_above_limit(self):
        with pytest.raises(ConfigurationError, match="SCRIPT_TARGET_WORDS"):
            validate(_cfg(script_target_words=10_001))

    def test_dedup_ttl_days_at_limit_ok(self):
        validate(_cfg(dedup_ttl_days=365))

    def test_dedup_ttl_days_above_limit(self):
        with pytest.raises(ConfigurationError, match="DEDUP_TTL_DAYS"):
            validate(_cfg(dedup_ttl_days=366))

    def test_end_card_duration_at_limit_ok(self):
        validate(_cfg(end_card_duration=60.0))

    def test_end_card_duration_above_limit(self):
        with pytest.raises(ConfigurationError, match="END_CARD_DURATION"):
            validate(_cfg(end_card_duration=61.0))


# ── Numeric ordering checks ───────────────────────────────────────────────────

class TestNumericOrderingChecks:
    def test_clip_timeout_less_than_probe_raises(self):
        with pytest.raises(ConfigurationError, match="FFMPEG_CLIP_TIMEOUT"):
            validate(_cfg(ffmpeg_probe_timeout=300, ffmpeg_clip_timeout=29))

    def test_long_timeout_less_than_clip_raises(self):
        with pytest.raises(ConfigurationError, match="FFMPEG_LONG_TIMEOUT"):
            validate(_cfg(ffmpeg_clip_timeout=300, ffmpeg_long_timeout=299))

    def test_daily_budget_exceeds_monthly_raises(self):
        with pytest.raises(ConfigurationError, match="DAILY_BUDGET_USD"):
            validate(_cfg(daily_budget_usd=600.0, monthly_budget_usd=500.0))

    def test_daily_budget_exceeds_monthly_only_when_monthly_positive(self):
        # monthly=0 means "no monthly cap" — any daily amount is fine
        validate(_cfg(daily_budget_usd=999.0, monthly_budget_usd=0.0))

    def test_timeouts_ordering_with_probe_equals_clip(self):
        validate(_cfg(ffmpeg_probe_timeout=300, ffmpeg_clip_timeout=300, ffmpeg_long_timeout=600))


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


# ── Non-empty / format checks ─────────────────────────────────────────────────

class TestNonEmptyAndFormatChecks:
    def test_openai_model_empty_raises(self):
        with pytest.raises(ConfigurationError, match="OPENAI_MODEL"):
            validate(_cfg(openai_model=""))

    def test_openai_model_unknown_prefix_raises(self):
        with pytest.raises(ConfigurationError, match="OPENAI_MODEL"):
            validate(_cfg(openai_model="custom-model-v1"))

    def test_openai_model_gpt_prefix_accepted(self):
        validate(_cfg(openai_model="gpt-4o"))

    def test_elevenlabs_voice_id_empty_raises(self):
        with pytest.raises(ConfigurationError, match="ELEVENLABS_VOICE_ID"):
            validate(_cfg(elevenlabs_voice_id=""))

    def test_elevenlabs_model_empty_raises(self):
        with pytest.raises(ConfigurationError, match="ELEVENLABS_MODEL"):
            validate(_cfg(elevenlabs_model=""))

    def test_channel_name_empty_raises(self):
        with pytest.raises(ConfigurationError, match="CHANNEL_NAME"):
            validate(_cfg(channel_name=""))

    def test_channel_handle_without_at_raises(self):
        with pytest.raises(ConfigurationError, match="CHANNEL_HANDLE"):
            validate(_cfg(channel_handle="AIToday"))

    def test_channel_handle_starting_with_at_ok(self):
        validate(_cfg(channel_handle="@my_channel"))

    def test_youtube_category_id_empty_raises(self):
        with pytest.raises(ConfigurationError, match="YOUTUBE_CATEGORY_ID"):
            validate(_cfg(youtube_category_id=""))

    def test_youtube_category_id_non_numeric_raises(self):
        with pytest.raises(ConfigurationError, match="YOUTUBE_CATEGORY_ID"):
            validate(_cfg(youtube_category_id="news"))

    def test_youtube_category_id_numeric_string_ok(self):
        validate(_cfg(youtube_category_id="28"))


# ── URL format checks ─────────────────────────────────────────────────────────

class TestURLFormatChecks:
    def test_slack_webhook_must_be_https(self):
        with pytest.raises(ConfigurationError, match="SLACK_WEBHOOK_URL"):
            validate(_cfg(slack_webhook_url="http://hooks.slack.com/bad"))

    def test_slack_webhook_http_scheme_raises(self):
        with pytest.raises(ConfigurationError, match="SLACK_WEBHOOK_URL"):
            validate(_cfg(slack_webhook_url="hooks.slack.com/path"))

    def test_slack_webhook_valid_https_ok(self):
        validate(_cfg(slack_webhook_url="https://hooks.slack.com/services/T/B/X"))

    def test_telegram_rss_urls_bad_url_raises(self):
        with pytest.raises(ConfigurationError, match="TELEGRAM_RSS_URLS"):
            validate(_cfg(telegram_rss_urls="not-a-url"))

    def test_telegram_rss_urls_valid_ok(self):
        validate(_cfg(telegram_rss_urls="https://rsshub.app/chan1,https://rsshub.app/chan2"))

    def test_telegram_rss_urls_empty_ok(self):
        validate(_cfg(telegram_rss_urls=""))


# ── Path suffix checks ────────────────────────────────────────────────────────

class TestPathSuffixChecks:
    def test_youtube_client_secrets_not_json_raises(self):
        with pytest.raises(ConfigurationError, match="YOUTUBE_CLIENT_SECRETS"):
            validate(_cfg(youtube_client_secrets=Path("config/secrets.yaml")))

    def test_youtube_client_secrets_json_ok(self):
        validate(_cfg(youtube_client_secrets=Path("config/client_secrets.json")))


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

    def test_tiktok_token_file_must_be_json(self):
        with pytest.raises(ConfigurationError, match="TIKTOK_TOKEN_FILE"):
            validate(_cfg(
                tiktok_enabled=True,
                tiktok_client_key="k",
                tiktok_client_secret="s",
                tiktok_token_file=Path("config/token.pkl"),
            ))

    def test_presenter_enabled_requires_did_api_key(self):
        with pytest.raises(ConfigurationError, match="DID_API_KEY"):
            validate(_cfg(presenter_enabled=True, did_api_key=""))

    def test_presenter_enabled_requires_avatar_path(self):
        with pytest.raises(ConfigurationError, match="PRESENTER_AVATAR_PATH"):
            validate(_cfg(presenter_enabled=True, did_api_key="key", presenter_avatar_path=""))

    def test_presenter_enabled_with_both_ok(self):
        validate(_cfg(presenter_enabled=True, did_api_key="key", presenter_avatar_path="avatar.png"))

    def test_presenter_disabled_no_key_required(self):
        validate(_cfg(presenter_enabled=False, did_api_key=""))

    def test_presenter_enabled_with_heygen_active_no_did_key_required(self):
        # When HeyGen is the active presenter, D-ID credentials are not needed
        validate(_cfg(
            presenter_enabled=True,
            heygen_enabled=True,
            heygen_api_key="key",
            heygen_avatar_id="avatar123",
            heygen_voice_id="voice123",
            did_api_key="",
            presenter_avatar_path="",
        ))

    def test_heygen_enabled_requires_api_key(self):
        with pytest.raises(ConfigurationError, match="HEYGEN_API_KEY"):
            validate(_cfg(heygen_enabled=True, heygen_api_key="", heygen_avatar_id="avatar123"))

    def test_heygen_enabled_requires_avatar_id_when_no_persona_ids(self):
        with pytest.raises(ConfigurationError, match="HEYGEN_AVATAR_ID"):
            validate(_cfg(heygen_enabled=True, heygen_api_key="key", heygen_avatar_id="",
                          heygen_voice_id="voice123"))

    def test_heygen_enabled_requires_voice_id(self):
        with pytest.raises(ConfigurationError, match="HEYGEN_VOICE_ID"):
            validate(_cfg(heygen_enabled=True, heygen_api_key="key",
                          heygen_avatar_id="avatar123", heygen_voice_id=""))

    def test_heygen_enabled_persona_id_satisfies_avatar_requirement(self):
        validate(_cfg(
            heygen_enabled=True,
            heygen_api_key="key",
            heygen_avatar_id="",
            heygen_avatar_id_direct="persona_avatar_xyz",
            heygen_voice_id="voice123",
        ))

    def test_heygen_enabled_with_default_avatar_id_and_voice_valid(self):
        validate(_cfg(heygen_enabled=True, heygen_api_key="key",
                      heygen_avatar_id="avatar123", heygen_voice_id="voice456"))

    def test_heygen_disabled_no_fields_required(self):
        validate(_cfg(heygen_enabled=False, heygen_api_key="", heygen_avatar_id=""))


# ── RU cross-field checks ─────────────────────────────────────────────────────

class TestRuCrossFieldChecks:
    def test_ru_enabled_requires_voice_id(self):
        with pytest.raises(ConfigurationError, match="RU_ELEVENLABS_VOICE_ID"):
            validate(_cfg(ru_enabled=True, ru_elevenlabs_voice_id=""))

    def test_ru_enabled_requires_model(self):
        with pytest.raises(ConfigurationError, match="RU_ELEVENLABS_MODEL"):
            validate(_cfg(ru_enabled=True, ru_elevenlabs_model=""))

    def test_ru_enabled_requires_json_client_secrets(self):
        with pytest.raises(ConfigurationError, match="RU_YOUTUBE_CLIENT_SECRETS"):
            validate(_cfg(ru_enabled=True,
                          ru_youtube_client_secrets=Path("config/ru_creds.yaml")))

    def test_ru_enabled_requires_json_token_file(self):
        with pytest.raises(ConfigurationError, match="RU_YOUTUBE_TOKEN_FILE"):
            validate(_cfg(ru_enabled=True,
                          ru_youtube_token_file=Path("config/token_ru.pkl")))

    def test_ru_enabled_with_valid_fields_ok(self):
        validate(_cfg(
            ru_enabled=True,
            ru_elevenlabs_voice_id="voice_ru",
            ru_elevenlabs_model="eleven_multilingual_v2",
            ru_youtube_client_secrets=Path("config/client_secrets_ru.json"),
            ru_youtube_token_file=Path("config/token_ru.json"),
        ))

    def test_ru_disabled_missing_fields_ok(self):
        validate(_cfg(
            ru_enabled=False,
            ru_elevenlabs_voice_id="",
            ru_elevenlabs_model="",
        ))


# ── Music cross-field checks ──────────────────────────────────────────────────

class TestMusicCrossFieldChecks:
    def test_music_path_set_requires_positive_volume(self):
        with pytest.raises(ConfigurationError, match="BACKGROUND_MUSIC_VOLUME"):
            validate(_cfg(background_music_path="source/music.mp3", background_music_volume=0.0))

    def test_music_path_set_with_positive_volume_ok(self):
        validate(_cfg(background_music_path="source/music.mp3", background_music_volume=0.05))

    def test_music_path_empty_zero_volume_ok(self):
        validate(_cfg(background_music_path="", background_music_volume=0.0))


# ── End-card cross-field checks ───────────────────────────────────────────────

class TestEndCardChecks:
    def test_end_card_enabled_requires_positive_duration(self):
        with pytest.raises(ConfigurationError, match="END_CARD_DURATION"):
            validate(_cfg(end_card_enabled=True, end_card_duration=0.0))

    def test_end_card_enabled_with_positive_duration_ok(self):
        validate(_cfg(end_card_enabled=True, end_card_duration=10.0))

    def test_end_card_disabled_zero_duration_ok(self):
        validate(_cfg(end_card_enabled=False, end_card_duration=0.0))


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

    def test_ru_enabled_reports_all_missing_fields(self):
        with pytest.raises(ConfigurationError) as exc_info:
            validate(_cfg(
                ru_enabled=True,
                ru_elevenlabs_voice_id="",
                ru_elevenlabs_model="",
            ))
        msg = str(exc_info.value)
        assert "RU_ELEVENLABS_VOICE_ID" in msg
        assert "RU_ELEVENLABS_MODEL" in msg


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
