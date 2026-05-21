"""Centralized configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def _env(key: str, default: str | None = None, required: bool = False) -> str:
    val = os.getenv(key, default)
    if required and not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val or ""


@dataclass(frozen=True)
class Settings:
    # LLM
    openai_api_key: str = _env("OPENAI_API_KEY", required=True)
    openai_model: str = _env("OPENAI_MODEL", "gpt-4o-mini")

    # TTS
    elevenlabs_api_key:   str = _env("ELEVENLABS_API_KEY", required=True)
    elevenlabs_voice_id:  str = _env("ELEVENLABS_VOICE_ID",  "qSeXEcewz7tA0Q0qk9fH")
    elevenlabs_model:     str = _env("ELEVENLABS_MODEL",     "eleven_turbo_v2_5")

    # Video (RunwayML no longer used; key kept optional for backward compat)
    runwayml_api_key: str = _env("RUNWAYML_API_KEY", "")
    runwayml_model: str = _env("RUNWAYML_MODEL", "gen3a_turbo")

    # YouTube
    youtube_client_secrets: Path = ROOT / _env("YOUTUBE_CLIENT_SECRETS", "config/client_secrets.json")
    youtube_token_file: Path = ROOT / _env("YOUTUBE_TOKEN_FILE", "config/token.json")
    youtube_category_id: str = _env("YOUTUBE_CATEGORY_ID", "28")
    youtube_privacy: str = _env("YOUTUBE_PRIVACY", "public")

    # Russian language variant
    ru_enabled:                  bool = _env("RU_ENABLED", "false").lower() in ("1", "true", "yes")
    ru_elevenlabs_voice_id:      str  = _env("RU_ELEVENLABS_VOICE_ID",  "TUQNWEvVPBLzMBSVDPUA")
    ru_elevenlabs_model:         str  = _env("RU_ELEVENLABS_MODEL",     "eleven_multilingual_v2")
    ru_youtube_client_secrets:   Path = ROOT / _env("RU_YOUTUBE_CLIENT_SECRETS", "config/client_secrets_ru.json")
    ru_youtube_token_file:       Path = ROOT / _env("RU_YOUTUBE_TOKEN_FILE",     "config/token_ru.json")

    # Stock B-roll video (Pexels primary, Pixabay fallback)
    pexels_api_key:  str = _env("PEXELS_API_KEY",  "")
    pixabay_api_key: str = _env("PIXABAY_API_KEY", "")

    # Stock photos as DALL-E fallback for generic scenes
    unsplash_access_key: str = _env("UNSPLASH_ACCESS_KEY", "")

    # Stable Diffusion via Stability AI — used for breaking-news clips (3× cheaper than DALL-E HD)
    stability_api_key: str = _env("STABILITY_API_KEY", "")

    # Channel branding
    channel_name:      str   = _env("CHANNEL_NAME",      "AI Today")
    channel_handle:    str   = _env("CHANNEL_HANDLE",    "@AIToday")
    channel_cta:       str   = _env("CHANNEL_CTA",       "Subscribe · Daily AI News")
    end_card_enabled:  bool  = _env("END_CARD_ENABLED",  "true").lower() in ("1", "true", "yes")
    end_card_duration: float = float(_env("END_CARD_DURATION", "10"))

    # AI Presenter — D-ID Talks API (optional — leave DID_API_KEY blank to disable)
    # DID_API_KEY: base64 credential from d-id.com dashboard ("Basic <key>" → set just <key>)
    # PRESENTER_AVATAR_PATH: local PNG/JPG that becomes the channel's on-screen host
    did_api_key:           str  = _env("DID_API_KEY",           "")
    presenter_avatar_path: str  = _env("PRESENTER_AVATAR_PATH", "assets/avatar.png")
    presenter_enabled:     bool = _env("PRESENTER_ENABLED", "false").lower() in ("1", "true", "yes")

    # AI Presenter — HeyGen Video API (preferred over D-ID when enabled)
    # HEYGEN_API_KEY:   from app.heygen.com → Settings → API
    # HEYGEN_AVATAR_ID: avatar ID from HeyGen's avatar library or custom avatar
    # HEYGEN_VOICE_ID:  HeyGen voice used for text-mode TTS (fallback when audio fails)
    # HEYGEN_ASPECT:    "16:9" (default) or "9:16" (Shorts)
    heygen_api_key:  str  = _env("HEYGEN_API_KEY",  "")
    heygen_avatar_id: str = _env("HEYGEN_AVATAR_ID", "")
    heygen_voice_id:  str = _env("HEYGEN_VOICE_ID",  "")
    heygen_aspect:    str = _env("HEYGEN_ASPECT",    "16:9")
    heygen_enabled:   bool = _env("HEYGEN_ENABLED", "false").lower() in ("1", "true", "yes")

    # Per-persona avatar IDs (optional — all fall back to heygen_avatar_id)
    heygen_avatar_id_direct:       str = _env("HEYGEN_AVATAR_ID_DIRECT",       "")
    heygen_avatar_id_serious:      str = _env("HEYGEN_AVATAR_ID_SERIOUS",       "")
    heygen_avatar_id_curious:      str = _env("HEYGEN_AVATAR_ID_CURIOUS",       "")
    heygen_avatar_id_professional: str = _env("HEYGEN_AVATAR_ID_PROFESSIONAL",  "")
    heygen_avatar_id_casual:       str = _env("HEYGEN_AVATAR_ID_CASUAL",        "")

    # Slack notifications (optional — leave blank to disable)
    slack_webhook_url: str = _env("SLACK_WEBHOOK_URL", "")

    # Background music (optional — leave blank to disable)
    # Path can be absolute or relative to source_dir (e.g. "background_music.mp3")
    background_music_path:   str   = _env("BACKGROUND_MUSIC_PATH",   "")
    background_music_volume: float = float(_env("BACKGROUND_MUSIC_VOLUME", "0.10"))
    shorts_music_volume:     float = float(_env("SHORTS_MUSIC_VOLUME",     "0.13"))

    # TikTok
    tiktok_enabled:       bool = _env("TIKTOK_ENABLED", "false").lower() in ("1", "true", "yes")
    tiktok_client_key:    str  = _env("TIKTOK_CLIENT_KEY",    "")
    tiktok_client_secret: str  = _env("TIKTOK_CLIENT_SECRET", "")
    tiktok_token_file:    Path = ROOT / _env("TIKTOK_TOKEN_FILE", "config/tiktok_token.json")
    tiktok_privacy:       str  = _env("TIKTOK_PRIVACY", "PUBLIC_TO_EVERYONE")

    # Telegram RSS (via RSSHub or any RSS proxy — comma-separated for multiple channels)
    # Default: t.me/ai_for_devs via the public RSSHub instance.
    # Override with TELEGRAM_RSS_URLS=https://your-rsshub/telegram/channel/name,...
    telegram_rss_urls: str = _env(
        "TELEGRAM_RSS_URLS",
        "https://rsshub.app/telegram/channel/ai_for_devs",
    )

    # Budget alerts — Slack notification when daily or monthly spend exceeds limit.
    # Set to 0 to disable a threshold.
    daily_budget_usd:   float = float(_env("DAILY_BUDGET_USD",   "50.0"))
    monthly_budget_usd: float = float(_env("MONTHLY_BUDGET_USD", "500.0"))

    # Pipeline
    script_target_words: int = int(_env("SCRIPT_TARGET_WORDS", "2200"))
    daily_run_hour_utc: int  = int(_env("DAILY_RUN_HOUR_UTC", "8"))
    topic_run_hour_utc: int  = int(_env("TOPIC_RUN_HOUR_UTC", "10"))  # Tue & Thu upload time
    output_dir: Path = ROOT / _env("OUTPUT_DIR", "output")
    log_dir: Path = ROOT / _env("LOG_DIR", "logs")
    data_dir: Path = ROOT / _env("DATA_DIR", "data")
    dedup_ttl_days: int = int(_env("DEDUP_TTL_DAYS", "30"))
    source_dir: Path = ROOT / _env("SOURCE_DIR", "source")

    # FFmpeg subprocess timeouts (seconds). Increase for slow CI environments.
    ffmpeg_probe_timeout: int = int(_env("FFMPEG_PROBE_TIMEOUT", "30"))
    ffmpeg_clip_timeout:  int = int(_env("FFMPEG_CLIP_TIMEOUT",  "300"))
    ffmpeg_long_timeout:  int = int(_env("FFMPEG_LONG_TIMEOUT",  "600"))

    def __post_init__(self) -> None:
        from src.config_validator import validate
        validate(self)


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
settings.data_dir.mkdir(parents=True, exist_ok=True)
