"""Centralized configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
import PIL.Image

# PIL.Image.ANTIALIAS was removed in Pillow 10; MoviePy 1.x still uses it.
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

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
    elevenlabs_voice_id:  str = _env("ELEVENLABS_VOICE_ID",  "Gfpl8Yo74Is0W6cPUWWT")
    elevenlabs_model:     str = _env("ELEVENLABS_MODEL",     "eleven_turbo_v2_5")

    # Video (RunwayML no longer used; key kept optional for backward compat)
    runwayml_api_key: str = _env("RUNWAYML_API_KEY", "")
    runwayml_model: str = _env("RUNWAYML_MODEL", "gen3a_turbo")

    # YouTube
    youtube_client_secrets: Path = ROOT / _env("YOUTUBE_CLIENT_SECRETS", "config/client_secrets.json")
    youtube_token_file: Path = ROOT / _env("YOUTUBE_TOKEN_FILE", "config/token.pickle")
    youtube_category_id: str = _env("YOUTUBE_CATEGORY_ID", "28")
    youtube_privacy: str = _env("YOUTUBE_PRIVACY", "public")

    # Russian language variant
    ru_enabled:                  bool = _env("RU_ENABLED", "false").lower() in ("1", "true", "yes")
    ru_elevenlabs_voice_id:      str  = _env("RU_ELEVENLABS_VOICE_ID",  "TUQNWEvVPBLzMBSVDPUA")
    ru_elevenlabs_model:         str  = _env("RU_ELEVENLABS_MODEL",     "eleven_multilingual_v2")
    ru_youtube_client_secrets:   Path = ROOT / _env("RU_YOUTUBE_CLIENT_SECRETS", "config/client_secrets_ru.json")
    ru_youtube_token_file:       Path = ROOT / _env("RU_YOUTUBE_TOKEN_FILE",     "config/token_ru.pickle")

    # Stock B-roll (Pexels primary, Pixabay fallback)
    pexels_api_key:  str = _env("PEXELS_API_KEY",  "")
    pixabay_api_key: str = _env("PIXABAY_API_KEY", "")

    # Slack notifications (optional — leave blank to disable)
    slack_webhook_url: str = _env("SLACK_WEBHOOK_URL", "")

    # TikTok
    tiktok_enabled:       bool = _env("TIKTOK_ENABLED", "false").lower() in ("1", "true", "yes")
    tiktok_client_key:    str  = _env("TIKTOK_CLIENT_KEY",    "")
    tiktok_client_secret: str  = _env("TIKTOK_CLIENT_SECRET", "")
    tiktok_token_file:    Path = ROOT / _env("TIKTOK_TOKEN_FILE", "config/tiktok_token.json")
    tiktok_privacy:       str  = _env("TIKTOK_PRIVACY", "PUBLIC_TO_EVERYONE")

    # Pipeline
    script_target_words: int = int(_env("SCRIPT_TARGET_WORDS", "2200"))
    daily_run_hour_utc: int = int(_env("DAILY_RUN_HOUR_UTC", "8"))
    output_dir: Path = ROOT / _env("OUTPUT_DIR", "output")
    log_dir: Path = ROOT / _env("LOG_DIR", "logs")
    data_dir: Path = ROOT / _env("DATA_DIR", "data")
    dedup_ttl_days: int = int(_env("DEDUP_TTL_DAYS", "30"))
    source_dir: Path = ROOT / _env("SOURCE_DIR", "source")


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
settings.data_dir.mkdir(parents=True, exist_ok=True)
