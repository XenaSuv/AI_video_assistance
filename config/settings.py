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
    elevenlabs_api_key: str = _env("ELEVENLABS_API_KEY", required=True)
    elevenlabs_voice_id: str = _env("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
    elevenlabs_model: str = _env("ELEVENLABS_MODEL", "eleven_turbo_v2_5")

    # Video
    runwayml_api_key: str = _env("RUNWAYML_API_KEY", required=True)
    runwayml_model: str = _env("RUNWAYML_MODEL", "gen3a_turbo")

    # YouTube
    youtube_client_secrets: Path = ROOT / _env("YOUTUBE_CLIENT_SECRETS", "config/client_secrets.json")
    youtube_token_file: Path = ROOT / _env("YOUTUBE_TOKEN_FILE", "config/token.pickle")
    youtube_category_id: str = _env("YOUTUBE_CATEGORY_ID", "28")
    youtube_privacy: str = _env("YOUTUBE_PRIVACY", "public")

    # Pipeline
    script_target_words: int = int(_env("SCRIPT_TARGET_WORDS", "2200"))
    daily_run_hour_utc: int = int(_env("DAILY_RUN_HOUR_UTC", "8"))
    output_dir: Path = ROOT / _env("OUTPUT_DIR", "output")
    log_dir: Path = ROOT / _env("LOG_DIR", "logs")


settings = Settings()
settings.output_dir.mkdir(parents=True, exist_ok=True)
settings.log_dir.mkdir(parents=True, exist_ok=True)
