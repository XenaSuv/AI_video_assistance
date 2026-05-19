# AI Video Assistance

A self-improving AI media system that generates, publishes, and learns from daily AI news videos, weekly tutorials, breaking alerts, and weekly digests — fully automated, English + Russian, YouTube + TikTok.

---

## What Makes This Different

Most AI content pipelines generate videos. This system does three things:

1. **Decides what is worth saying** — scores stories for virality, selects narrative angles, assigns personas
2. **Controls how it is told** — generates scripts, humanises narration, injects micro-hooks every 20–40 s
3. **Learns what actually worked** — pulls real retention data from YouTube, updates angle weights, evolves hooks

The result behaves like a media team, not a script generator.

---

## Content Calendar

| Schedule | Pipeline | Output | Platforms |
|---|---|---|---|
| Mon – Sat 08:00 UTC | Daily AI News | 15-min video + Short | YouTube EN/RU, TikTok |
| Breaking (every 30 min) | Breaking News | 2–5 min + Short | YouTube EN/RU |
| Mon/Wed/Fri 09:00 UTC | Weekly Tutorial | 20-min + Short | YouTube EN/RU |
| Sun 10:00 UTC | Weekly Digest | 35-min recap + Short | YouTube EN/RU |

Monthly output: ~150 long-form videos + unlimited Shorts.

---

## Architecture

```
News Sources
    ↓
Layer 1 — Story Intelligence      editorial_brain · viral_selector · hook_optimizer
    ↓
Layer 2 — Narrative Engine        script_generator · humanizer_agent · voice_generator
                                  video_generator · image_generator
    ↓
Layer 3 — Learning System         feedback_analyzer · thompson_bandit · decision_engine
    ↓
YouTube / TikTok metrics ─────────────────────────────────────────────(back to Layer 1)
```

Each published video feeds real retention data back into Layer 1. The system continuously shifts angle weights, hook styles, and scene structure toward what retains viewers.

---

## Requirements

- Python 3.12
- FFmpeg: `sudo apt-get install ffmpeg`
- ImageMagick: `sudo apt-get install imagemagick fonts-dejavu`

---

## Setup

```bash
git clone https://github.com/XenaSuv/AI_video_assistance.git
cd AI_video_assistance
pip install -r requirements.txt
```

Create `.env` in the project root:

```bash
# Required
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...

# YouTube — EN channel
YOUTUBE_CLIENT_SECRETS=config/client_secrets.json
YOUTUBE_TOKEN_FILE=config/token.pickle
YOUTUBE_PRIVACY=public

# Russian variant (optional)
RU_ENABLED=true
RU_ELEVENLABS_VOICE_ID=TUQNWEvVPBLzMBSVDPUA
RU_YOUTUBE_CLIENT_SECRETS=config/client_secrets_ru.json
RU_YOUTUBE_TOKEN_FILE=config/token_ru.pickle

# Stock media (optional but recommended)
PEXELS_API_KEY=...
PIXABAY_API_KEY=...

# TikTok (optional)
TIKTOK_ENABLED=false
TIKTOK_CLIENT_KEY=...
TIKTOK_CLIENT_SECRET=...

# Slack notifications (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...

# Pipeline tuning
OPENAI_MODEL=gpt-4o-mini
SCRIPT_TARGET_WORDS=2200
DAILY_BUDGET_USD=5.00
MONTHLY_BUDGET_USD=120.00
```

YouTube OAuth tokens are generated on first run (requires browser auth).

---

## Running Pipelines

All pipelines are idempotent — re-running reuses cached artifacts from `output/YYYY-MM-DD/`.

```bash
# Daily AI news
python src/main.py

# Weekly tutorials
python src/weekly_main.py --tool claude     # Monday
python src/weekly_main.py --tool chatgpt   # Wednesday
python src/weekly_main.py --tool gemini    # Friday

# Breaking news
python src/breaking_main.py --item data/breaking_current.json

# Weekly digest
python src/digest_main.py

# Breaking news detector
python src/breaking_detector.py --check

# Automated scheduler (all pipelines on cron)
python src/scheduler.py

# Docker
docker build -t ai-video .
docker run --env-file .env ai-video
```

---

## Testing

```bash
python -m pytest tests/ -v
python -m pytest tests/ --cov=src --cov-report=term-missing
```

Test suite: 4 034 tests, 88% overall coverage. CI enforces 75% floor (80% for new modules).

---

## CI / CD

GitHub Actions runs on every push and PR to `main`:

| Step | Tool | Requirement |
|---|---|---|
| Lint | ruff | Zero violations |
| Type check | mypy --strict | Zero errors |
| Tests + coverage | pytest-cov | ≥ 75% overall, ≥ 80% new modules |

Production pipelines run on schedule via separate workflows (`daily.yml`, `weekly.yml`, `breaking.yml`, `digest.yml`, `topic.yml`).

---

## Project Structure

```
AI_video_assistance/
├── src/                    # 106 source modules
├── config/
│   ├── settings.py         # Frozen dataclass — all config loaded at import
│   └── client_secrets.json # YouTube OAuth2 credentials
├── tests/                  # 121 test files, 4 034 tests
├── scripts/
│   └── check_new_module_coverage.py
├── data/                   # Performance store, dedup DB, bandit state
├── source/                 # Static media assets (intro, outro, music)
├── output/                 # Generated videos (per-date subdirectories)
├── logs/                   # Loguru logs (10 MB rotation)
├── Dockerfile
├── requirements.txt
└── .github/workflows/
```

---

## License

MIT
