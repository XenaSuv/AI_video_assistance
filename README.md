# AI Video Assistance — Automated AI Content Pipeline

Fully automated content factory that produces daily AI news videos and three weekly tutorial series — all posted to YouTube (English + Russian) and TikTok, hands-free from scraping to publishing.

## Content calendar

| Day | Pipeline | Output | Platforms |
|---|---|---|---|
| Mon – Sun | Daily AI News | 15-min news video + Short | YouTube EN, YouTube RU, TikTok |
| Monday | Weekly Tutorial — Claude | 20-min how-to video | YouTube EN, YouTube RU |
| Wednesday | Weekly Tutorial — ChatGPT | 20-min how-to video | YouTube EN, YouTube RU |
| Friday | Weekly Tutorial — Gemini | 20-min how-to video | YouTube EN, YouTube RU |

---

## Pipelines

### Daily news pipeline

```
scrape → deduplicate → script → voice → video → shorts → thumbnail → upload
                                                                    └── TikTok (short)
└─ if RU_ENABLED: translate → ru-voice → reassemble → ru-shorts → ru-thumbnail → ru-upload
```

1. **Scrape** — pulls the latest AI news from arXiv (cs.AI/LG/CL/CV), HuggingFace Daily Papers, Hacker News, and official company blogs (OpenAI, Anthropic, Google DeepMind, Microsoft AI, Meta AI, DeepSeek, Mistral, xAI) via Google News RSS. Official company items are scored 5.0 (vs. community max ~2.0) and get guaranteed slots in the final selection.
2. **Deduplicate** — SQLite rolling window (30 days) prevents the same story appearing twice.
3. **Script** — GPT-4o-mini writes a structured narration script with title, description, chapters, and tags.
4. **Voice** — ElevenLabs TTS synthesises each scene; `narration.duration` (float) drives all video timing to prevent black-screen gaps.
5. **Video** — DALL-E 3 generates one image per scene; MoviePy animates with Ken Burns pan. Scene title cards (dark navy, sky-blue accent bars) are inserted before each scene.
6. **Shorts** — hook scene auto-cropped to 9:16 (1080×1920) with animated captions.
7. **Thumbnail** — 16 candidate frames scored for colorfulness + contrast; title overlaid; uploaded via YouTube Thumbnails API.
8. **Publish** — YouTube Data API v3 uploads both videos; TikTok Content Posting API v2 uploads the Short.
9. **Russian variant** (optional) — full translate → voice → reassemble → upload loop for a second YouTube channel.

All steps cache under `output/YYYY-MM-DD/` — safe to re-run after any failure.

### Weekly tutorial pipeline

```
pick_topic → script → voice → video → shorts → thumbnail → upload
└─ if RU_ENABLED: translate → ru-voice → reassemble → ru-shorts → ru-thumbnail → ru-upload
```

1. **Topic rotation** — each tool (claude/chatgpt/gemini) has its own pool of ~33–36 topics tracked in `data/weekly_topics_{tool}.json`. Topics cool down for 26 weeks (~6 months) before repeating.
2. **Script** — GPT-4o-mini writes a 7-scene tutorial script (~2,200 words) tailored to the specific AI tool and its audience.
3. **Voice / Video / Shorts / Thumbnail** — identical pipeline to the daily news video.
4. **Upload** — English and Russian variants posted to their respective YouTube channels.

Output lives in `output/weekly/{tool}/YYYY-MM-DD/` so all three tools stay separate.

---

## Project layout

```
├── src/
│   ├── main.py                    # Daily pipeline orchestrator
│   ├── weekly_main.py             # Weekly tutorial pipeline orchestrator
│   ├── scraper.py                 # arXiv + HuggingFace + HN + official company RSS
│   ├── script_generator.py        # GPT-4o-mini daily news script
│   ├── weekly_script_generator.py # Per-tool topic pool + GPT tutorial script
│   ├── voice_generator.py         # ElevenLabs TTS (per-scene mp3)
│   ├── image_generator.py         # DALL-E 3 images + Ken Burns clips
│   ├── video_generator.py         # MoviePy assembly + PIL scene title cards
│   ├── shorts_generator.py        # 9:16 crop + caption burn-in
│   ├── thumbnail_generator.py     # Frame scoring + title overlay
│   ├── youtube_uploader.py        # OAuth2 YouTube upload + thumbnail
│   ├── tiktok_uploader.py         # TikTok Content Posting API v2 + OAuth2+PKCE
│   ├── deduplicator.py            # SQLite cross-day story dedup
│   ├── translator.py              # GPT-powered EN→RU script translation
│   └── scheduler.py              # APScheduler local daemon
├── config/
│   └── settings.py               # Centralised config (env vars)
├── data/
│   ├── seen_stories.db            # Dedup database (auto-created)
│   ├── weekly_topics_claude.json  # Claude topic usage log
│   ├── weekly_topics_chatgpt.json # ChatGPT topic usage log
│   └── weekly_topics_gemini.json  # Gemini topic usage log
├── requirements.txt
├── Dockerfile
└── .github/workflows/
    ├── daily.yml                  # Runs 08:00 UTC every day
    └── weekly.yml                 # Mon 09:00 Claude, Wed 09:00 ChatGPT, Fri 09:00 Gemini
```

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in API keys
```

### YouTube OAuth (one-time per channel)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com), enable YouTube Data API v3, and create an OAuth 2.0 Desktop client.
2. Download `client_secrets.json` → place in `config/`.
3. Run the auth flow to save `config/token.pickle`:

```bash
python src/youtube_uploader.py --auth
```

For the Russian channel, repeat with `client_secrets_ru.json` and save as `config/token_ru.pickle`:

```bash
YOUTUBE_CLIENT_SECRETS=config/client_secrets_ru.json \
YOUTUBE_TOKEN_FILE=config/token_ru.pickle \
python src/youtube_uploader.py --auth
```

### TikTok OAuth (one-time)

1. Create a TikTok developer app at [developers.tiktok.com](https://developers.tiktok.com) with scopes: `video.publish`, `video.upload`, `user.info.basic`.
2. Set your app credentials:

```bash
export TIKTOK_CLIENT_KEY=your_key
export TIKTOK_CLIENT_SECRET=your_secret
```

3. Run the interactive auth flow (opens a browser):

```bash
python src/tiktok_uploader.py --auth
```

This saves `config/tiktok_token.json`. Encode for GitHub Actions:

```bash
base64 -w0 config/tiktok_token.json   # → TIKTOK_TOKEN_JSON_B64 secret
```

The access token (24h TTL) auto-refreshes using the stored refresh token (365-day TTL).

---

## Running locally

### Daily news pipeline

```bash
python src/main.py                          # full run
python src/main.py --skip-upload            # build video, skip all uploads
```

### Weekly tutorial pipeline

```bash
python src/weekly_main.py --tool claude     # Monday tutorial
python src/weekly_main.py --tool chatgpt    # Wednesday tutorial
python src/weekly_main.py --tool gemini     # Friday tutorial
python src/weekly_main.py --tool claude --skip-upload
python src/weekly_main.py --tool claude --topic "Custom topic override"
```

### Local scheduler (daemon)

```bash
python src/scheduler.py    # runs daily at DAILY_RUN_HOUR_UTC (default 08:00 UTC)
```

---

## Environment variables

### Core

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Script generation + DALL-E 3 images |
| `OPENAI_MODEL` | `gpt-4o-mini` | GPT model for scripts |
| `ELEVENLABS_API_KEY` | — | **Required.** TTS synthesis |
| `ELEVENLABS_VOICE_ID` | Rachel | ElevenLabs voice ID for English |
| `ELEVENLABS_MODEL` | `eleven_turbo_v2_5` | TTS model |
| `SCRIPT_TARGET_WORDS` | `2200` | Target narration length |
| `OUTPUT_DIR` | `output` | Episode artifacts root |
| `DATA_DIR` | `data` | Persistent data (dedup DB, topic logs) |
| `LOG_DIR` | `logs` | Log files |
| `DEDUP_TTL_DAYS` | `30` | Days before a story is eligible again |

### YouTube (English channel)

| Variable | Default | Description |
|---|---|---|
| `YOUTUBE_CLIENT_SECRETS` | `config/client_secrets.json` | OAuth credentials |
| `YOUTUBE_TOKEN_FILE` | `config/token.pickle` | Cached OAuth token |
| `YOUTUBE_CATEGORY_ID` | `28` | YouTube category (28 = Science & Tech) |
| `YOUTUBE_PRIVACY` | `public` | `public` / `unlisted` / `private` |

### YouTube (Russian channel)

| Variable | Default | Description |
|---|---|---|
| `RU_ENABLED` | `false` | Set `true` to enable Russian variant |
| `RU_ELEVENLABS_VOICE_ID` | — | ElevenLabs voice ID for Russian |
| `RU_ELEVENLABS_MODEL` | `eleven_multilingual_v2` | Multilingual TTS model |
| `RU_YOUTUBE_CLIENT_SECRETS` | `config/client_secrets_ru.json` | Russian channel OAuth credentials |
| `RU_YOUTUBE_TOKEN_FILE` | `config/token_ru.pickle` | Russian channel OAuth token |

### TikTok

| Variable | Default | Description |
|---|---|---|
| `TIKTOK_ENABLED` | `false` | Set `true` to enable TikTok posting |
| `TIKTOK_CLIENT_KEY` | — | TikTok developer app client key |
| `TIKTOK_CLIENT_SECRET` | — | TikTok developer app client secret |
| `TIKTOK_TOKEN_FILE` | `config/tiktok_token.json` | OAuth token file path |
| `TIKTOK_PRIVACY` | `PUBLIC_TO_EVERYONE` | `PUBLIC_TO_EVERYONE` / `MUTUAL_FOLLOW_FRIENDS` / `SELF_ONLY` |

---

## GitHub Actions deployment

### Secrets (Settings → Secrets → Actions)

| Secret | How to obtain |
|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `ELEVENLABS_API_KEY` | [elevenlabs.io](https://elevenlabs.io) |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice library |
| `YOUTUBE_CLIENT_SECRETS_B64` | `base64 -w0 config/client_secrets.json` |
| `YOUTUBE_TOKEN_PICKLE_B64` | `base64 -w0 config/token.pickle` |
| `YOUTUBE_CLIENT_SECRETS_RU_B64` | `base64 -w0 config/client_secrets_ru.json` |
| `YOUTUBE_TOKEN_PICKLE_RU_B64` | `base64 -w0 config/token_ru.pickle` |
| `RU_ELEVENLABS_VOICE_ID` | ElevenLabs voice library (Russian voice) |
| `TIKTOK_CLIENT_KEY` | TikTok developer portal |
| `TIKTOK_CLIENT_SECRET` | TikTok developer portal |
| `TIKTOK_TOKEN_JSON_B64` | `base64 -w0 config/tiktok_token.json` |

### Variables (Settings → Variables → Actions)

| Variable | Value | Description |
|---|---|---|
| `RU_ENABLED` | `true` / `false` | Enable Russian variant |
| `TIKTOK_ENABLED` | `true` / `false` | Enable TikTok posting |

### Persistent caches (automatic)

| Cache key | Contents |
|---|---|
| `seen-stories-*` | `data/seen_stories.db` — daily dedup database |
| `tiktok-token-*` | `config/tiktok_token.json` — refreshed TikTok tokens |
| `weekly-topics-*` | `data/weekly_topics_*.json` — per-tool topic rotation logs |

> **Token rotation note:** The TikTok cache key rolls each run. When refreshed tokens are written back to disk, the next run's cache restores the latest version. On a cache miss (first run or cache eviction) the workflow falls back to decoding `TIKTOK_TOKEN_JSON_B64`.

> **YouTube token note:** After regenerating YouTube OAuth tokens, re-encode them (`base64 -w0 config/token.pickle`) and update the GitHub secret.

---

## Workflow schedule

```
daily.yml   → every day  08:00 UTC
weekly.yml  → Monday     09:00 UTC  (Claude tutorial)
            → Wednesday  09:00 UTC  (ChatGPT tutorial)
            → Friday     09:00 UTC  (Gemini tutorial)
```

Both workflows also have a **manual trigger** (`workflow_dispatch`) with optional overrides:
- `daily.yml` — no inputs (just runs)
- `weekly.yml` — `tool` (claude/chatgpt/gemini), `topic` (override rotation), `skip_upload`

---

## Cost estimate

### Daily news episode

| Service | Cost | Notes |
|---|---|---|
| OpenAI GPT-4o-mini | ~$0.05 | Script |
| OpenAI DALL-E 3 | ~$0.32 | 8 images × $0.04 |
| ElevenLabs TTS | ~$0.60 | ~15 min EN narration |
| ElevenLabs TTS (RU) | ~$0.60 | ~15 min RU narration |
| **Daily total** | **~$0.65 EN / ~$1.25 EN+RU** | |

### Weekly tutorial episode (per video)

| Service | Cost | Notes |
|---|---|---|
| OpenAI GPT-4o-mini | ~$0.06 | Script (~2,200 words) |
| OpenAI DALL-E 3 | ~$0.28 | 7 images × $0.04 |
| ElevenLabs TTS | ~$0.75 | ~20 min EN narration |
| ElevenLabs TTS (RU) | ~$0.75 | ~20 min RU narration |
| **Per tutorial** | **~$1.10 EN / ~$1.85 EN+RU** | |

### Monthly total (estimate)

| Config | Est. monthly cost |
|---|---|
| Daily EN only | ~$20 |
| Daily EN + RU | ~$38 |
| Daily + Weekly EN only | ~$23 |
| Daily + Weekly EN + RU | ~$44 |

---

## Docker

```bash
docker build -t ai-news .
docker run --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  ai-news
```

Mount `data/` for the dedup database and topic logs, and `config/` for OAuth credentials.
