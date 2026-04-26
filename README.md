# AI News Daily Pipeline

Automated daily pipeline that produces a 15-minute AI news video plus a YouTube Short — fully hands-off from scraping to publishing.

```
scrape → deduplicate → script → voice → video → shorts → thumbnail → upload
```

## How it works

1. **Scrape** — pulls the latest AI papers and posts from arXiv (cs.AI/LG/CL/CV), HuggingFace Daily Papers, and Hacker News
2. **Deduplicate** — filters out stories featured in the last 30 days (SQLite-backed, rolling window) so the same paper never appears twice in a month
3. **Script** — GPT-4o-mini writes a structured 15-min narration script with title, description, chapter markers, and tags
4. **Voice** — ElevenLabs TTS synthesises each scene's narration; durations drive video timing
5. **Video** — DALL-E 3 generates one image per scene; MoviePy animates each with a Ken Burns pan and assembles the final 16:9 video
6. **Shorts** — auto-crops the hook to 9:16 (1080×1920) with animated captions
7. **Thumbnail** — scores 16 candidate frames for colorfulness + contrast, overlays the episode title, uploads to YouTube via the thumbnails API
8. **Publish** — YouTube Data API v3 uploads both videos with chapters, tags, and the custom thumbnail

All steps are cached under `output/YYYY-MM-DD/` — safe to re-run after any failure.

## Project layout

```
├── src/
│   ├── main.py               # Pipeline orchestrator
│   ├── scraper.py            # arXiv + HuggingFace + HN
│   ├── script_generator.py   # GPT-4o-mini script + metadata
│   ├── voice_generator.py    # ElevenLabs TTS (per-scene mp3)
│   ├── image_generator.py    # DALL-E 3 images + Ken Burns clips
│   ├── video_generator.py    # MoviePy assembly + PIL chyrons
│   ├── shorts_generator.py   # 9:16 crop + caption burn-in
│   ├── thumbnail_generator.py# Frame scoring + title overlay
│   ├── youtube_uploader.py   # OAuth2 upload + thumbnail set
│   ├── deduplicator.py       # SQLite cross-day story dedup
│   └── scheduler.py          # APScheduler daemon
├── config/
│   └── settings.py           # Centralised config (env vars)
├── data/
│   └── seen_stories.db       # Dedup database (auto-created)
├── requirements.txt
├── Dockerfile
└── .github/workflows/daily.yml   # Runs 08:00 UTC daily
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill in API keys
```

**YouTube OAuth (one-time):**

1. Create a project in [Google Cloud Console](https://console.cloud.google.com), enable the YouTube Data API v3, and create an OAuth 2.0 Desktop client
2. Download `client_secrets.json` → place in `config/`
3. Run `python src/youtube_uploader.py --auth` to complete the OAuth flow (saves `config/token.pickle`)

**Test the pipeline:**

```bash
python src/main.py --dry-run     # scrape only, no API costs
python src/main.py --skip-upload # full build, no YouTube upload
python src/main.py               # full run
```

## Required API keys

| Service | Env var(s) | Notes |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | Used for GPT-4o-mini (script) and DALL-E 3 (images) |
| ElevenLabs | `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | Creator tier (~$22/mo) covers ~15 min/day |
| YouTube | `client_secrets.json` + `token.pickle` | OAuth desktop flow, one-time setup |

> RunwayML is no longer required — b-roll is generated with DALL-E 3.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required |
| `OPENAI_MODEL` | `gpt-4o-mini` | Script generation model |
| `ELEVENLABS_API_KEY` | — | Required |
| `ELEVENLABS_VOICE_ID` | Rachel | ElevenLabs voice ID |
| `ELEVENLABS_MODEL` | `eleven_turbo_v2_5` | TTS model |
| `YOUTUBE_CLIENT_SECRETS` | `config/client_secrets.json` | OAuth credentials path |
| `YOUTUBE_TOKEN_FILE` | `config/token.pickle` | Cached OAuth token path |
| `YOUTUBE_PRIVACY` | `public` | `public` / `unlisted` / `private` |
| `SCRIPT_TARGET_WORDS` | `2200` | Target narration length |
| `DEDUP_TTL_DAYS` | `30` | Days before a story becomes eligible again |
| `OUTPUT_DIR` | `output` | Episode artifacts root |
| `DATA_DIR` | `data` | Persistent data (dedup DB) |
| `LOG_DIR` | `logs` | Log files |

## Deploying

### GitHub Actions (recommended)

The included workflow runs at 08:00 UTC daily. Add these secrets to your repo (**Settings → Secrets → Actions**):

| Secret | How to get it |
|---|---|
| `OPENAI_API_KEY` | platform.openai.com |
| `ELEVENLABS_API_KEY` | elevenlabs.io |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice library |
| `YOUTUBE_CLIENT_SECRETS_B64` | `base64 -w 0 config/client_secrets.json` |
| `YOUTUBE_TOKEN_PICKLE_B64` | `base64 -w 0 config/token.pickle` |

The dedup database (`data/seen_stories.db`) is persisted between runs via `actions/cache` — no manual setup needed.

> **Note:** if you change the OAuth scopes (e.g. after pulling this repo fresh), regenerate the token and re-encode it: `python src/youtube_uploader.py --auth` then re-run the base64 command above.

### Local daemon

```bash
python src/scheduler.py    # runs daily at DAILY_RUN_HOUR_UTC (default 08:00 UTC)
```

### Docker

```bash
docker build -t ai-news .
docker run --env-file .env -v $(pwd)/data:/app/data ai-news
```

Mount `data/` as a volume so the dedup database persists across container restarts.

## Cost estimate (per daily episode)

| Service | Cost | Notes |
|---|---|---|
| OpenAI GPT-4o-mini | ~$0.05 | Script generation |
| OpenAI DALL-E 3 | ~$0.32 | 8 images × $0.04 |
| ElevenLabs TTS | ~$0.60 | ~15 min narration |
| YouTube API | free | |
| **Total** | **~$1.00/day (~$30/mo)** | Down from ~$10–25 with RunwayML |
