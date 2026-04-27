# AI Video Assistance — Automated AI Content Pipeline

Fully automated production system that generates daily AI news videos, weekly tutorials, and breaking news alerts — posting to YouTube (English + Russian) and TikTok with zero manual intervention. **Status: Production Ready** (v1.0, 5.7K LOC, 29 modules).

> **Current Architecture**: 4 parallel pipelines (daily, weekly, breaking, digest) with multi-language support, intelligent content deduplication, and cached artifact reuse. Runs entirely on GitHub Actions with optional local scheduler.

## Content calendar

| Day | Pipeline | Output | Platforms | Runtime |
|---|---|---|---|---|
| Mon – Sun | Daily AI News | 15-min news video + Short | YouTube EN, YouTube RU, TikTok | ~20 min |
| **Breaking** | Triggered on major AI news | 2–5 min news short + full | YouTube EN, YouTube RU | ~10 min |
| Monday 09:00 UTC | Weekly Tutorial — Claude | 20-min tutorial + Short | YouTube EN, YouTube RU | ~30 min |
| Wednesday 09:00 UTC | Weekly Tutorial — ChatGPT | 20-min tutorial + Short | YouTube EN, YouTube RU | ~30 min |
| Friday 09:00 UTC | Weekly Tutorial — Gemini | 20-min tutorial + Short | YouTube EN, YouTube RU | ~30 min |
| Sunday 12:00 UTC | Weekly Digest | 35-min recap + Short | YouTube EN, YouTube RU | ~45 min |

**Estimated monthly output**: ~75 videos (EN) + ~75 videos (RU) + unlimited shorts

---

## Pipelines

### Daily news pipeline (runs 08:00 UTC daily)

```
scrape → deduplicate → script → voice → video → shorts → thumbnail → upload
                                                                    └── TikTok (short)
└─ if RU_ENABLED: translate → ru-voice → reassemble → ru-shorts → ru-thumbnail → ru-upload
```

**Steps:**
1. **Scrape** — Multi-source AI news aggregation:
   - **arXiv** — Computer Science (cs.AI, cs.LG, cs.CL, cs.CV) with automatic weekly trend analysis
   - **HuggingFace Daily Papers** — Ranked by popularity + downloads
   - **Hacker News** — AI-tagged stories (past 24h)
   - **Official blogs** — OpenAI, Anthropic, Google DeepMind, Microsoft Research, Meta AI, Mistral, DeepSeek, xAI (automatic RSS discovery)
   - **Scoring**: Official company posts scored 5.0 (guaranteed inclusion); community sources capped at ~2.0

2. **Deduplicate** — SQLite rolling window (30 days, configurable) prevents story recycling across runs

3. **Script** — GPT-4o-mini generates structured narration:
   - Title + hook (engaging opener)
   - 8 scenes (configurable, default ~2,200 words)
   - Per-scene: heading, description, AI-generated image prompt
   - Metadata: chapters, tags, YouTube description

4. **Voice** — ElevenLabs TTS synthesizes narration:
   - Per-scene MP3 files (sample rate 24 kHz)
   - Narration duration drives all video timing
   - Supports 30+ languages (Russian: `eleven_multilingual_v2` model)

5. **Video** — DALL-E 3 images + Ken Burns animation:
   - 1 image per scene (~$0.04 each)
   - 1792×1024 landscape (512px horizontal pan room, 304px vertical)
   - Slow zoom-pan animation (Ken Burns) synced to narration duration
   - Scene title cards (dark navy background, sky-blue accent bars) inserted before each scene

6. **Shorts** — 9:16 vertical crop (1080×1920) of hook scene:
   - Animated caption burn-in with wrapping text
   - Auto-scaled to video dimensions
   - TikTok/Instagram Reels compatible

7. **Thumbnail** — A/B variant selection:
   - 16 candidate frames extracted from final video
   - Scored for colorfulness + contrast (local maxima detection)
   - Title overlay using PIL (DejaVu Sans Bold, 48pt)
   - Top candidate uploaded via YouTube Thumbnails API

8. **Publish** — Multi-platform release:
   - **YouTube** — Long video (1280×720, 24fps, h.264, AAC audio)
   - **TikTok** — Short video (using TikTok Content Posting API v2)
   - Usage tracking (hook scoring for future recommendations)

9. **Russian variant** (optional, runs in parallel if `RU_ENABLED=true`):
   - Full translation of script via GPT-4o (context-aware, preserving tone)
   - Re-record narration with Russian voice
   - Reassemble video (same DALL-E images reused)
   - Generate Russian shorts + thumbnail overlay
   - Upload to Russian YouTube channel

**Caching**: All artifacts cached under `output/YYYY-MM-DD/`; safe to re-run after any failure

---

### Weekly tutorial pipeline (Mon/Wed/Fri 09:00 UTC)

```
pick_topic → script → voice → video → shorts → thumbnail → upload
└─ if RU_ENABLED: translate → ru-voice → reassemble → ru-shorts → ru-thumbnail → ru-upload
```

**Features:**
- **Topic rotation** — Each tool (Claude, ChatGPT, Gemini) maintains independent pool (~33–36 topics):
  - Topics tracked in `data/weekly_topics_{tool}.json`
  - Auto-cooldown: 26-week (~6-month) grace period before repeat
  - Manual override via `--topic` CLI flag for special episodes

- **Tool-specific scripting** — GPT-4o-mini writes tailored tutorial:
  - 7 scenes (~2,200 words)
  - Beginner-friendly explanations
  - Real-world use cases
  - Code examples (if applicable)

- **Screenshot capture** (weekly only):
  - Runs headless browser (Playwright) for Claude/ChatGPT/Gemini UI capture
  - Can override DALL-E with real screenshots when available
  - Use `--tool claude` to enable

- **Publishing** — English and Russian variants to their respective channels
- **Output**: `output/weekly/{tool}/YYYY-MM-DD/`

---

### Breaking news pipeline (triggered on demand)

```
detect_breaking → script → voice → video → shorts → thumbnail → upload
                                                            └─ mark_featured
```

**How it works:**
1. **Detection** — `breaking_detector.py` checks hourly:
   - Monitors official blogs + top arXiv posts
   - Triggers on keyword matches + engagement signals
   - Stores candidate in `data/breaking_current.json`

2. **Scripting** — Fast turnaround (single scene)
3. **Publishing** — YouTube long-form + shorts
4. **Featured tracking** — Marks stories as "featured" in dedup DB

**Runtimes**: ~10 minutes (vs. 20 min for daily)

---

### Sunday digest pipeline (12:00 UTC Sundays)

```
collect_week_scripts → digest_script → voice → video → shorts → thumbnail → upload
```

**Features:**
- **Weekly recap** — Combines Mon–Fri daily scripts into 1 coherent 35-min video
- **Automatic collection** — Pulls best daily scripts from the week
- **Digest scripting** — GPT-4o-mini synthesizes a recap narrative (~2,500 words)
- **Output**: `output/digest/YYYY-MM-DD/`

---

## Project layout

```
├── src/
│   ├── main.py                    # Daily pipeline orchestrator
│   ├── weekly_main.py             # Weekly tutorial pipeline orchestrator
│   ├── digest_main.py             # Sunday digest pipeline orchestrator
│   ├── breaking_main.py           # Breaking news pipeline executor
│   ├── breaking_detector.py       # Hourly breaking news detector
│   ├── scraper.py                 # arXiv, HuggingFace, Hacker News, official RSS (422 LOC)
│   ├── script_generator.py        # GPT-4o-mini daily script generation
│   ├── weekly_script_generator.py # Per-tool topic rotation + GPT tutorials (408 LOC)
│   ├── digest_script_generator.py # Weekly recap script synthesis
│   ├── breaking_script_generator.py # Single-scene breaking news script
│   ├── voice_generator.py         # ElevenLabs TTS with retry logic
│   ├── image_generator.py         # DALL-E 3 + Ken Burns animation
│   ├── video_generator.py         # MoviePy assembly + PIL title cards (254 LOC)
│   ├── shorts_generator.py        # 9:16 crop + animated captions
│   ├── weekly_shorts_generator.py # Tutorial-specific shorts
│   ├── thumbnail_generator.py     # Frame scoring + title overlay
│   ├── thumbnail_ab.py            # A/B variant selection (365 LOC)
│   ├── youtube_uploader.py        # OAuth2 YouTube upload + metadata
│   ├── tiktok_uploader.py         # TikTok Content Posting API v2
│   ├── deduplicator.py            # SQLite cross-day story dedup
│   ├── translator.py              # GPT-powered EN↔RU translation
│   ├── subtitle_generator.py      # SRT subtitle generation
│   ├── screenshot_capturer.py     # Playwright headless browser
│   ├── infographic_generator.py   # Visual data cards (399 LOC)
│   ├── hook_selector.py           # Hook/thumbnail usage tracking
│   ├── scheduler.py               # APScheduler local daemon
│   └── __init__.py
├── config/
│   ├── settings.py                # Centralized config (env vars, paths)
│   ├── client_secrets.json        # YouTube OAuth (English)
│   ├── client_secrets_ru.json     # YouTube OAuth (Russian)
│   ├── token.pickle               # YouTube cached token (English)
│   ├── token_ru.pickle            # YouTube cached token (Russian)
│   └── tiktok_token.json          # TikTok OAuth token
├── data/
│   ├── seen_stories.db            # Dedup SQLite database (auto-created)
│   ├── breaking_seen.json         # Breaking news dedup tracker
│   ├── weekly_topics_claude.json  # Claude topic rotation log
│   ├── weekly_topics_chatgpt.json # ChatGPT topic rotation log
│   ├── weekly_topics_gemini.json  # Gemini topic rotation log
│   └── *.json                     # Other tracking files
├── output/
│   ├── YYYY-MM-DD/                # Daily pipeline output
│   │   ├── news.json              # Scraped items
│   │   ├── script.json            # Generated script
│   │   ├── script_ru.json         # Russian translation
│   │   ├── audio/                 # Scene MP3s (EN)
│   │   ├── audio_ru/              # Scene MP3s (RU)
│   │   ├── clips/                 # Ken Burns video clips
│   │   ├── images/                # DALL-E images
│   │   ├── final_video.mp4        # Main video (1280×720, h.264)
│   │   ├── shorts.mp4             # TikTok short (1080×1920)
│   │   ├── thumbnail.jpg          # Selected thumbnail
│   │   └── run.log                # Execution log
│   ├── weekly/
│   │   └── {tool}/YYYY-MM-DD/     # Per-tool tutorial output
│   ├── digest/
│   │   └── YYYY-MM-DD/            # Sunday digest output
│   └── breaking/
│       └── YYYY-MM-DD-HHMM/       # Breaking news output
├── logs/
│   └── *.log                      # Pipeline execution logs
├── tests/                         # Unit tests (pytest suite, 60%+ coverage target)
├── requirements.txt
├── Dockerfile
├── README.md                      # This file
├── IMPROVEMENTS_RECOMMENDATIONS.md # Code quality analysis (see below)
└── .github/workflows/
    ├── daily.yml                  # Runs daily at 08:00 UTC
    ├── weekly.yml                 # Runs Mon/Wed/Fri at 09:00 UTC
    ├── breaking.yml               # Runs on breaking news detection
    └── digest.yml                 # Runs Sunday at 12:00 UTC
```

**Code Statistics**:
- **Total LOC**: ~5,700
- **Largest modules**: scraper (422), weekly_script_generator (408), infographic_generator (399), thumbnail_ab (365), main (348)
- **Test coverage**: 0% (see [IMPROVEMENTS_RECOMMENDATIONS.md](IMPROVEMENTS_RECOMMENDATIONS.md) for roadmap)

---

## Setup

### Prerequisites

- Python 3.10+
- FFmpeg (for video encoding)
- ~50 MB disk per daily episode (cached)

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Or use Docker (see below)
```

### Installation

```bash
# 1. Clone repo
git clone https://github.com/XenaSuv/AI_video_assistance.git
cd AI_video_assistance

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create environment file
cp .env.example .env
# Edit .env and fill in your API keys (see Environment variables section)
```

### API Credentials Setup

#### OpenAI (GPT-4o-mini for scripts, DALL-E 3 for images)

1. Get key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Add to `.env`:
   ```bash
   OPENAI_API_KEY=sk-...
   OPENAI_MODEL=gpt-4o-mini  # or gpt-4o for higher quality
   ```

#### ElevenLabs (Text-to-speech)

1. Sign up at [elevenlabs.io](https://elevenlabs.io)
2. Get API key from dashboard
3. Choose a voice ID (e.g., "Rachel" = Gfpl8Yo74Is0W6cPUWWT)
4. Add to `.env`:
   ```bash
   ELEVENLABS_API_KEY=sk_...
   ELEVENLABS_VOICE_ID=Gfpl8Yo74Is0W6cPUWWT
   ELEVENLABS_MODEL=eleven_turbo_v2_5  # Fast, low-cost model
   ```

#### YouTube OAuth (one-time per channel)

**English channel:**
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create new project, enable **YouTube Data API v3**
3. Create OAuth 2.0 Desktop client credentials
4. Download JSON → save as `config/client_secrets.json`
5. Run auth flow:
   ```bash
   python src/youtube_uploader.py --auth
   # → Opens browser, saves token to config/token.pickle
   ```

**Russian channel** (if `RU_ENABLED=true`):
```bash
YOUTUBE_CLIENT_SECRETS=config/client_secrets_ru.json \
YOUTUBE_TOKEN_FILE=config/token_ru.pickle \
python src/youtube_uploader.py --auth
```

#### TikTok OAuth (one-time)

1. Create developer app at [developers.tiktok.com](https://developers.tiktok.com)
   - Request scopes: `video.publish`, `video.upload`, `user.info.basic`
2. Get Client Key and Client Secret
3. Run interactive auth:
   ```bash
   export TIKTOK_CLIENT_KEY=your_key
   export TIKTOK_CLIENT_SECRET=your_secret
   python src/tiktok_uploader.py --auth
   # → Opens browser, saves token to config/tiktok_token.json
   ```
4. For GitHub Actions, encode token:
   ```bash
   base64 -w0 config/tiktok_token.json > /tmp/token_b64.txt
   # Copy output to GitHub secret TIKTOK_TOKEN_JSON_B64
   ```

---

## Running locally

### Daily news pipeline

```bash
# Full run (scrape → script → voice → video → upload)
python src/main.py

# Build only, skip uploads
python src/main.py --skip-upload

# Dry-run (scrape only, no API calls)
python src/main.py --dry-run
```

### Weekly tutorial pipeline

```bash
# Claude tutorial (Monday)
python src/weekly_main.py --tool claude

# ChatGPT tutorial (Wednesday)
python src/weekly_main.py --tool chatgpt

# Gemini tutorial (Friday)
python src/weekly_main.py --tool gemini

# Custom topic override
python src/weekly_main.py --tool claude --topic "Building RAG Systems"

# Build only
python src/weekly_main.py --tool claude --skip-upload

# With real screenshots (instead of DALL-E)
python src/weekly_main.py --tool claude  # Auto-detects and uses screenshots if available
```

### Breaking news pipeline

```bash
# Detector runs hourly in GitHub Actions, but can run manually:
python src/breaking_detector.py --check
# → Writes breaking news item to data/breaking_current.json if found

# Then publish it:
python src/breaking_main.py --item data/breaking_current.json
python src/breaking_main.py --item data/breaking_current.json --skip-upload
```

### Sunday digest pipeline

```bash
# This Sunday's digest
python src/digest_main.py

# Specific Sunday (ISO date)
python src/digest_main.py --date 2026-04-27

# Build only
python src/digest_main.py --skip-upload
```

### Local scheduler (daemon)

```bash
# Runs daily pipelines at 08:00 UTC
python src/scheduler.py
```

Logs are written to `logs/` directory with timestamps.

---

## Testing

```bash
# Run test suite (pytest, 60%+ coverage target)
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/ --cov-report=html
# Opens htmlcov/index.html for detailed coverage
```

Currently there are **no tests** (coverage = 0%). See [IMPROVEMENTS_RECOMMENDATIONS.md](IMPROVEMENTS_RECOMMENDATIONS.md#1-no-test-coverage) for roadmap.

---

## Environment variables

### Required (no defaults)

| Variable | Description | Example |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key (scripts + DALL-E) | `sk-...` |
| `ELEVENLABS_API_KEY` | ElevenLabs API key (TTS) | `sk_...` |

### Core Settings

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | GPT model (gpt-4o for higher quality) |
| `ELEVENLABS_VOICE_ID` | `Gfpl8Yo74Is0W6cPUWWT` | Voice ID for English narration |
| `ELEVENLABS_MODEL` | `eleven_turbo_v2_5` | TTS model (faster, cheaper) |
| `SCRIPT_TARGET_WORDS` | `2200` | Target narration length (words) |
| `DEDUP_TTL_DAYS` | `30` | Days before story is eligible again |
| `DAILY_RUN_HOUR_UTC` | `8` | Scheduled daily run time (UTC hour) |

### Paths

| Variable | Default | Description |
|---|---|---|
| `OUTPUT_DIR` | `output` | Video/artifact storage root |
| `DATA_DIR` | `data` | Persistent data (dedup DB, topic logs) |
| `LOG_DIR` | `logs` | Log file directory |
| `YOUTUBE_CLIENT_SECRETS` | `config/client_secrets.json` | YouTube OAuth credentials (English) |
| `YOUTUBE_TOKEN_FILE` | `config/token.pickle` | YouTube OAuth token cache (English) |

### Video Encoding (Advanced)

| Variable | Default | Description |
|---|---|---|
| `VIDEO_FPS` | `24` | Frames per second |
| `VIDEO_CODEC` | `libx264` | FFmpeg codec |
| `VIDEO_BITRATE` | `6M` | Output bitrate |
| `VIDEO_WIDTH` | `1280` | Output width (pixels) |
| `VIDEO_HEIGHT` | `720` | Output height (pixels) |
| `FONT_PATH_BOLD` | `/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf` | Title font path |

### YouTube (English channel)

| Variable | Default | Description |
|---|---|---|
| `YOUTUBE_CATEGORY_ID` | `28` | YouTube category (28 = Science & Tech) |
| `YOUTUBE_PRIVACY` | `public` | Visibility (`public` / `unlisted` / `private`) |

### YouTube (Russian channel)

| Variable | Default | Description |
|---|---|---|
| `RU_ENABLED` | `false` | Enable Russian variant |
| `RU_ELEVENLABS_VOICE_ID` | `TUQNWEvVPBLzMBSVDPUA` | Russian voice ID |
| `RU_ELEVENLABS_MODEL` | `eleven_multilingual_v2` | Multilingual TTS model |
| `RU_YOUTUBE_CLIENT_SECRETS` | `config/client_secrets_ru.json` | Russian OAuth credentials |
| `RU_YOUTUBE_TOKEN_FILE` | `config/token_ru.pickle` | Russian OAuth token cache |

### TikTok

| Variable | Default | Description |
|---|---|---|
| `TIKTOK_ENABLED` | `false` | Enable TikTok posting |
| `TIKTOK_CLIENT_KEY` | — | TikTok developer client key |
| `TIKTOK_CLIENT_SECRET` | — | TikTok developer client secret |
| `TIKTOK_TOKEN_FILE` | `config/tiktok_token.json` | OAuth token file |
| `TIKTOK_PRIVACY` | `PUBLIC_TO_EVERYONE` | Visibility setting |

### Example `.env` file

```bash
# Required
OPENAI_API_KEY=sk-proj-...
ELEVENLABS_API_KEY=sk_...

# Recommended
OPENAI_MODEL=gpt-4o-mini
ELEVENLABS_VOICE_ID=Gfpl8Yo74Is0W6cPUWWT
ELEVENLABS_MODEL=eleven_turbo_v2_5

# Optional (Russian)
RU_ENABLED=false
RU_ELEVENLABS_VOICE_ID=TUQNWEvVPBLzMBSVDPUA

# Optional (TikTok)
TIKTOK_ENABLED=false
# TIKTOK_CLIENT_KEY=...
# TIKTOK_CLIENT_SECRET=...
```

---

## GitHub Actions deployment

### Workflow triggers

| Workflow | Trigger | Schedule |
|---|---|---|
| `daily.yml` | Automatic + manual | 08:00 UTC daily |
| `weekly.yml` | Automatic + manual | Mon/Wed/Fri 09:00 UTC |
| `digest.yml` | Automatic + manual | Sunday 12:00 UTC |
| `breaking.yml` | Automatic + manual | On demand (hourly check) |

### GitHub secrets setup

**Settings → Secrets → Actions**

| Secret | How to obtain | Example |
|---|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) | `sk-...` |
| `ELEVENLABS_API_KEY` | [elevenlabs.io/api](https://elevenlabs.io/api) | `sk_...` |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice library | `Gfpl8Yo74Is0W6cPUWWT` |
| `RU_ELEVENLABS_VOICE_ID` | ElevenLabs voice library (Russian) | `TUQNWEvVPBLzMBSVDPUA` |
| `YOUTUBE_CLIENT_SECRETS_B64` | `base64 -w0 config/client_secrets.json` | Base64-encoded JSON |
| `YOUTUBE_TOKEN_PICKLE_B64` | `base64 -w0 config/token.pickle` | Base64-encoded pickle |
| `YOUTUBE_CLIENT_SECRETS_RU_B64` | Russian channel OAuth | Base64-encoded JSON |
| `YOUTUBE_TOKEN_PICKLE_RU_B64` | Russian channel token | Base64-encoded pickle |
| `TIKTOK_CLIENT_KEY` | [developers.tiktok.com](https://developers.tiktok.com) | TikTok dev portal |
| `TIKTOK_CLIENT_SECRET` | [developers.tiktok.com](https://developers.tiktok.com) | TikTok dev portal |
| `TIKTOK_TOKEN_JSON_B64` | `base64 -w0 config/tiktok_token.json` | Base64-encoded JSON |

### GitHub variables setup

**Settings → Variables → Actions**

| Variable | Value | Notes |
|---|---|---|
| `RU_ENABLED` | `true` or `false` | Enable/disable Russian variant |
| `TIKTOK_ENABLED` | `true` or `false` | Enable/disable TikTok posting |

### Workflow outputs

Each workflow creates artifacts (logs, scripts, video files):

```bash
# Access in GitHub: Actions → [workflow run] → Artifacts
# Retention: 30 days
# Contents:
#   - logs/run.log          # Full execution log
#   - output/*/script.json  # Generated script
#   - output/*/thumbnail.* # Selected thumbnail
```

### Caching strategy

| Cache | Contents | TTL | Key pattern |
|---|---|---|---|
| `seen-stories-*` | Dedup database | ~1 week | `seen-stories-${{ github.run_id }}` |
| `weekly-topics-*` | Topic rotation logs | ~1 week | `weekly-topics-${{ github.run_id }}` |
| `tiktok-token-*` | TikTok OAuth token (refreshed) | ~1 week | `tiktok-token-${{ github.run_id }}` |

> **Token refresh**: TikTok access tokens (24h TTL) auto-refresh using cached refresh tokens (365-day TTL). On cache miss, workflows fall back to decoding `TIKTOK_TOKEN_JSON_B64`.

---

## Troubleshooting

### Common issues

#### NameError: name 'ImageFont' is not defined
- **Cause**: Pillow not installed
- **Fix**: `pip install Pillow>=10.4.0`

#### RateLimitError from OpenAI / ElevenLabs
- **Cause**: API quota exhausted
- **Fix**: Retry logic auto-handles with exponential backoff. Check logs for throttle timing.

#### YouTube upload fails ("Could not authenticate")
- **Cause**: Token expired or invalid
- **Fix**: Re-run auth flow:
  ```bash
  python src/youtube_uploader.py --auth
  base64 -w0 config/token.pickle  # Update GitHub secret
  ```

#### FFmpeg not found
- **Cause**: FFmpeg not installed
- **Fix**: `brew install ffmpeg` (macOS) or `sudo apt-get install ffmpeg` (Linux)

#### Script generation too short / too long
- **Fix**: Adjust `SCRIPT_TARGET_WORDS` (default 2200)

#### Video takes too long to encode
- **Fix**: Lower quality:
  ```bash
  VIDEO_BITRATE=4M  # Down from 6M
  VIDEO_FPS=24      # Keep or lower to 20
  ```

### Logs

- **Local runs**: `logs/YYYY-MM-DD.log` (auto-created)
- **GitHub Actions**: Download artifacts from workflow run page
- **Real-time monitoring**: `tail -f logs/$(date +%Y-%m-%d).log`

---

## Workflow schedule

### Daily (08:00 UTC)

```
08:00 UTC → daily.yml runs
  ├─ Scrapes latest AI news
  ├─ Generates script + voice
  ├─ Creates video + shorts
  └─ Uploads to YouTube + TikTok
```

### Weekly (Mon/Wed/Fri 09:00 UTC)

```
Monday 09:00 UTC   → Claude tutorial
Wednesday 09:00 UTC → ChatGPT tutorial
Friday 09:00 UTC   → Gemini tutorial
  (each runs independently, same pipeline structure)
```

### Sunday (12:00 UTC)

```
12:00 UTC → digest.yml runs
  ├─ Collects Mon–Fri daily scripts
  ├─ Synthesizes weekly recap narrative
  └─ Creates 35-min digest video
```

### Breaking (hourly check)

```
Every hour → breaking_detector.py checks for major AI news
  ├─ If found: breaking.yml publishes within 10 min
  ├─ Marks story as featured
  └─ Logs to breaking_seen.json dedup tracker
```

### Manual triggers

All workflows can be triggered manually from **Actions → [workflow] → Run workflow**:

```bash
# Daily: no inputs
# Weekly: --tool {claude|chatgpt|gemini}, --topic "override", --skip-upload
# Digest: --date {YYYY-MM-DD}, --skip-upload
# Breaking: --item data/breaking_current.json (auto-populated)
```

---

## Cost estimate

### API pricing (as of April 2026)

| Service | Cost | Factor |
|---|---|---|
| **GPT-4o-mini** | $0.15 / 1M input tokens | Script generation (~5K tokens) ≈ $0.0006 |
| **DALL-E 3** | $0.04 / image | 8 images/daily, 7 images/weekly |
| **ElevenLabs** | $0.003 / 1K characters | ~$0.60 per 15-min narration |

### Daily episode cost

| Component | Count | Unit Cost | Total |
|---|---|---|---|
| Script (GPT-4o-mini) | 1 | $0.0006 | $0.0006 |
| Images (DALL-E 3) | 8 | $0.04 | $0.32 |
| Narration (ElevenLabs EN) | 15 min | $0.04/min | $0.60 |
| Narration (ElevenLabs RU, if enabled) | 15 min | $0.04/min | $0.60 |
| **Daily subtotal** | — | — | **$0.93 (EN) / $1.53 (EN+RU)** |

### Weekly tutorial cost (per episode)

| Component | Count | Unit Cost | Total |
|---|---|---|---|
| Script (GPT-4o-mini) | 1 | $0.0008 | $0.0008 |
| Images (DALL-E 3) | 7 | $0.04 | $0.28 |
| Narration (ElevenLabs EN) | 20 min | $0.04/min | $0.80 |
| Narration (ElevenLabs RU, if enabled) | 20 min | $0.04/min | $0.80 |
| **Per tutorial** | — | — | **$1.09 (EN) / $1.89 (EN+RU)** |

### Monthly estimate (typical schedule)

| Configuration | Daily (30×) | Weekly (12×) | **Monthly** |
|---|---|---|---|
| EN only | $28 | $13 | **~$41** |
| EN + RU | $46 | $23 | **~$69** |
| Digest (Sun) + EN | +$10 | — | **~$51** |
| All (Daily + Weekly + Digest + EN+RU) | $46 | $23 | **~$98** |

> Digest cost: ~$1.30 per episode (same as daily, slightly higher for 35-min narration)

Mount `data/`, `config/`, `output/`, and `logs/` as volumes. Environment variables can be passed via `--env-file .env`.

---

## Architecture & Design Decisions

### Why these tools?

- **OpenAI GPT-4o-mini** — Cost-effective (~$0.15/1M tokens), fast script generation
- **DALL-E 3** — High-quality AI images, good for news/tutorials ($0.04/image vs. video generation ~$0.50)
- **ElevenLabs TTS** — Natural-sounding voices, supports 30+ languages, affordable ($0.003/1K chars)
- **MoviePy** — Pure Python, no ImageMagick dependency, easy deployment
- **Playwright** — Headless browser capture for weekly tutorial screenshots (Claude/ChatGPT/Gemini UI)

### Why SQLite for dedup?

- **Lightweight**: Single file, no server needed
- **Portable**: Works everywhere (local, Docker, GitHub Actions)
- **Efficient**: 30-day rolling window (configurable TTL)
- **Atomic**: No race conditions across concurrent runs

### Caching strategy

All artifacts cached under `output/YYYY-MM-DD/` (or `output/{pipeline_type}/{date}/`):
- **Safe to re-run**: Scripts, audio, video clips, thumbnails all checked before regeneration
- **Resume-friendly**: Pipeline can be interrupted and resumed without reprocessing
- **Cost-effective**: Avoids redundant API calls (~$0.30/episode saved on retry)

### Error handling

- **Retry logic**: Exponential backoff for OpenAI/ElevenLabs rate limits (implemented via `tenacity`)
- **Graceful degradation**: Missing images → fallback font on PIL errors
- **Non-fatal failures**: TikTok/subtitle generation failures don't halt pipeline
- **Detailed logging**: Every step logged to `run.log` with timestamps

### Known limitations

- **No real-time streaming**: Videos generated offline, uploaded post-production
- **Single content source priority**: Official blogs prioritized over community (configurable in `scraper.py`)
- **Language support**: Currently EN + RU; adding new languages requires translation logic
- **Thumbnail scoring**: Frame-based (colorfulness + contrast); doesn't account for faces/composition
- **No A/B testing**: Shorts always auto-crop from hook; no user testing integration

---

## Contributing

### Local development workflow

```bash
# 1. Clone repo
git clone https://github.com/XenaSuv/AI_video_assistance.git && cd AI_video_assistance

# 2. Create venv + install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Test a module
python -m pytest tests/ -v

# 4. Run pipeline locally
python src/main.py --skip-upload --dry-run

# 5. Submit PR with tests + docs
```

### Code structure

- **`src/main.py`** — Daily orchestrator (~350 LOC)
- **`src/scraper.py`** — News sources (~420 LOC) → extend for new sources
- **`src/script_generator.py`** — GPT prompts (~240 LOC) → customize tone/length
- **`src/image_generator.py`** — DALL-E prompts (~180 LOC) → different image styles
- **`src/video_generator.py`** — MoviePy assembly (~250 LOC) → customize transitions

### Before submitting a PR

- Run existing tests: `pytest tests/`
- Add tests for new features (target: 60%+ coverage)
- Check code style: `flake8 src/ --max-line-length=100`
- Update README if adding new features

---

## Roadmap

### Q2 2026 (Current)

- ✅ Daily + weekly + breaking pipelines
- ✅ Multi-language support (EN, RU)
- ✅ TikTok integration
- ⚠️ Zero test coverage → Add pytest suite (60%+)

### Q3 2026 (Planned)

- 📋 Checkpoint-based recovery (resume failed jobs)
- 📋 Abstract base pipeline class (reduce duplication)
- 📋 Language variant generalization (add support for more languages)
- 📋 Real-time analytics dashboard

### Q4 2026 (Exploration)

- 🔍 Async I/O for parallel operations
- 🔍 Hook/thumbnail ML scoring (instead of frame-based)
- 🔍 Real-time breaking news detection (use external service?)
- 🔍 Expand to other platforms (YouTube Shorts, Instagram Reels, LinkedIn)

See [IMPROVEMENTS_RECOMMENDATIONS.md](IMPROVEMENTS_RECOMMENDATIONS.md) for detailed code quality analysis and optimization opportunities.

---

## License

MIT — See LICENSE file

---

## Contact

- **GitHub**: [@XenaSuv](https://github.com/XenaSuv)
- **Issues**: [GitHub Issues](https://github.com/XenaSuv/AI_video_assistance/issues)
- **Email**: Open an issue or PR for questions

---

## Acknowledgments

- Inspired by: [NewsAI](https://github.com/search?q=news+video+ai), [AutoDub](https://autodub.com)
- Built with: OpenAI, ElevenLabs, MoviePy, Google APIs, TikTok API
- Deployed on: GitHub Actions
- Monitored via: loguru + manual review

---

**Last updated**: April 27, 2026  
**Status**: Production Ready (v1.0)  
**Maintenance**: Active development
