# CLAUDE.md — AI Video Assistance

Developer guide for working with this codebase. Read before making changes.

---

## What This Is

A **self-improving AI media system** that generates daily AI news videos, weekly tutorials, breaking news alerts, and weekly digests — publishing to YouTube (EN + RU) and TikTok automatically.

The key distinction: the system doesn't just generate content, it learns from audience retention data and adjusts future content strategy. Each pipeline run feeds metrics back into the learning loop.

**Monthly output**: ~150 long-form videos + unlimited Shorts.

---

## Architecture: Three Layers

```
News Sources
    ↓
┌─────────────────────────────────┐
│  Layer 1: Story Intelligence    │  What to say and why
│  editorial_brain, decision_engine,
│  viral_selector, hook_optimizer │
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  Layer 2: Narrative Engine      │  How to tell it
│  script_generator, humanizer_  │
│  agent, voice_generator,       │
│  video_generator, image_generator│
└────────────┬────────────────────┘
             ↓
┌─────────────────────────────────┐
│  Layer 3: Learning System       │  What actually worked
│  feedback_analyzer, drop_      │
│  predictor, thompson_bandit,   │
│  scene_bandit, decision_engine │
└─────────────────────────────────┘
             ↓
    YouTube / TikTok metrics
             ↓
        (back to Layer 1)
```

### Layer 1 — Story Intelligence

| Module | Responsibility |
|--------|---------------|
| `src/editorial_brain.py` | Story ranking, angle selection, persona assignment, scene planning |
| `src/viral_selector.py` | Scores stories for virality; filters for engagement potential |
| `src/hook_optimizer.py` | Learns which hooks retain viewers past 30s |
| `src/hook_mutation_engine.py` | Evolves high-performing hooks into new variants |
| `src/decision_engine.py` | Decay-weighted feedback → strategic decisions (angle/format weights) |
| `src/decision_engine_v2.py` | Advanced version with performance windows and channel-level strategy |
| `src/scraper.py` | Aggregates news: arXiv, HuggingFace, HackerNews, official blogs, RSS |

### Layer 2 — Narrative Engine

| Module | Responsibility |
|--------|---------------|
| `src/script_generator.py` | Structured narration from editorial plans; produces `VideoScript` |
| `src/humanizer_agent.py` | Breaks rigid AI structure, adds reactions and rhythm |
| `src/micro_hook_agent.py` | Inserts engagement hooks every 20–40s |
| `src/voice_generator.py` | ElevenLabs TTS with SSML pacing, 5-retry exponential backoff |
| `src/video_generator.py` | Clips + Ken Burns effects + music + intro/outro assembly |
| `src/image_generator.py` | DALL-E 3 primary, Pexels/Pixabay fallback, infographics |
| `src/subtitle_generator.py` | Auto-transcription from audio (non-fatal if fails) |
| `src/presenter.py` | D-ID AI avatar integration (optional) |
| `src/translator.py` | Script translation EN → RU |

### Shorts Stack (primary engine)

| Module | Role |
|--------|------|
| `src/shorts_pipeline.py` | **PRIMARY** — standalone pipeline, entry point; run directly or call `ShortsPipeline().run()` |
| `src/shorts_engine_v2.py` | **PRIMARY** — editorial + hook-generation layer; produces `ShortScript` objects |
| `src/shorts_experiment_engine.py` | **PRIMARY** — learning layer; stores per-experiment metrics, feeds best hooks back to daily pipeline |
| `src/shorts_beat_engine.py` | Sub-component of v2; per-beat execution plan for 16-second Shorts |
| `src/shorts_generator.py` | **DEPRECATED** — legacy "clip-the-video" approach; kept for backward-compat callers (breaking_main, weekly_main, topic_main, language_variant). New code must not import from this module. |

### Layer 3 — Learning System

| Module | Responsibility |
|--------|---------------|
| `src/feedback_analyzer.py` | Parses retention curves, detects drop points, scores hooks |
| `src/youtube_analytics.py` | Fetches real metrics from YouTube Data API v3 |
| `src/tiktok_analytics.py` | TikTok engagement tracking |
| `src/drop_predictor.py` | Predicts retention drop risk per scene before publishing |
| `src/retention_correction_engine.py` | Detects drop points, suggests scene-level fixes |
| `src/thompson_bandit.py` | Beta-posterior Thompson Sampling for variant selection |
| `src/scene_bandit.py` | Multi-armed bandit over scene policies |
| `src/ab_testing_engine.py` | Sequential A/B testing with CTR × retention scoring |
| `src/bandit_engine.py` | UCB1 multi-armed bandit for continuous optimization |

### Supporting Infrastructure

| Module | Responsibility |
|--------|---------------|
| `src/checkpoint.py` | Resumable pipeline execution — reruns reuse cached artifacts |
| `src/deduplicator.py` | SQLite DB, 30-day story deduplication |
| `src/quality_gate.py` | Hard/soft checks; blocks publish on failures |
| `src/pipeline_observer.py` | Step-by-step execution tracking |
| `src/cost_tracker.py` | API usage ledger per run |
| `src/slack_notifier.py` | Async Slack notifications on success/failure |
| `src/performance_tracker.py` | Logs video metadata for learning loop |
| `config/settings.py` | Frozen dataclass; all env vars loaded once at import time |
| `src/shared_types.py` | Cross-module dataclasses (`ContentStrategy`, `PerformanceStats`, etc.) |
| `src/constants.py` | `VARIANT_TYPE_DELTAS`, `RateLimitMixin` |

---

## Directory Layout

```
AI_video_assistance/
├── src/                    # All source modules
├── config/
│   ├── settings.py         # Settings dataclass (frozen) — env-var driven
│   ├── client_secrets.json # YouTube OAuth2 app credentials (EN channel)
│   └── token.pickle        # YouTube OAuth2 token (generated on first auth)
├── tests/
│   ├── conftest.py         # Fixtures + heavy-dep stubs (read this first)
│   └── test_*.py           # 25 test files
├── data/
│   ├── performance_store.json   # Video performance records
│   └── scene_performance.json  # Per-scene retention metrics
├── source/                 # Static media assets
│   ├── ai-news-intro.mp4
│   ├── ai-news-outro.mp4
│   └── background_music.mp3
├── assets/
│   └── avatar.png          # D-ID presenter image
├── output/                 # Generated videos (per-date subdirectories)
├── logs/                   # Loguru logs (10 MB rotation, per run)
├── Dockerfile
├── requirements.txt
└── .github/workflows/      # CI + 5 production pipelines
```

---

## Setup

### Prerequisites

- Python 3.12
- FFmpeg: `sudo apt-get install ffmpeg`
- ImageMagick: `sudo apt-get install imagemagick fonts-dejavu`

### Install

```bash
git clone https://github.com/XenaSuv/AI_video_assistance.git
cd AI_video_assistance
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# ── Required ──────────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-...
ELEVENLABS_API_KEY=...

# ── YouTube (EN channel) ──────────────────────────────────────────────────────
# client_secrets.json: download from Google Cloud Console (OAuth2 Desktop app)
# token.pickle: generated automatically on first run (requires browser auth)
YOUTUBE_CLIENT_SECRETS=config/client_secrets.json
YOUTUBE_TOKEN_FILE=config/token.pickle
YOUTUBE_PRIVACY=public           # public | unlisted | private

# ── Russian variant (optional) ────────────────────────────────────────────────
RU_ENABLED=true
RU_ELEVENLABS_VOICE_ID=TUQNWEvVPBLzMBSVDPUA
RU_YOUTUBE_CLIENT_SECRETS=config/client_secrets_ru.json
RU_YOUTUBE_TOKEN_FILE=config/token_ru.pickle

# ── Stock media ───────────────────────────────────────────────────────────────
PEXELS_API_KEY=...               # Primary B-roll source
PIXABAY_API_KEY=...              # Fallback B-roll
UNSPLASH_ACCESS_KEY=...          # Fallback stock photos
STABILITY_API_KEY=...            # Breaking news clips (cheaper than DALL-E)

# ── TikTok (optional) ─────────────────────────────────────────────────────────
TIKTOK_ENABLED=false
TIKTOK_CLIENT_KEY=...
TIKTOK_CLIENT_SECRET=...

# ── AI Presenter (optional) ───────────────────────────────────────────────────
# DID_API_KEY: get "Basic <key>" from d-id.com, set just <key> here
PRESENTER_ENABLED=false
DID_API_KEY=...
PRESENTER_AVATAR_PATH=assets/avatar.png

# ── Channel branding ──────────────────────────────────────────────────────────
CHANNEL_NAME=AI Today
CHANNEL_HANDLE=@AIToday
CHANNEL_CTA=Subscribe · Daily AI News

# ── Pipeline tuning ───────────────────────────────────────────────────────────
OPENAI_MODEL=gpt-4o-mini         # or gpt-4o for higher quality
SCRIPT_TARGET_WORDS=2200
DAILY_RUN_HOUR_UTC=8
BACKGROUND_MUSIC_PATH=source/background_music.mp3
BACKGROUND_MUSIC_VOLUME=0.10
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

---

## Running Pipelines

All pipelines are **idempotent** — re-running reuses cached artifacts from `output/YYYY-MM-DD/`.

```bash
# Daily AI news (main pipeline)
python src/main.py

# Weekly tutorials
python src/weekly_main.py --tool claude     # Monday
python src/weekly_main.py --tool chatgpt   # Wednesday
python src/weekly_main.py --tool gemini    # Friday

# Breaking news (pass detected item JSON)
python src/breaking_main.py --item data/breaking_current.json

# Weekly digest
python src/digest_main.py

# Breaking news detector (checks sources, writes data/breaking_current.json)
python src/breaking_detector.py --check

# Automated scheduler (all pipelines on cron)
python src/scheduler.py
python src/scheduler.py --list   # Print next scheduled runs and exit

# Docker
docker build -t ai-video .
docker run --env-file .env ai-video
```

### Daily Pipeline Flow (`src/main.py`)

```
scrape_all()
  → deduplicate (SQLite, 30-day TTL)
  → pick_viral_news()
  → EditorialBrain.run()        ← uses DecisionEngine + FeedbackAnalyzer
  → generate_script()
  → HumanizerAgent.run()
  → MicroHookAgent.run()
  → quality_gate.run_gate()     ← blocks publish on hard failures
  → synthesize_script()         ← ElevenLabs TTS
  → build_video()               ← DALL-E + B-roll + Ken Burns + music
  → generate_thumbnail()
  → publish_episode()           ← YouTube upload + captions
  → [if RU_ENABLED] translate → ru-TTS → reassemble → ru-upload
  → save_result()               ← feeds learning loop
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_decision_engine.py -v

# With coverage (not yet in CI — run locally)
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Test Architecture

`tests/conftest.py` **must be read before writing new tests**. It:
1. Sets dummy env vars (`OPENAI_API_KEY=test-key-openai`) before any module import
2. Stubs all Google SDK modules and numpy as `MagicMock` to avoid install requirements
3. Provides shared fixtures: `fresh_story`, `feedback_history`, `high_performing_feedback`, etc.

**Well-tested modules**: `decision_engine`, `feedback_analyzer`, `drop_predictor`, `scraper`, `thompson_bandit`, `scene_bandit`, `ab_testing_engine`, `quality_gate`, `checkpoint`

**Untested modules** (gaps): `youtube_uploader`, `video_generator`, `voice_generator`, `humanizer_agent`, `breaking_main`

---

## Key Data Types

Defined in `src/shared_types.py`:

```python
ContentStrategy       # Output of DecisionEngine.decide()
                      # angle_weights, format_weights, exploration_rate, mode, confidence

HookOptimizationResult # Output of HookOptimizer.run()
                       # recommended_patterns, avoid_patterns, style_bias

MicroHookResult       # Output of MicroHookAgent.run()
InsertedHook          # Single micro-hook injected at a position
HumanizationResult    # Output of HumanizerAgent.run()
PerformanceStats      # Per-angle/format aggregated metrics
```

Core pipeline types in `src/script_generator.py`:
- `NewsItem` — scraped news story
- `Scene` — single video segment with narration + visual_prompt
- `VideoScript` — title + hook + list of scenes + metadata

---

## Configuration System

`config/settings.py` loads **all** env vars into a **frozen dataclass** at module import time. This means:

- Changing env vars after import has no effect
- Tests must set env vars **before** importing any project module (handled by `conftest.py`)
- The singleton is accessed as `from config import settings` everywhere

Key paths auto-created on startup: `output/`, `logs/`, `data/`.

---

## Learning Loop Details

### Hook Score
Calculated from 30-second retention. Target: > 0.6. Stored in `data/performance_store.json`.

### Angle Weights
`DecisionEngine` adjusts weights for 5 angles:
- `technical_breakthrough`, `industry_impact`, `threat_to_jobs`, `overhyped_vs_reality`, `what_this_means_for_you`

Weights use **exponential decay** — recent videos count more.

### Bandit Algorithms
- **Thompson Sampling** (`thompson_bandit.py`): Beta-posterior inference for variant selection. State persisted to JSON.
- **UCB1** (`bandit_engine.py`): Upper confidence bound for continuous optimization.
- **Scene Bandit** (`scene_bandit.py`): Multi-armed bandit over per-scene policies.

All bandits use `RateLimitMixin` from `src/constants.py` to prevent too-frequent switching.

### Quality Gate Tiers
- **Hard failures** → `QualityGateError` raised → publish blocked
  - Hook too short (<10 chars), too few scenes (<3), audio missing
- **Soft warnings** → logged, publish proceeds
  - Short scenes, low historical hook score

---

## GitHub Actions Workflows

| Workflow | Schedule | Timeout |
|----------|----------|---------|
| `ci.yml` | On push/PR | 10 min |
| `daily.yml` | Mon–Sat 08:00 UTC | 60 min |
| `weekly.yml` | Mon/Wed/Fri 09:00 UTC | 90 min |
| `breaking.yml` | Every 30 min, 24/7 | 180 min |
| `digest.yml` | Sun 10:00 UTC | 120 min |
| `topic.yml` | Tue/Thu 10:00 UTC | 90 min |

YouTube OAuth credentials are stored as base64-encoded GitHub Secrets (`YOUTUBE_TOKEN_PICKLE_B64`, `YOUTUBE_CLIENT_SECRETS_B64`) and decoded in each workflow.

---

## Known Gaps

- **No linting in CI** — `black`, `flake8`, `mypy` not enforced. Run locally before committing.
- **No coverage gate** — `pytest --cov` not in CI. Target is >80% for new modules.
- **`main.py` is large** (28.5K) — orchestration, caching, and variant logic are mixed. Extract to `PipelineOrchestrator` if adding features.
- **All I/O is synchronous** — ElevenLabs, DALL-E, and YouTube calls block. Refactor to `asyncio` before scaling to higher volume.
- **SQLite has no migrations** — if `deduplicator.py` schema changes, delete `data/dedup.db` and it recreates.
- **OAuth tokens in `config/*.pickle`** — do not commit these files. They are in `.gitignore`.

---

## Adding a New Pipeline

1. Create `src/your_main.py` modeled after `src/main.py`
2. Use `PipelineCheckpoint` for resumability
3. Use `PipelineObserver` for step tracking
4. Call `run_gate()` before any upload step
5. Call `save_result()` after successful upload to feed the learning loop
6. Add a test file in `tests/` — stub heavy deps via `conftest.py` pattern
7. Add a GitHub Actions workflow in `.github/workflows/`

## Adding a New Source to the Scraper

`src/scraper.py` has per-source functions that return `list[NewsItem]`. Add a new function and register it in `scrape_all()`. The caching layer is source+date keyed — new sources get cached automatically.
