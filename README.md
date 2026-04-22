# AI News Daily Pipeline

Automated daily pipeline that produces 15-minute AI news videos plus YouTube Shorts:

1. **Scrape** — arXiv (cs.AI/cs.LG/cs.CL) + HuggingFace Daily Papers + HN AI posts
2. **Script** — GPT-4 writes a structured 15-min script + metadata
3. **Voice** — ElevenLabs TTS narration
4. **Video** — RunwayML generates b-roll clips; MoviePy assembles the final video
5. **Publish** — YouTube Data API v3 uploads long-form + auto-cut 60s Short

## Project layout

```
ai_news_pipeline/
├── src/
│   ├── main.py              # Pipeline orchestrator
│   ├── scraper.py           # arXiv + HuggingFace + HN
│   ├── script_generator.py  # GPT script + title/tags/description
│   ├── voice_generator.py   # ElevenLabs TTS
│   ├── video_generator.py   # RunwayML + MoviePy assembly
│   ├── shorts_generator.py  # Auto-cut Shorts version
│   ├── youtube_uploader.py  # YouTube upload w/ OAuth2
│   └── scheduler.py         # APScheduler daily runner
├── config/settings.py       # Centralized config
├── requirements.txt
├── .env.example
├── Dockerfile
└── .github/workflows/daily.yml  # Runs at 08:00 UTC every day
```

## Setup

1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and fill in keys
4. Download `client_secrets.json` from Google Cloud Console (YouTube Data API v3, OAuth2 desktop app) and place it in `config/`
5. First run: `python src/youtube_uploader.py --auth` to complete OAuth once (stores `token.pickle`)
6. Manual test run: `python src/main.py`
7. Daemon mode: `python src/scheduler.py`

## Required API keys / accounts

| Service | Env var | Notes |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | gpt-4o-mini is enough for script writing |
| ElevenLabs | `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID` | ~$22/mo Creator tier covers ~15 min/day |
| RunwayML | `RUNWAYML_API_KEY` | Gen-3 Alpha Turbo endpoint |
| YouTube | `client_secrets.json` | OAuth desktop flow, one-time |

## Deploying

- **Local / VPS:** `python src/scheduler.py` under systemd or tmux
- **Docker:** `docker build -t ai-news . && docker run --env-file .env ai-news`
- **GitHub Actions:** push to repo and configure repo secrets matching `.env`

## Cost estimate (per daily episode)

- OpenAI GPT-4o-mini: ~$0.05
- ElevenLabs (15 min): ~$0.60
- RunwayML (6 × 10s clips): ~$3.00
- YouTube API: free
- **Total ≈ $3.65/day, ~$110/mo**