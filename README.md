# AI Video Assistance — Automated AI Content Pipeline

Fully automated production system that generates daily AI news videos, weekly tutorials, and breaking news alerts — posting to YouTube (English + Russian) and TikTok with zero manual intervention. **Status: Production Ready** (v1.0, 5.7K LOC, 29 modules).

> **Current Architecture**: 4 parallel pipelines (daily, weekly, breaking, digest) with multi-language support, intelligent editorial planning, human-like narration, and cached artifact reuse. Runs entirely on GitHub Actions with optional local scheduler.

## Content Calendar

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

## Core Architecture

The pipeline combines AI-powered editorial intelligence with human-like storytelling to create engaging content, and continuously learns from viewer engagement metrics.

### Editorial Intelligence Layer
- **Editorial Brain**: Analyzes news stories, selects angles, defines personas, and plans scene structure
- **Humanizer Agent**: Transforms AI-generated scripts into natural, conversational narration
- **Micro-Hook Agent**: Inserts attention-grabbing hooks with voice pacing annotations
- **Feedback Analyzer**: Monitors video performance and learns what works

### Learning Loop
```
published_video → YouTube/TikTok metrics → feedback_analyzer → insights → editorial_brain
                                                                           └── boost high-performing angles/formats
                                                                           └── hook_optimizer → micro_hooks
```

### Production Pipeline
```
scrape → editorial_brain → generate_script → humanizer → micro_hooks → voice → video → shorts → thumbnail → upload
                                                                    └── TikTok (short)
└─ if RU_ENABLED: translate → ru-voice → reassemble → ru-shorts → ru-thumbnail → ru-upload
```

---

## Key Components

### 1. Editorial Brain (`src/editorial_brain.py`)
Intelligent content director that transforms raw news into editorial plans:
- **Story Ranking**: Hype, novelty, controversy, and audience fit scoring
- **Angle Selection**: Conflict-aware angles (hype vs reality, innovation vs adoption)
- **Persona Assignment**: Skeptical insider, curious engineer, analytic, explainer
- **Format Planning**: Quick-hit, deep-dive, hot-take with scene count and pacing
- **Scene Planning**: Mandatory structure with goals and styles

### 2. Humanizer Agent (`src/humanizer_agent.py`)
Multi-pass agent that makes scripts sound human:
- **Structure Breaker**: Converts perfect paragraphs to conversational chunks
- **Sentence Variator**: Mixes short punchy sentences with longer explanations
- **Reaction Injector**: Adds emotional responses and subjectivity
- **Imperfect Layer**: Inserts repetitions, fillers, and natural hesitations
- **Rhythm Optimizer**: Adds text-based pauses and emphasis
- **Voice Tuner**: Fine-tunes to match persona and editorial tone

### 3. Micro-Hook Agent (`src/micro_hook_agent.py`)
Inserts attention-sustaining hooks with voice annotations:
- **Hook Types**: Curiosity, conflict, twist, personal, doubt
- **Voice Tags**: `[PAUSE_SHORT]`, `[PAUSE_LONG]`, `[EMPHASIS]`
- **Persona-Aware**: Adapts hook style to editorial voice
- **SSML Integration**: Converts annotations to ElevenLabs-compatible SSML

### 4. Voice Generator (`src/voice_generator.py`)
ElevenLabs-powered TTS with dramatic pacing:
- **SSML Support**: Automatic conversion of voice annotations to `<break>` and `<emphasis>`
- **Multi-Language**: 30+ languages with native voice models
- **Scene-Based**: Per-scene MP3 generation with duration tracking

### 5. Feedback Analyzer (`src/feedback_analyzer.py`)
Monitors video performance and guides editorial decisions:
- **Metrics Collection**: Pulls views, retention curves, watch time from YouTube/TikTok
- **Retention Analysis**: Identifies drop points and best-performing segments
- **Hook Scoring**: Measures effectiveness of opening hooks (30s retention metric)
- **Performance Learning**: Tracks angle and format success rates over time
- **Recommendations**: Generates actionable insights for future content
- **Editorial Feedback Loop**: Boosts high-performing angles/formats in editorial brain
- **YouTube Analytics Integration**: Uses `src/youtube_analytics.py` for real-time metrics and retention curves

### 6. YouTube Analytics Client (`src/youtube_analytics.py`)
Direct integration with YouTube Analytics API:
- **Video Metrics**: Retrieves views, average view duration, and view percentage
- **Retention Curves**: Gets audience watch ratio over elapsed video time
- **Hook Performance**: Measures 30-second retention for hook effectiveness analysis
- **Real-time Data**: Fetches live performance data for continuous optimization

### 7. Hook Optimizer (`src/hook_optimizer.py`)
Learns from hook performance to optimize future hook generation:
- **Pattern Extraction**: Analyzes successful vs unsuccessful hook patterns from feedback data
- **Performance Scoring**: Ranks hook types and templates by engagement metrics
- **Context Filtering**: Recommends hooks based on editorial angle, format, and persona
- **Avoidance Patterns**: Identifies and filters out underperforming hook types
- **Optimization Integration**: Provides recommendations to micro-hook agent for improved engagement

### 8. Other Components
- **Scraper**: Multi-source AI news aggregation (arXiv, HuggingFace, HackerNews, official blogs)
- **Deduplicator**: SQLite-based rolling window prevents content recycling
- **Script Generator**: GPT-4o structured narration with metadata
- **Video Generator**: DALL-E 3 images with Ken Burns animation
- **Thumbnail Generator**: A/B testing with automated selection
- **Uploader**: Multi-platform publishing (YouTube, TikTok)

---

## Installation

### Prerequisites
- Python 3.11+
- FFmpeg
- Docker (optional, for dev container)

### Setup
```bash
git clone https://github.com/XenaSuv/AI_video_assistance.git
cd AI_video_assistance
pip install -r requirements.txt
```

### Configuration
Create `config/settings.py` with:
```python
# API Keys
openai_api_key = "sk-..."
elevenlabs_api_key = "sk_..."
youtube_api_key = "..."
tiktok_access_token = "..."

# Channels
channel_name = "Your AI Channel"
youtube_channel_id = "..."
tiktok_username = "..."

# Content Settings
script_target_words = 2200
elevenlabs_voice_id = "your-voice-id"
elevenlabs_model = "eleven_multilingual_v2"
```

---

## Usage

### Local Development
```bash
# Run daily pipeline
python -m src.main

# Run specific pipeline
python -m src.main --pipeline daily

# Test components
python -c "from src.editorial_brain import EditorialBrain; print('OK')"
```

### Production Deployment
The system runs on GitHub Actions with scheduled workflows. Configure secrets for API keys and deploy.

### Docker
```bash
docker build -t ai-video .
docker run -v $(pwd)/output:/app/output ai-video
```

---

## Learning & Optimization

The system continuously learns from viewer engagement to improve future content:

### Feedback Loop
1. **Video Published** → YouTube/TikTok
2. **Feedback Analyzer** → Fetches retention curves, views, watch time
3. **Insights Generated** → Hook scores, drop points, recommendations
4. **Editorial Brain Updates** → Boosts high-performing angles and formats
5. **Next Video** → Benefits from learned patterns

### Key Metrics
- **Hook Score**: Retention at 30-second mark (target: >0.6)
- **Average Watch %**: Overall completion rate (target: >50%)
- **Best Segment**: Where viewers stayed longest
- **Drop Points**: Where retention fell >15%
- **Angle Performance**: Which editorial angles drive engagement
- **Format Performance**: Which structures (hot-take vs deep-dive) work best

### Feedback History
Performance data stored in `data/feedback_history.json` for trend analysis and optimization.

---

## Pipeline Details

### Daily News Pipeline
1. **Scrape** — Aggregate AI news from arXiv, HuggingFace, HackerNews, official blogs
2. **Editorial Brain** — Rank stories, select angles, define personas and formats
3. **Generate Script** — GPT-4o creates structured narration with editorial guidance
4. **Humanizer** — Transform to conversational, human-like speech
5. **Micro-Hooks** — Insert attention hooks with voice pacing
6. **Voice** — ElevenLabs TTS with SSML pauses and emphasis
7. **Video** — DALL-E 3 images + Ken Burns animation
8. **Shorts** — 9:16 vertical crops with captions
9. **Thumbnail** — A/B testing and automated selection
10. **Upload** — Multi-platform publishing

### Weekly Tutorials
- Topic rotation with cooldown periods
- Tool-specific scripting (Claude, ChatGPT, Gemini)
- Same production pipeline as daily news

### Breaking News
- Triggered on high-impact stories
- Accelerated pipeline (10 min runtime)
- Short-form focused

### Weekly Digest
- Aggregates past week's content
- Extended format (35 min)
- Comprehensive recap

---

## Architecture Benefits

- **Editorial Intelligence**: No more generic AI scripts — each video has personality and conflict
- **Human-Like Narration**: Multi-pass humanization creates engaging, conversational content
- **Attention Engineering**: Micro-hooks with voice pacing maximize viewer retention
- **Multi-Platform**: Optimized for YouTube, TikTok, and international audiences
- **Production Ready**: Fully automated with error recovery and caching

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new components
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
