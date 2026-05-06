# AI Video Assistance — Self-Improving AI Media System

AI Video Assistance is not just a video generator.

It is a **self-improving AI media system** that:
- learns from audience retention
- adapts storytelling in real time
- evolves content strategy automatically

Fully automated production system that generates daily AI news videos, weekly tutorials, breaking news alerts, and weekly digests — publishing to YouTube (EN + RU) and TikTok with zero manual intervention.

**Status**: Production Ready (v1.0, 5.7K LOC, 29 modules)

---

## 🎯 What Makes This Different

Most AI content pipelines generate videos.

This system:
- decides **what is worth saying**
- controls **how it is told**
- learns **what actually works**

→ It doesn't just scale content
→ It scales attention

---

## 📅 Content Calendar

| Day | Pipeline | Output | Platforms | Runtime |
|---|---|---|---|---|
| Mon – Sun | Daily AI News | 15-min video + Short | YouTube EN/RU, TikTok | ~20 min |
| Breaking | Triggered events | 2–5 min + Short | YouTube EN/RU | ~10 min |
| Mon | Claude Tutorial | 20-min + Short | YouTube EN/RU | ~30 min |
| Wed | ChatGPT Tutorial | 20-min + Short | YouTube EN/RU | ~30 min |
| Fri | Gemini Tutorial | 20-min + Short | YouTube EN/RU | ~30 min |
| Sun | Weekly Digest | 35-min recap + Short | YouTube EN/RU | ~45 min |

**Monthly Output**: ~150 long-form videos + unlimited shorts

---

# 🧠 Core Architecture

The system is structured into three layers:

Story Intelligence → Narrative Engine → Learning System

---

## 🧠 1. Story Intelligence

Decides **what to say and why it matters**

- **Editorial Brain (`src/editorial_brain.py`)**
  - Ranks stories (hype, novelty, controversy)
  - Selects angles (hype vs reality, impact, risk)
  - Assigns personas
  - Defines format and scene structure

- **Viral Selector (`src/viral_selector.py`)**
  - Identifies high-potential news stories
  - Scores stories for virality potential
  - Filters content for maximum engagement

- **Hook Optimizer (`src/hook_optimizer.py`)**
  - Learns which hooks actually retain viewers
  - Scores patterns based on real data
  - Filters weak hooks

- **Hook Mutation Engine (`src/hook_mutation_engine.py`)**
  - Evolves high-performing hooks into new variations
  - Prevents repetition
  - Continuously generates new phrasing

---

## 🎭 2. Narrative Engine

Controls **how the story is told**

- **Script Generator (`src/script_generator.py`)**
  - Generates structured narration guided by editorial decisions

- **Humanizer Agent (`src/humanizer_agent.py`)**
  - Breaks rigid AI structure
  - Adds reactions, imperfections, rhythm
  - Makes narration sound human

- **Micro-Hook Agent (`src/micro_hook_agent.py`)**
  - Inserts engagement hooks every 20–40 seconds
  - Maintains viewer attention
  - Adds pacing annotations

- **Voice Generator (`src/voice_generator.py`)**
  - ElevenLabs TTS with SSML pacing
  - Emotional delivery via pauses and emphasis
  - Multi-language support (EN/RU)

- **Video Generator (`src/video_generator.py`)**
  - Assembles clips with DALL-E + Ken Burns effects
  - Adds background music throughout entire video
  - Supports intro/outro segments
  - Generates 16:9 HD output

---

## 📊 3. Learning System

Continuously improves based on audience behavior

- **Feedback Analyzer (`src/feedback_analyzer.py`)**
  - Tracks retention curves
  - Detects drop points
  - Scores hook effectiveness
  - Generates insights

- **YouTube Analytics Client (`src/youtube_analytics.py`)**
  - Retrieves real performance data
  - Audience watch ratio
  - Engagement metrics

- **TikTok Analytics (`src/tiktok_analytics.py`)**
  - Tracks TikTok performance metrics
  - Engagement and retention analysis

- **Decision Engine (`src/decision_engine.py`)**
  - Controls system strategy
  - Balances exploration vs exploitation
  - Adjusts angle & format weights
  - Prevents overfitting
  - Selects mode (growth / safe / experimental)

- **Performance Tracker (`src/performance_tracker.py`)**
  - Records pipeline performance metrics
  - Tracks costs and efficiency
  - Monitors system health

---

# 🔁 Learning Loop

```
published_video
↓
YouTube / TikTok metrics
↓
feedback_analyzer
↓
decision_engine
↓
hook_optimizer + mutation
↓
editorial_brain
↓
next video (improved)
```

Hook mutation is applied before script generation to evolve stronger hooks.

---

# 📈 Key Metrics

- **Hook Score** (30s retention): target > 0.6
- **Average Watch %**: target > 50%
- **Drop Points**: retention -15%
- **Angle Performance**: what narratives work
- **Format Performance**: what structures retain viewers

---

# 🧠 Why This Matters

Most systems generate content.

This system:
- learns what people actually watch
- adapts storytelling automatically
- evolves over time

→ It behaves like a **media team, not a script generator**

---

# 🛠 Installation

## Requirements
- Python 3.12+
- FFmpeg
- Docker (optional, for containerized deployment)

## Quick Setup

```bash
git clone https://github.com/XenaSuv/AI_video_assistance.git
cd AI_video_assistance
pip install -r requirements.txt
```

## Configuration

Create a `.env` file or set environment variables:

```bash
# Required API Keys
OPENAI_API_KEY=sk-your-openai-key
ELEVENLABS_API_KEY=your-elevenlabs-key

# Optional: YouTube Upload (EN)
YOUTUBE_CLIENT_SECRETS=config/client_secrets.json
YOUTUBE_TOKEN_FILE=config/token.pickle

# Optional: Russian Language Support
RU_ENABLED=true
RU_ELEVENLABS_VOICE_ID=your-ru-voice-id
RU_YOUTUBE_CLIENT_SECRETS=config/client_secrets_ru.json
RU_YOUTUBE_TOKEN_FILE=config/token_ru.pickle

# Optional: TikTok Upload
TIKTOK_ENABLED=true
TIKTOK_CLIENT_KEY=your-tiktok-key
TIKTOK_CLIENT_SECRET=your-tiktok-secret

# Optional: Background Music
BACKGROUND_MUSIC_PATH=background_music.mp3
BACKGROUND_MUSIC_VOLUME=0.10

# Optional: Slack Notifications
SLACK_WEBHOOK_URL=https://hooks.slack.com/...

# Optional: AI Presenter (D-ID)
DID_API_KEY=your-did-key
PRESENTER_ENABLED=true
```

---

# ▶️ Usage

## Running Pipelines

### Daily AI News
```bash
python src/main.py
```

### Weekly Tutorials
```bash
# Claude tutorial (Monday)
python src/weekly_main.py --tool claude

# ChatGPT tutorial (Wednesday)
python src/weekly_main.py --tool chatgpt

# Gemini tutorial (Friday)
python src/weekly_main.py --tool gemini
```

### Breaking News (when detected)
```bash
python src/breaking_main.py --item data/breaking_current.json
```

### Weekly Digest (Sunday)
```bash
python src/digest_main.py
```

### Breaking News Detection
```bash
python src/breaking_detector.py --check
```

## Automated Scheduling

Run the scheduler daemon:
```bash
python src/scheduler.py
```

Or use Docker:
```bash
docker build -t ai-video .
docker run -e OPENAI_API_KEY=... ai-video
```

---

# 🧪 Testing

Run the test suite:
```bash
python -m pytest tests/ -v
```

---

# 🚀 Deployment

## GitHub Actions

The system includes automated workflows for:
- Daily news pipeline (Mon-Sat 08:00 UTC)
- Weekly tutorials (Mon/Wed/Fri 09:00 UTC)
- Breaking news detection (every 30 min)
- Weekly digest (Sun 10:00 UTC)

Configure repository secrets for API keys and deploy.

## Docker Deployment

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y ffmpeg imagemagick fonts-dejavu
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "src/scheduler.py"]
```

---

# 📁 Project Structure

```
AI_video_assistance/
├── src/                    # Core modules
│   ├── main.py            # Daily news pipeline
│   ├── weekly_main.py     # Weekly tutorials
│   ├── breaking_main.py   # Breaking news
│   ├── digest_main.py     # Weekly digest
│   ├── editorial_brain.py # Story intelligence
│   ├── script_generator.py # Narrative engine
│   ├── voice_generator.py # TTS synthesis
│   ├── video_generator.py # Video assembly
│   └── ...
├── config/                # Configuration
│   ├── settings.py        # Environment config
│   └── client_secrets.json # YouTube API
├── data/                  # Persistent data
├── output/                # Generated content
├── logs/                  # Execution logs
├── assets/                # Static assets
├── tests/                 # Test suite
└── .github/workflows/     # CI/CD automation
```

---

# 🔧 Key Features

- **Multi-language Support**: English + Russian variants
- **Multi-platform Publishing**: YouTube, TikTok
- **AI Learning Loop**: Continuous improvement from analytics
- **Quality Gates**: Automated content validation
- **Cost Tracking**: API usage monitoring
- **Checkpointing**: Resumable pipeline execution
- **Slack Notifications**: Pipeline status alerts
- **Background Music**: Optional audio tracks
- **AI Presenter**: D-ID talking head integration
- **Thumbnail A/B Testing**: Automated thumbnail optimization

---

# 📜 License

MIT License

Continuously improves based on audience behavior

- **Feedback Analyzer (`src/feedback_analyzer.py`)**  
  - Tracks retention curves  
  - Detects drop points  
  - Scores hook effectiveness  
  - Generates insights  

- **YouTube Analytics Client (`src/youtube_analytics.py`)**  
  - Retrieves real performance data  
  - Audience watch ratio  
  - Engagement metrics  

- **Decision Engine (`src/decision_engine.py`)**  
  - Controls system strategy  
  - Balances exploration vs exploitation  
  - Adjusts angle & format weights  
  - Prevents overfitting  
  - Selects mode (growth / safe / experimental)  

---

# 🔁 Learning Loop

published_video
↓
YouTube / TikTok metrics
↓
feedback_analyzer
↓
decision_engine
↓
hook_optimizer + mutation
↓
editorial_brain
↓
next video (improved)


Hook mutation is applied before script generation to evolve stronger hooks.

---

# 📈 Key Metrics

- **Hook Score** (30s retention): target > 0.6  
- **Average Watch %**: target > 50%  
- **Drop Points**: retention -15%  
- **Angle Performance**: what narratives work  
- **Format Performance**: what structures удерживают  

---

# 🧠 Why This Matters

Most systems generate content.

This system:
- learns what people actually watch  
- adapts storytelling automatically  
- evolves over time  

→ It behaves like a **media team, not a script generator**

---

# 🛠 Installation

## Requirements
- Python 3.11+
- FFmpeg
- Docker (optional)

## Setup

```bash
git clone https://github.com/XenaSuv/AI_video_assistance.git
cd AI_video_assistance
pip install -r requirements.txt

Configuration

# API Keys
openai_api_key = "sk-..."
elevenlabs_api_key = "sk_..."
youtube_api_key = "..."
tiktok_access_token = "..."

# Channel
channel_name = "Your AI Channel"

# Settings
script_target_words = 2200

▶️ Usage

python -m src.main

🚀 What This Becomes

This is not just automation.

It is a system that:

experiments
learns
improves

→ A foundation for a fully autonomous AI media channel

📜 License

MIT License