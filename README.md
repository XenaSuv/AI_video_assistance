# AI Video Assistance — Self-Improving AI Media System

AI Video Assistance is not just a video generator.

It is a **self-improving AI media system** that:
- learns from audience retention
- adapts storytelling in real time
- evolves content strategy automatically

Fully automated production system that generates daily AI news videos, weekly tutorials, and breaking news alerts — publishing to YouTube (EN + RU) and TikTok with zero manual intervention.

**Status**: Production Ready (v1.0, 5.7K LOC, 29 modules)

---

## 🎯 What Makes This Different

Most AI content pipelines generate videos.

This system:
- decides **what is worth saying**
- controls **how it is told**
- learns **what actually works**

→ It doesn’t just scale content  
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
  - Multi-language support  

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