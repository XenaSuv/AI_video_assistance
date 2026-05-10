"""Shared fixtures and env setup for all tests.

Must set env vars and stub heavy imports before any project modules are
imported, because config/settings.py evaluates Settings() at import time
and some modules import googleapiclient / google.auth at the module level.
"""
import os
import sys
from unittest.mock import MagicMock

os.environ.setdefault("OPENAI_API_KEY", "test-key-openai")
os.environ.setdefault("ELEVENLABS_API_KEY", "test-key-elevenlabs")

# Stub heavy C-extension / optional deps so tests run without full installs
for _mod in (
    # Google SDK
    "googleapiclient",
    "googleapiclient.discovery",
    "googleapiclient.errors",
    "google",
    "google.oauth2",
    "google.auth",
    "google.auth.transport",
    "google.auth.transport.requests",
    "google_auth_oauthlib",
    "google_auth_oauthlib.flow",
    # numpy / ffmpeg_utils (pulled in by quality_gate and video modules)
    "numpy",
    # Pillow — imported at module level by image_generator
    "PIL",
    "PIL.Image",
    # arxiv pulls in feedparser which requires sgmllib (removed in Python 3)
    "arxiv",
    "feedparser",
):
    sys.modules.setdefault(_mod, MagicMock())

import datetime
import pytest
from src.scraper import NewsItem


def _make_story(
    title: str = "GPT-5 launches with record benchmarks",
    source: str = "OpenAI",
    summary: str = "OpenAI announced GPT-5 with state of the art results.",
    published: str | None = None,
    url: str = "https://openai.com/gpt5",
) -> NewsItem:
    if published is None:
        published = datetime.date.today().isoformat()
    return NewsItem(
        source=source,
        title=title,
        url=url,
        summary=summary,
        authors=[],
        published=published,
    )


@pytest.fixture
def fresh_story() -> NewsItem:
    return _make_story()


@pytest.fixture
def old_story() -> NewsItem:
    old_date = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
    return _make_story(
        title="Old AI paper from last month",
        source="arXiv",
        published=old_date,
    )


@pytest.fixture
def controversy_story() -> NewsItem:
    return _make_story(
        title="AI model faces ban over privacy and bias lawsuit",
        summary="Regulators are considering banning the model due to security and ethical concerns.",
    )


@pytest.fixture
def technical_story() -> NewsItem:
    return _make_story(
        title="New benchmark tests model parameters and fine-tuning dataset",
        summary="Researchers published a dataset for benchmarking architecture latency and throughput.",
    )


@pytest.fixture
def feedback_history():
    """10 realistic feedback records with varied angles and retention."""
    angles = ["technical_breakthrough", "industry_impact", "threat_to_jobs"]
    formats = ["quick_hit", "deep_dive", "hot_take"]
    records = []
    for i in range(10):
        records.append({
            "video_id": f"vid_{i}",
            "platform": "youtube",
            "hook_score": 0.5 + (i % 3) * 0.1,
            "avg_view_percentage": 0.4 + (i % 4) * 0.1,
            "angle": angles[i % len(angles)],
            "format": formats[i % len(formats)],
            "drop_point": None,
        })
    return records


@pytest.fixture
def high_performing_feedback():
    """Feedback where all videos have excellent retention."""
    return [
        {
            "video_id": f"vid_{i}",
            "hook_score": 0.80 + i * 0.01,
            "avg_view_percentage": 0.70 + i * 0.01,
            "angle": "technical_breakthrough",
            "format": "deep_dive",
        }
        for i in range(10)
    ]


@pytest.fixture
def low_performing_feedback():
    """Feedback where all videos have poor retention."""
    return [
        {
            "video_id": f"vid_{i}",
            "hook_score": 0.30 + i * 0.01,
            "avg_view_percentage": 0.25 + i * 0.01,
            "angle": "overhyped_vs_reality",
            "format": "hot_take",
        }
        for i in range(10)
    ]
