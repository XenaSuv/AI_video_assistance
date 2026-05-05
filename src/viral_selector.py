from __future__ import annotations

import re
from typing import Iterable

from openai import OpenAI
from src.retry_utils import make_openai_client
from loguru import logger

from config import settings
from src.cost_tracker import get_ledger
from src.scraper import NewsItem

HIGH_IMPACT_KEYWORDS = [
    "replace", "kill", "better than", "faster", "automate",
    "launch", "new model", "breakthrough", "AGI"
]

COMPANIES = [
    "OpenAI", "Google", "Meta", "Microsoft", "Anthropic"
]

BORING_WORDS = ["paper", "study", "dataset"]


def keyword_score(item: NewsItem) -> int:
    score = 0
    text = f"{item.title} {item.summary}".lower()

    for kw in HIGH_IMPACT_KEYWORDS:
        if kw in text:
            score += 2

    for company in COMPANIES:
        if company.lower() in text:
            score += 3

    if len(text) < 300:
        score += 1

    return score


def gpt_score(item: NewsItem) -> int:
    prompt = f"""
Rate how viral this AI news story is from 1 to 10.

News:
{item.title} — {item.summary}

SCORING CRITERIA:

1–3 → Low:
- Minor update, niche, no clear impact

4–6 → Medium:
- Interesting but not surprising or widely relevant

7–8 → High:
- Clear impact (pricing, capabilities, major company)
- Likely to attract clicks

9–10 → Very high:
- Strong surprise, controversy, or major shift
- People will want to share or argue about it

CONSIDER:
- Click potential (would you click this title?)
- Surprise or contrast (unexpected vs expected)
- Real-world impact (money, jobs, capabilities)
- Discussion potential (would people comment?)

RULES:
- Avoid overrating minor updates
- Be strict — most news should fall in 5–7 range

Return ONLY a number (1–10).
    """

    client = make_openai_client()
    try:
        res = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = (res.choices[0].message.content or "").strip()
        if res.usage:
            get_ledger().record_llm("viral-score", settings.openai_model, res.usage.prompt_tokens or 0, res.usage.completion_tokens or 0)
        match = re.search(r"(\d+)", text)
        if match:
            return max(1, min(10, int(match.group(1))))
        logger.warning("Viral selector: GPT returned unexpected score '%s'", text)
    except Exception as exc:
        logger.warning("Viral selector GPT scoring failed: %s", exc)

    return 0


def is_boring(item: NewsItem) -> bool:
    title = item.title.lower()
    return any(word in title for word in BORING_WORDS)


def score_news(item: NewsItem) -> int:
    if is_boring(item):
        return -1

    return keyword_score(item) + gpt_score(item)


def pick_viral_news(news_items: Iterable[NewsItem], top_n: int = 3) -> list[NewsItem]:
    scored: list[tuple[int, NewsItem]] = []

    for item in news_items:
        score = score_news(item)
        if score < 0:
            continue
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_n]]
