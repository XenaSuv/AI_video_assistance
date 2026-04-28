from __future__ import annotations

import json
import re
from openai import OpenAI
from loguru import logger

from config import settings


client = OpenAI(api_key=settings.openai_api_key)


def generate_titles(news_item, n: int = 5) -> list[str]:
    prompt = f"""
    Create {n} highly clickable YouTube titles.

    News:
    {news_item['title']} - {news_item['summary']}

    Rules:
    - Max 60 characters
    - Use curiosity
    - Make people want to click
    - Avoid boring wording

    Examples of style:
    - "This AI changes everything"
    - "You won’t believe what AI just did"
    - "This replaces your job?"

    Return JSON array.
    """

    res = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    text = (res.choices[0].message.content or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        titles = json.loads(text or "[]")
        if isinstance(titles, list):
            return [str(t).strip() for t in titles if str(t).strip()]
        logger.warning("title_optimizer: generated titles response is not a list: %r", titles)
    except Exception as exc:
        logger.warning("title_optimizer failed to parse titles: %s | response=%r", exc, text)

    fallback = [news_item.get("title", "").strip() or "AI NEWS"]
    return fallback


def pick_best_title(titles: list[str]) -> str:
    prompt = f"""
    Choose the most clickable YouTube title.

    Titles:
    {titles}

    Criteria:
    - curiosity
    - emotional impact
    - clarity

    Return ONLY the best title.
    """

    res = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
    )
    result = (res.choices[0].message.content or "").strip()
    if result:
        return result
    return titles[0] if titles else "AI NEWS"
