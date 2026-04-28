from __future__ import annotations

import json
from openai import OpenAI

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

    return json.loads(res.choices[0].message.content or "[]")


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
    return res.choices[0].message.content.strip()
