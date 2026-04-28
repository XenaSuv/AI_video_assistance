from __future__ import annotations

import re
from openai import OpenAI
from loguru import logger

from config import settings


def generate_hooks(news_text: str, n: int = 3) -> list[str]:
    prompt = f"""
    Generate {n} viral hooks for a YouTube Shorts video.

    News:
    {news_text}

    Rules:
    - Max 10 words
    - Each hook must be different
    - Use curiosity, fear, or benefit

    Output as JSON array.
    """

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    import json
    text = (response.choices[0].message.content or "").strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        hooks = json.loads(text or "[]")
        if not isinstance(hooks, list):
            raise ValueError("Hooks response was not a JSON array")
        return [str(h).strip() for h in hooks if str(h).strip()]
    except Exception as exc:
        logger.warning(f"hooks_generator failed: {exc} | response={text!r}")
        return [f"Why this AI news matters #{i+1}" for i in range(n)]
