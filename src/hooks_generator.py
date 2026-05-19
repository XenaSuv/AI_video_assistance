from __future__ import annotations

import re

from loguru import logger

from config import settings
from src.cost_tracker import get_ledger
from src.retry_utils import make_openai_client


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

    client = make_openai_client()
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    import json
    text = (response.choices[0].message.content or "").strip()
    if response.usage:
        get_ledger().record_llm("hooks-generate", settings.openai_model, response.usage.prompt_tokens or 0, response.usage.completion_tokens or 0)
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
