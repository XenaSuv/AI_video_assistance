from __future__ import annotations

import json
import re
from src.retry_utils import make_openai_client
from loguru import logger

from config import settings
from src.cost_tracker import get_ledger


client = make_openai_client()


def generate_titles(news_item, n: int = 5) -> list[str]:
    prompt = f"""
    Create {n} highly clickable YouTube titles.

News:
{news_item['title']} — {news_item['summary']}

GOAL:
Maximize click-through rate while staying specific and believable.

--------------------------------
RULES
--------------------------------
- Max 60 characters
- Each title must feel different (vary angle)
- Use curiosity, but avoid generic clickbait

--------------------------------
PREFER:
- Specific details (numbers, results, comparisons)
- Clear outcome ("what changed", "what this means")
- Tension or contrast (better vs worse, faster vs slower)
- Real-world impact (cost, jobs, speed, capability)

--------------------------------
AVOID:
- "This changes everything"
- "You won't believe"
- vague or generic phrases
- titles that could apply to any AI news

--------------------------------
TITLE STYLES (mix these):
- Result-driven ("This AI is 10x cheaper than GPT-4")
- Question ("Is this the first real GPT-4 competitor?")
- Contrarian ("This AI is impressive — but flawed")
- Impact ("This could replace junior developers")

--------------------------------
OUTPUT:
Return a JSON array of titles.
    """

    res = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
    )

    text = (res.choices[0].message.content or "").strip()
    if res.usage:
        get_ledger().record_llm("title-generate", settings.openai_model, res.usage.prompt_tokens or 0, res.usage.completion_tokens or 0)
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

GOAL:
Maximize click-through rate (CTR).

CRITERIA (in priority order):
1. Curiosity gap (does it make you want to click immediately?)
2. Clear value (what will the viewer gain?)
3. Specificity (numbers, constraints, concrete outcome)
4. Emotional or surprising angle
5. Clarity (easy to understand at a glance)

PREFER titles that:
- include tension or conflict
- suggest a result or outcome
- use numbers or constraints ("7 days", "$0", "500 pages")

AVOID titles that:
- are generic or safe
- sound like summaries
- lack a clear hook

If multiple are similar, choose the more bold and curiosity-driven one.

Return ONLY the best title.
    """

    res = client.chat.completions.create(
        model=settings.openai_model,
        messages=[{"role": "user", "content": prompt}],
    )
    result = (res.choices[0].message.content or "").strip()
    if res.usage:
        get_ledger().record_llm("title-select", settings.openai_model, res.usage.prompt_tokens or 0, res.usage.completion_tokens or 0)
    if result:
        return result
    return titles[0] if titles else "AI NEWS"
