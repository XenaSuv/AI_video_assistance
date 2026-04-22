"""Scrape latest AI news from arXiv + HuggingFace + Hacker News.

Each source returns a list of NewsItem dicts. Downstream code treats them uniformly.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any

import arxiv
import requests
from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class NewsItem:
    source: str
    title: str
    url: str
    summary: str
    authors: list[str]
    published: str  # ISO date
    score: float = 0.0  # popularity signal, higher = better

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------- arXiv ---------------------------

ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"]


def scrape_arxiv(max_results: int = 20, days_back: int = 2) -> list[NewsItem]:
    """Pull recent cs.AI/LG/CL submissions. arXiv is relevance-first, so we
    post-filter by submission date and keep only the freshest."""
    query = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    search = arxiv.Search(
        query=query,
        max_results=max_results * 3,  # over-fetch, filter later
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_back)
    items: list[NewsItem] = []

    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    for r in client.results(search):
        if r.published < cutoff:
            continue
        items.append(
            NewsItem(
                source="arXiv",
                title=r.title.strip().replace("\n", " "),
                url=r.entry_id,
                summary=r.summary.strip().replace("\n", " ")[:1200],
                authors=[a.name for a in r.authors][:5],
                published=r.published.date().isoformat(),
                # More authors => often more collaborative / higher-profile work.
                score=min(len(r.authors) / 10, 1.0),
            )
        )
        if len(items) >= max_results:
            break

    logger.info(f"arXiv: {len(items)} papers")
    return items


# --------------------- HuggingFace Daily Papers ---------------------

HF_DAILY_URL = "https://huggingface.co/api/daily_papers"


def scrape_huggingface(max_results: int = 15) -> list[NewsItem]:
    """HF Daily Papers — curated + ranked by community upvotes. The best signal
    of 'what's actually trending in AI today'."""
    try:
        resp = requests.get(HF_DAILY_URL, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"HF Daily API failed, falling back to HTML scrape: {e}")
        return _scrape_huggingface_html(max_results)

    items = []
    for entry in data[:max_results]:
        p = entry.get("paper", {})
        upvotes = entry.get("numComments", 0) + p.get("upvotes", 0)
        items.append(
            NewsItem(
                source="HuggingFace",
                title=p.get("title", "").strip(),
                url=f"https://huggingface.co/papers/{p.get('id', '')}",
                summary=(p.get("summary") or "").strip()[:1200],
                authors=[a.get("name", "") for a in p.get("authors", [])][:5],
                published=(entry.get("publishedAt") or "")[:10],
                score=min(upvotes / 50, 2.0),  # HF upvotes are the best signal
            )
        )

    logger.info(f"HuggingFace: {len(items)} papers")
    return items


def _scrape_huggingface_html(max_results: int) -> list[NewsItem]:
    """Fallback — parse the HTML page if the JSON API changes."""
    resp = requests.get("https://huggingface.co/papers", timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")
    items = []
    for art in soup.select("article")[:max_results]:
        a = art.find("a", href=True)
        if not a or not a["href"].startswith("/papers/"):
            continue
        items.append(
            NewsItem(
                source="HuggingFace",
                title=a.get_text(strip=True),
                url=f"https://huggingface.co{a['href']}",
                summary="",
                authors=[],
                published=dt.date.today().isoformat(),
                score=1.0,
            )
        )
    return items


# --------------------- Hacker News (AI-related) ---------------------

HN_SEARCH = "https://hn.algolia.com/api/v1/search_by_date"
AI_KEYWORDS = ["AI", "LLM", "GPT", "Claude", "Gemini", "Llama",
               "transformer", "diffusion", "OpenAI", "Anthropic", "DeepMind"]


def scrape_hackernews(max_results: int = 10, hours_back: int = 24) -> list[NewsItem]:
    """HN stories tagged with AI keywords from the last 24h, sorted by points."""
    since = int((dt.datetime.now() - dt.timedelta(hours=hours_back)).timestamp())
    params = {
        "query": " OR ".join(AI_KEYWORDS),
        "tags": "story",
        "numericFilters": f"created_at_i>{since},points>20",
        "hitsPerPage": max_results,
    }
    try:
        resp = requests.get(HN_SEARCH, params=params, timeout=20)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
    except Exception as e:
        logger.warning(f"HN scrape failed: {e}")
        return []

    items = []
    for h in hits:
        items.append(
            NewsItem(
                source="HackerNews",
                title=h.get("title", ""),
                url=h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
                summary=(h.get("story_text") or "")[:800],
                authors=[h.get("author", "")],
                published=h.get("created_at", "")[:10],
                score=min(h.get("points", 0) / 100, 2.0),
            )
        )
    logger.info(f"HackerNews: {len(items)} stories")
    return items


# --------------------- Aggregator ---------------------

def scrape_all(top_n: int = 10) -> list[NewsItem]:
    """Run every source, merge, dedupe, and return top-N by score."""
    all_items = scrape_huggingface() + scrape_arxiv() + scrape_hackernews()

    # Dedupe by fuzzy title (lowercased, first 60 chars)
    seen: set[str] = set()
    unique: list[NewsItem] = []
    for it in all_items:
        key = it.title.lower().strip()[:60]
        if key and key not in seen:
            seen.add(key)
            unique.append(it)

    unique.sort(key=lambda x: x.score, reverse=True)
    top = unique[:top_n]
    logger.info(f"Aggregated {len(top)} top items from {len(all_items)} total")
    return top


if __name__ == "__main__":
    import json

    results = scrape_all()
    print(json.dumps([r.to_dict() for r in results], indent=2))
