"""Scrape latest AI news from arXiv + HuggingFace + Hacker News +
official company blogs (OpenAI, Anthropic, Google DeepMind, Microsoft AI,
Meta AI, Mistral, DeepSeek, xAI).

Official company items receive a high base score so they are always
prioritised over community/research sources in the final selection.
"""
from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urljoin, quote as _url_quote

import xml.etree.ElementTree as ET

import arxiv
import requests
from bs4 import BeautifulSoup
from src.retry_utils import http_get
from loguru import logger


@dataclass
class NewsItem:
    source: str
    title: str
    url: str
    summary: str
    authors: list[str]
    published: str  # ISO date
    score: float = 0.0  # higher = more important

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────── helpers ───────────────────

_UA = {"User-Agent": "Mozilla/5.0 (compatible; AI-News-Bot/1.0)"}
_OFFICIAL_DAYS_BACK = 3   # how far back to look for official posts
OFFICIAL_SCORE      = 5.0  # beats community max of ~2.0


def _parse_date(raw: str) -> str | None:
    """Return ISO date string from any common date format, or None."""
    if not raw:
        return None
    m = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", raw)
    if m:
        try:
            dt.date(int(m[1]), int(m[2]), int(m[3]))
            return f"{m[1]}-{m[2]}-{m[3]}"
        except ValueError:
            pass
    return None


# ─────────────────── arXiv ───────────────────

ARXIV_CATEGORIES = ["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"]


def scrape_arxiv(max_results: int = 20, days_back: int = 2) -> list[NewsItem]:
    query  = " OR ".join(f"cat:{c}" for c in ARXIV_CATEGORIES)
    search = arxiv.Search(
        query=query,
        max_results=max_results * 3,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_back)
    items: list[NewsItem] = []
    client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
    for r in client.results(search):
        if r.published < cutoff:
            continue
        items.append(NewsItem(
            source="arXiv",
            title=r.title.strip().replace("\n", " "),
            url=r.entry_id,
            summary=r.summary.strip().replace("\n", " ")[:1200],
            authors=[a.name for a in r.authors][:5],
            published=r.published.date().isoformat(),
            score=min(len(r.authors) / 10, 1.0),
        ))
        if len(items) >= max_results:
            break
    logger.info(f"arXiv: {len(items)} papers")
    return items


# ─────────────────── HuggingFace ───────────────────

HF_DAILY_URL = "https://huggingface.co/api/daily_papers"


def scrape_huggingface(max_results: int = 15) -> list[NewsItem]:
    try:
        resp = http_get(HF_DAILY_URL, timeout=20)
        data = resp.json()
    except Exception as e:
        logger.warning(f"HF Daily API failed: {e}")
        return _scrape_huggingface_html(max_results)

    items = []
    for entry in data[:max_results]:
        p       = entry.get("paper", {})
        upvotes = entry.get("numComments", 0) + p.get("upvotes", 0)
        items.append(NewsItem(
            source="HuggingFace",
            title=p.get("title", "").strip(),
            url=f"https://huggingface.co/papers/{p.get('id', '')}",
            summary=(p.get("summary") or "").strip()[:1200],
            authors=[a.get("name", "") for a in p.get("authors", [])][:5],
            published=(entry.get("publishedAt") or "")[:10],
            score=min(upvotes / 50, 2.0),
        ))
    logger.info(f"HuggingFace: {len(items)} papers")
    return items


def _scrape_huggingface_html(max_results: int) -> list[NewsItem]:
    resp = http_get("https://huggingface.co/papers", timeout=20)
    soup = BeautifulSoup(resp.text, "html.parser")
    items = []
    for art in soup.select("article")[:max_results]:
        a = art.find("a", href=True)
        if not a or not a["href"].startswith("/papers/"):
            continue
        items.append(NewsItem(
            source="HuggingFace",
            title=a.get_text(strip=True),
            url=f"https://huggingface.co{a['href']}",
            summary="",
            authors=[],
            published=dt.date.today().isoformat(),
            score=1.0,
        ))
    return items


# ─────────────────── Hacker News ───────────────────

HN_SEARCH   = "https://hn.algolia.com/api/v1/search_by_date"
AI_KEYWORDS = ["AI", "LLM", "GPT", "Claude", "Gemini", "Llama",
               "transformer", "diffusion", "OpenAI", "Anthropic", "DeepMind"]


def scrape_hackernews(max_results: int = 10, hours_back: int = 24) -> list[NewsItem]:
    since  = int((dt.datetime.now() - dt.timedelta(hours=hours_back)).timestamp())
    params = {
        "query":         " OR ".join(AI_KEYWORDS),
        "tags":          "story",
        "numericFilters": f"created_at_i>{since},points>20",
        "hitsPerPage":   max_results,
    }
    try:
        resp = http_get(HN_SEARCH, params=params, timeout=20)
        hits = resp.json().get("hits", [])
    except Exception as e:
        logger.warning(f"HN scrape failed: {e}")
        return []

    items = []
    for h in hits:
        items.append(NewsItem(
            source="HackerNews",
            title=h.get("title", ""),
            url=h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
            summary=(h.get("story_text") or "")[:800],
            authors=[h.get("author", "")],
            published=h.get("created_at", "")[:10],
            score=min(h.get("points", 0) / 100, 2.0),
        ))
    logger.info(f"HackerNews: {len(items)} stories")
    return items


# ─────────────────── Official company blogs ───────────────────
#
# Direct blog scraping is blocked by all major AI companies (403).
# Instead we use Google News RSS search feeds — public, unblocked, and
# specifically filtered to each company's own domain so results are
# authoritative announcements, not just third-party coverage.

_GN_BASE = "https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q="

# (display_name, google_news_query, score)
_COMPANY_GN_FEEDS: list[tuple[str, str, float]] = [
    ("OpenAI",       'OpenAI site:openai.com',                          OFFICIAL_SCORE),
    ("Anthropic",    'Anthropic site:anthropic.com',                    OFFICIAL_SCORE),
    ("Google DeepMind", 'DeepMind site:deepmind.google',               OFFICIAL_SCORE),
    ("Microsoft AI", 'Microsoft AI site:blogs.microsoft.com',          OFFICIAL_SCORE),
    ("Meta AI",      'Meta AI site:ai.meta.com',                       OFFICIAL_SCORE - 0.5),
    ("DeepSeek",     'DeepSeek site:deepseek.com',                     OFFICIAL_SCORE - 0.5),
    ("Mistral",      'Mistral AI site:mistral.ai',                     OFFICIAL_SCORE - 0.5),
    ("xAI Grok",     'xAI Grok site:x.ai',                            OFFICIAL_SCORE - 0.5),
]


def _scrape_rss(name: str, feed_url: str, score: float,
                max_results: int = 5) -> list[NewsItem]:
    """Parse an RSS 2.0 or Atom feed using stdlib XML — no feedparser needed."""
    cutoff = dt.date.today() - dt.timedelta(days=_OFFICIAL_DAYS_BACK)
    try:
        resp = http_get(feed_url, timeout=20, headers=_UA)
        root = ET.fromstring(resp.content)
    except Exception as e:
        logger.warning(f"{name} RSS failed: {e}")
        return []

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    items: list[NewsItem] = []

    # RSS 2.0
    for item in root.findall(".//item"):
        title   = (item.findtext("title") or "").strip()
        url     = (item.findtext("link") or "").strip()
        summary = re.sub(r"<[^>]+>", " ", item.findtext("description") or "").strip()[:1200]
        pub_raw = item.findtext("pubDate") or item.findtext("dc:date", namespaces={"dc": "http://purl.org/dc/elements/1.1/"}) or ""
        pub     = _parse_date(pub_raw) or dt.date.today().isoformat()
        if dt.date.fromisoformat(pub) < cutoff:
            continue
        items.append(NewsItem(source=name, title=title, url=url, summary=summary,
                               authors=[], published=pub, score=score))
        if len(items) >= max_results:
            break

    # Atom (if no RSS items found)
    if not items:
        for entry in root.findall(".//atom:entry", ns):
            title   = (entry.findtext("atom:title", namespaces=ns) or "").strip()
            link_el = entry.find("atom:link", ns)
            url     = (link_el.get("href", "") if link_el is not None else "").strip()
            summary = re.sub(r"<[^>]+>", " ",
                             entry.findtext("atom:summary", namespaces=ns) or
                             entry.findtext("atom:content", namespaces=ns) or "").strip()[:1200]
            pub_raw = entry.findtext("atom:published", namespaces=ns) or entry.findtext("atom:updated", namespaces=ns) or ""
            pub     = _parse_date(pub_raw) or dt.date.today().isoformat()
            if dt.date.fromisoformat(pub) < cutoff:
                continue
            items.append(NewsItem(source=name, title=title, url=url, summary=summary,
                                   authors=[], published=pub, score=score))
            if len(items) >= max_results:
                break

    logger.info(f"{name} RSS: {len(items)} items")
    return items


def _deep_collect(obj: Any, target_keys: frozenset[str],
                  depth: int = 0) -> list[dict]:
    """Recursively collect dicts that contain any of target_keys."""
    if depth > 8 or not isinstance(obj, (dict, list)):
        return []
    if isinstance(obj, dict):
        found = target_keys & set(obj.keys())
        results = [obj] if found else []
        for v in obj.values():
            results.extend(_deep_collect(v, target_keys, depth + 1))
        return results
    return [r for item in obj for r in _deep_collect(item, target_keys, depth + 1)]


def _from_nextjs(html: str, base_url: str, name: str,
                 score: float, cutoff: dt.date) -> list[NewsItem]:
    """Extract posts from a Next.js __NEXT_DATA__ JSON blob."""
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return []

    date_keys = frozenset({"publishedAt", "published_at", "date", "createdAt",
                            "created_at", "postedAt", "posted_at"})
    candidates = _deep_collect(data, frozenset({"title", "slug"}) | date_keys)

    items: list[NewsItem] = []
    seen: set[str] = set()
    for c in candidates:
        title = (c.get("title") or c.get("heading") or "").strip()
        if not title or len(title) < 20 or title in seen:
            continue
        seen.add(title)

        slug = c.get("slug") or c.get("href") or c.get("url") or ""
        url  = urljoin(base_url, slug) if slug else base_url

        raw_date = next((c[k] for k in date_keys if k in c and c[k]), "")
        pub      = _parse_date(str(raw_date)) or dt.date.today().isoformat()
        try:
            if dt.date.fromisoformat(pub) < cutoff:
                continue
        except ValueError:
            pass

        summary = (c.get("description") or c.get("excerpt") or c.get("summary") or "")
        items.append(NewsItem(
            source=name, title=title, url=url,
            summary=str(summary).strip()[:1200],
            authors=[], published=pub, score=score,
        ))
        if len(items) >= 5:
            break
    return items


def _from_html_links(html: str, base_url: str, name: str,
                     score: float, cutoff: dt.date) -> list[NewsItem]:
    """Fallback: heuristic link extraction from blog HTML."""
    soup  = BeautifulSoup(html, "html.parser")
    items: list[NewsItem] = []
    seen:  set[str]       = set()

    for tag in soup.find_all(["article", "li", "div"], limit=120):
        a = tag.find("a", href=True)
        if not a:
            continue
        title = a.get_text(strip=True)
        if len(title) < 20:
            continue
        href = a["href"]
        url  = urljoin(base_url, href) if href.startswith("/") else href
        if url in seen or not url.startswith("http"):
            continue
        seen.add(url)

        # Try to extract date from URL path (e.g. /2025/01/15/ or /2025-04-26/)
        pub = dt.date.today().isoformat()
        dm  = re.search(r"(\d{4})[/-](\d{2})[/-](\d{2})", url)
        if dm:
            try:
                candidate = dt.date(int(dm[1]), int(dm[2]), int(dm[3]))
                if candidate < cutoff:
                    continue
                pub = candidate.isoformat()
            except ValueError:
                pass

        items.append(NewsItem(
            source=name, title=title, url=url,
            summary="", authors=[], published=pub, score=score,
        ))
        if len(items) >= 5:
            break
    return items


def _scrape_blog(name: str, url: str, score: float) -> list[NewsItem]:
    """Scrape a company blog page. Tries Next.js data first, then HTML."""
    cutoff = dt.date.today() - dt.timedelta(days=_OFFICIAL_DAYS_BACK)
    try:
        resp = http_get(url, timeout=20, headers=_UA)
    except Exception as e:
        logger.warning(f"{name} fetch failed: {e}")
        return []

    items = _from_nextjs(resp.text, url, name, score, cutoff)
    if not items:
        items = _from_html_links(resp.text, url, name, score, cutoff)

    logger.info(f"{name}: {len(items)} items")
    return items


def scrape_official_sources() -> list[NewsItem]:
    """Fetch official AI company news via Google News RSS. Items score ≥4.5."""
    all_items: list[NewsItem] = []
    for name, query, score in _COMPANY_GN_FEEDS:
        url = _GN_BASE + _url_quote(query)
        all_items.extend(_scrape_rss(name, url, score))
    return all_items


# ─────────────────── Aggregator ───────────────────

def scrape_all(top_n: int = 10) -> list[NewsItem]:
    """Run all sources. Official company announcements are prioritised;
    community/research items fill the remaining slots."""
    official  = scrape_official_sources()
    community = scrape_huggingface() + scrape_arxiv() + scrape_hackernews()
    all_items = official + community

    # Dedupe by lowercased title prefix
    seen:   set[str]       = set()
    unique: list[NewsItem] = []
    for it in all_items:
        key = it.title.lower().strip()[:60]
        if key and key not in seen:
            seen.add(key)
            unique.append(it)

    priority_items  = [x for x in unique if x.score >= 4.0]
    community_items = [x for x in unique if x.score <  4.0]
    priority_items.sort(key=lambda x: x.score, reverse=True)
    community_items.sort(key=lambda x: x.score, reverse=True)

    # Reserve at least 3 slots for community diversity
    max_priority = max(top_n - 3, top_n // 2 + 1)
    top = (priority_items[:max_priority] + community_items)[:top_n]

    n_official  = sum(1 for x in top if x.score >= 4.0)
    n_community = len(top) - n_official
    logger.info(
        f"Final selection: {len(top)} items "
        f"({n_official} official, {n_community} community)"
    )
    return top


if __name__ == "__main__":
    results = scrape_all()
    import json as _json
    print(_json.dumps([r.to_dict() for r in results], indent=2))
