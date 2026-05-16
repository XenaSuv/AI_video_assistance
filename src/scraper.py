"""Scrape latest AI news from arXiv + HuggingFace + Hacker News +
official company blogs (OpenAI, Anthropic, Google DeepMind, Microsoft AI,
Meta AI, Mistral, DeepSeek, xAI).

Official company items receive a high base score so they are always
prioritised over community/research sources in the final selection.
"""
from __future__ import annotations

import concurrent.futures
import datetime as dt
import json
import re
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, quote as _url_quote

import xml.etree.ElementTree as ET

import arxiv
import requests
from bs4 import BeautifulSoup
from src.retry_utils import http_get
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


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


# ─────────────────── cache ───────────────────

class ScraperCache:
    """Per-source, per-day JSON cache stored in *cache_dir*.

    Each entry is ``{key}.json`` containing::

        {"date": "YYYY-MM-DD", "items": [...NewsItem dicts...]}

    A cached file is considered fresh only when its ``date`` field matches
    today.  Writes are atomic (write to ``.tmp`` then rename).
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir   = cache_dir
        self._today = dt.date.today().isoformat()
        self._lock  = threading.Lock()

    def get(self, key: str) -> list[NewsItem] | None:
        """Return cached items if fresh, else None."""
        with self._lock:
            path = self._dir / f"{key}.json"
            if not path.exists():
                return None
            try:
                data = json.loads(path.read_text())
            except Exception as exc:
                logger.warning(f"ScraperCache: unreadable cache for '{key}': {exc}")
                return None
            if data.get("date") != self._today:
                return None
            items = [NewsItem(**it) for it in data.get("items", [])]
            logger.debug(f"ScraperCache: hit '{key}' — {len(items)} items")
            return items

    def set(self, key: str, items: list[NewsItem]) -> None:
        """Persist *items* to cache."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{key}.json"
        tmp  = path.with_suffix(".tmp")
        with self._lock:
            try:
                tmp.write_text(json.dumps(
                    {"date": self._today, "items": [i.to_dict() for i in items]},
                    indent=2,
                ))
                tmp.replace(path)
                logger.debug(f"ScraperCache: saved '{key}' — {len(items)} items")
            except Exception as exc:
                logger.warning(f"ScraperCache: could not write cache for '{key}': {exc}")


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
    try:
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
    except Exception as e:
        logger.warning(f"arXiv fetch failed (skipping): {e}")
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
    try:
        resp = http_get("https://huggingface.co/papers", timeout=20)
    except Exception as e:
        logger.warning(f"HuggingFace HTML fallback failed (skipping): {e}")
        return []
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


def scrape_telegram_channels() -> list[NewsItem]:
    """Scrape Telegram channels configured in settings.telegram_rss_urls.

    Requires an RSS proxy such as RSSHub (https://rsshub.app/telegram/channel/<name>).
    Returns an empty list silently when the setting is blank.
    Items are scored at 4.5 so they can trigger breaking-news detection.
    """
    raw = (settings.telegram_rss_urls or "").strip()
    if not raw:
        return []
    items: list[NewsItem] = []
    for url in [u.strip() for u in raw.split(",") if u.strip()]:
        channel_name = url.rstrip("/").split("/")[-1]
        source_name  = f"Telegram/{channel_name}"
        items.extend(_scrape_rss(source_name, url, score=4.5))
    return items


def scrape_official_sources() -> list[NewsItem]:
    """Fetch official AI company news via Google News RSS + Telegram channels.

    All items score ≥4.5 and are eligible for breaking-news detection.
    """
    all_items: list[NewsItem] = []
    for name, query, score in _COMPANY_GN_FEEDS:
        url = _GN_BASE + _url_quote(query)
        all_items.extend(_scrape_rss(name, url, score))
    all_items.extend(scrape_telegram_channels())
    return all_items


# ─────────────────── Aggregator ───────────────────

def scrape_all(top_n: int = 10, cache_dir: Path | None = None) -> list[NewsItem]:
    """Run all sources. Official company announcements are prioritised;
    community/research items fill the remaining slots.

    Results for each source are cached for the rest of the calendar day in
    *cache_dir* (defaults to ``settings.data_dir / "scraper_cache"``).
    Pass ``cache_dir=False`` to disable caching entirely.
    """
    cache: ScraperCache | None = None
    if cache_dir is not False:
        cache = ScraperCache(cache_dir or (settings.data_dir / "scraper_cache"))

    def _cached(key: str, fn):
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                logger.info(f"{key}: {len(hit)} items (cached)")
                return hit
        try:
            result = fn()
        except Exception as e:
            logger.warning(f"{key}: source failed, skipping ({e})")
            return []
        if cache is not None:
            cache.set(key, result)
        return result

    sources = {
        "official_sources": scrape_official_sources,
        "huggingface":      scrape_huggingface,
        "arxiv":            scrape_arxiv,
        "hackernews":       scrape_hackernews,
    }
    results: dict[str, list[NewsItem]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as pool:
        futures = {pool.submit(_cached, key, fn): key for key, fn in sources.items()}
        for fut in concurrent.futures.as_completed(futures):
            key = futures[fut]
            results[key] = fut.result()

    official  = results["official_sources"]
    community = results["huggingface"] + results["arxiv"] + results["hackernews"]
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
