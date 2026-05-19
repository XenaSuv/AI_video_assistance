"""Tests for src/scraper.py."""
from __future__ import annotations

import datetime as dt
import json
from unittest.mock import MagicMock, patch

import pytest

from src.scraper import (
    OFFICIAL_SCORE,
    NewsItem,
    ScraperCache,
    _deep_collect,
    _from_html_links,
    _from_nextjs,
    _parse_date,
    _scrape_rss,
    scrape_all,
    scrape_hackernews,
    scrape_huggingface,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_resp(json_data=None, text="", content=b"", status_code=200):
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    r.content = content
    r.json.return_value = json_data if json_data is not None else {}
    return r


def _rss_xml(items: list[dict], today_str: str | None = None) -> bytes:
    """Build a minimal RSS 2.0 feed."""
    today = today_str or dt.date.today().isoformat()
    item_xml = ""
    for it in items:
        item_xml += f"""
        <item>
            <title>{it.get('title', 'Test')}</title>
            <link>{it.get('url', 'https://example.com')}</link>
            <description>{it.get('summary', '')}</description>
            <pubDate>{it.get('pub', today)}</pubDate>
        </item>"""
    return f"""<?xml version="1.0"?>
<rss version="2.0"><channel>{item_xml}</channel></rss>""".encode()


def _atom_xml(entries: list[dict], today_str: str | None = None) -> bytes:
    """Build a minimal Atom feed."""
    today = today_str or dt.date.today().isoformat()
    ns = 'xmlns="http://www.w3.org/2005/Atom"'
    entry_xml = ""
    for e in entries:
        entry_xml += f"""
        <entry>
            <title>{e.get('title', 'Test')}</title>
            <link href="{e.get('url', 'https://example.com')}"/>
            <summary>{e.get('summary', '')}</summary>
            <published>{e.get('pub', today)}</published>
        </entry>"""
    return f"""<?xml version="1.0"?>
<feed {ns}>{entry_xml}</feed>""".encode()


# ── _parse_date ────────────────────────────────────────────────────────────────

class TestParseDate:
    def test_iso_format(self):
        assert _parse_date("2025-01-15") == "2025-01-15"

    def test_slash_format(self):
        assert _parse_date("2025/01/15") == "2025-01-15"

    def test_embedded_in_text(self):
        assert _parse_date("Published on 2025-04-20T10:00:00Z") == "2025-04-20"

    def test_empty_string_returns_none(self):
        assert _parse_date("") is None

    def test_none_returns_none(self):
        assert _parse_date(None) is None

    def test_no_date_returns_none(self):
        assert _parse_date("no date here") is None

    def test_invalid_date_returns_none(self):
        assert _parse_date("2025-13-45") is None

    def test_date_with_extra_text(self):
        assert _parse_date("Mon, 21 Jan 2025-01-21 00:00:00 GMT") == "2025-01-21"


# ── _deep_collect ──────────────────────────────────────────────────────────────

class TestDeepCollect:
    def test_finds_top_level(self):
        data = {"title": "Hello", "other": 1}
        results = _deep_collect(data, frozenset({"title"}))
        assert any(r.get("title") == "Hello" for r in results)

    def test_finds_nested(self):
        data = {"wrapper": {"inner": {"title": "Deep", "slug": "/post"}}}
        results = _deep_collect(data, frozenset({"title", "slug"}))
        assert any(r.get("title") == "Deep" for r in results)

    def test_finds_in_list(self):
        data = {"posts": [{"title": "A"}, {"title": "B"}]}
        results = _deep_collect(data, frozenset({"title"}))
        titles = {r.get("title") for r in results}
        assert {"A", "B"}.issubset(titles)

    def test_empty_data_returns_empty(self):
        assert _deep_collect({}, frozenset({"title"})) == []

    def test_depth_limit(self):
        # Build a very deeply nested structure — should not crash
        data: dict = {}
        node = data
        for _ in range(15):
            node["child"] = {}
            node = node["child"]
        node["title"] = "deep"
        # Should not raise and return some result (depth-limited)
        result = _deep_collect(data, frozenset({"title"}))
        assert isinstance(result, list)


# ── _from_nextjs ──────────────────────────────────────────────────────────────

class TestFromNextjs:
    def _make_html(self, data: dict) -> str:
        blob = json.dumps(data)
        return f'<script id="__NEXT_DATA__" type="application/json">{blob}</script>'

    def test_extracts_items(self):
        today = dt.date.today().isoformat()
        html = self._make_html({"props": {"posts": [
            {"title": "OpenAI releases new model today here", "slug": "/blog/new-model",
             "publishedAt": today, "description": "desc"},
        ]}})
        items = _from_nextjs(html, "https://openai.com", "OpenAI", 5.0,
                             dt.date.today() - dt.timedelta(days=3))
        assert len(items) == 1
        assert items[0].title == "OpenAI releases new model today here"
        assert items[0].source == "OpenAI"

    def test_skips_old_items(self):
        old_date = (dt.date.today() - dt.timedelta(days=30)).isoformat()
        html = self._make_html({"posts": [
            {"title": "Very old post that should be skipped", "slug": "/old",
             "publishedAt": old_date},
        ]})
        items = _from_nextjs(html, "https://example.com", "Test", 5.0,
                             dt.date.today() - dt.timedelta(days=3))
        assert items == []

    def test_skips_short_titles(self):
        today = dt.date.today().isoformat()
        html = self._make_html({"posts": [
            {"title": "Short", "slug": "/x", "publishedAt": today},
        ]})
        items = _from_nextjs(html, "https://example.com", "Test", 5.0,
                             dt.date.today() - dt.timedelta(days=3))
        assert items == []

    def test_no_next_data_returns_empty(self):
        items = _from_nextjs("<html><body>no data</body></html>",
                             "https://example.com", "Test", 5.0, dt.date.today())
        assert items == []

    def test_invalid_json_returns_empty(self):
        html = '<script id="__NEXT_DATA__">{broken json</script>'
        items = _from_nextjs(html, "https://example.com", "Test", 5.0, dt.date.today())
        assert items == []

    def test_deduplicates_by_title(self):
        today = dt.date.today().isoformat()
        html = self._make_html({"posts": [
            {"title": "Same title appears twice in the data", "slug": "/a", "publishedAt": today},
            {"title": "Same title appears twice in the data", "slug": "/b", "publishedAt": today},
        ]})
        items = _from_nextjs(html, "https://example.com", "Test", 5.0,
                             dt.date.today() - dt.timedelta(days=3))
        assert len(items) == 1

    def test_max_5_items(self):
        today = dt.date.today().isoformat()
        posts = [{"title": f"Article number {i} with a long enough title", "slug": f"/{i}",
                  "publishedAt": today} for i in range(10)]
        html = self._make_html({"posts": posts})
        items = _from_nextjs(html, "https://example.com", "Test", 5.0,
                             dt.date.today() - dt.timedelta(days=3))
        assert len(items) <= 5


# ── _from_html_links ──────────────────────────────────────────────────────────

class TestFromHtmlLinks:
    def _cutoff(self):
        return dt.date.today() - dt.timedelta(days=3)

    def test_extracts_article_links(self):
        html = """<html><body>
            <article><a href="https://openai.com/blog/great-new-model-release">
                OpenAI releases a great new model that everyone is talking about
            </a></article>
        </body></html>"""
        items = _from_html_links(html, "https://openai.com", "OpenAI", 5.0, self._cutoff())
        assert len(items) == 1
        assert "openai.com" in items[0].url

    def test_skips_short_titles(self):
        html = '<html><body><article><a href="https://example.com/x">Short</a></article></body></html>'
        items = _from_html_links(html, "https://example.com", "Test", 5.0, self._cutoff())
        assert items == []

    def test_deduplicates_by_url(self):
        html = """<html><body>
            <article><a href="https://example.com/same-url">
                This is a long enough title to pass the filter
            </a></article>
            <article><a href="https://example.com/same-url">
                This is a long enough title to pass the filter
            </a></article>
        </body></html>"""
        items = _from_html_links(html, "https://example.com", "Test", 5.0, self._cutoff())
        assert len(items) == 1

    def test_skips_old_url_dates(self):
        old = (dt.date.today() - dt.timedelta(days=30)).isoformat().replace("-", "/")
        html = f"""<html><body>
            <article><a href="https://example.com/{old}/old-post">
                This very old post should be excluded from results
            </a></article>
        </body></html>"""
        items = _from_html_links(html, "https://example.com", "Test", 5.0, self._cutoff())
        assert items == []

    def test_resolves_relative_urls(self):
        html = """<html><body>
            <article><a href="/blog/ai-research-new-post">
                New AI Research Post That Is Long Enough To Be Included
            </a></article>
        </body></html>"""
        items = _from_html_links(html, "https://example.com", "Test", 5.0, self._cutoff())
        assert len(items) == 1
        assert items[0].url.startswith("https://example.com")

    def test_max_5_items(self):
        links = "".join(
            f'<article><a href="https://example.com/post-{i}">'
            f'This is article number {i} with a long enough title</a></article>'
            for i in range(10)
        )
        html = f"<html><body>{links}</body></html>"
        items = _from_html_links(html, "https://example.com", "Test", 5.0, self._cutoff())
        assert len(items) <= 5


# ── _scrape_rss ───────────────────────────────────────────────────────────────

class TestScrapeRss:
    def test_parses_rss2_items(self):
        today = dt.date.today().isoformat()
        xml = _rss_xml([
            {"title": "Big AI announcement", "url": "https://openai.com/1", "pub": today},
            {"title": "Another AI story here", "url": "https://openai.com/2", "pub": today},
        ])
        with patch("src.scraper.http_get", return_value=_mock_resp(content=xml)):
            items = _scrape_rss("OpenAI", "https://feed.example.com", OFFICIAL_SCORE)
        assert len(items) == 2
        assert items[0].source == "OpenAI"
        assert items[0].score == OFFICIAL_SCORE

    def test_filters_old_items(self):
        old = (dt.date.today() - dt.timedelta(days=10)).isoformat()
        today = dt.date.today().isoformat()
        xml = _rss_xml([
            {"title": "Recent story", "pub": today},
            {"title": "Old story",    "pub": old},
        ])
        with patch("src.scraper.http_get", return_value=_mock_resp(content=xml)):
            items = _scrape_rss("OpenAI", "https://feed.example.com", OFFICIAL_SCORE)
        assert len(items) == 1
        assert items[0].title == "Recent story"

    def test_parses_atom_feed(self):
        today = dt.date.today().isoformat()
        xml = _atom_xml([{"title": "Atom entry title", "url": "https://deepmind.google/1",
                          "pub": today}])
        with patch("src.scraper.http_get", return_value=_mock_resp(content=xml)):
            items = _scrape_rss("DeepMind", "https://feed.example.com", OFFICIAL_SCORE)
        assert len(items) == 1
        assert items[0].title == "Atom entry title"

    def test_http_error_returns_empty(self):
        import requests
        with patch("src.scraper.http_get",
                   side_effect=requests.exceptions.ConnectionError("timeout")):
            items = _scrape_rss("OpenAI", "https://feed.example.com", OFFICIAL_SCORE)
        assert items == []

    def test_respects_max_results(self):
        today = dt.date.today().isoformat()
        xml = _rss_xml([{"title": f"Story {i}", "pub": today} for i in range(10)])
        with patch("src.scraper.http_get", return_value=_mock_resp(content=xml)):
            items = _scrape_rss("OpenAI", "https://feed.example.com", OFFICIAL_SCORE,
                                max_results=3)
        assert len(items) <= 3


# ── scrape_huggingface ────────────────────────────────────────────────────────

class TestScrapeHuggingface:
    def _hf_entry(self, title: str, upvotes: int = 10, paper_id: str = "abc123") -> dict:
        return {
            "paper": {
                "id": paper_id,
                "title": title,
                "summary": "A paper summary.",
                "upvotes": upvotes,
                "authors": [{"name": "Author A"}],
            },
            "publishedAt": dt.date.today().isoformat(),
            "numComments": 0,
        }

    def test_returns_news_items(self):
        data = [self._hf_entry("Large language model breakthrough paper")]
        with patch("src.scraper.http_get", return_value=_mock_resp(json_data=data)):
            items = scrape_huggingface(max_results=5)
        assert len(items) == 1
        assert items[0].source == "HuggingFace"
        assert items[0].url == "https://huggingface.co/papers/abc123"

    def test_score_capped_at_2(self):
        data = [self._hf_entry("Title", upvotes=10_000)]
        with patch("src.scraper.http_get", return_value=_mock_resp(json_data=data)):
            items = scrape_huggingface()
        assert items[0].score <= 2.0

    def test_respects_max_results(self):
        data = [self._hf_entry(f"Paper {i}", paper_id=str(i)) for i in range(20)]
        with patch("src.scraper.http_get", return_value=_mock_resp(json_data=data)):
            items = scrape_huggingface(max_results=5)
        assert len(items) == 5

    def test_falls_back_to_html_on_api_error(self):
        html = """<html><body>
            <article><a href="/papers/xyz">Great paper title here</a></article>
        </body></html>"""
        import requests
        with patch("src.scraper.http_get", side_effect=[
            requests.exceptions.ConnectionError(),
            _mock_resp(text=html),
        ]):
            items = scrape_huggingface()
        assert len(items) >= 1
        assert items[0].source == "HuggingFace"

    def test_authors_truncated_to_5(self):
        entry = self._hf_entry("Paper with many authors")
        entry["paper"]["authors"] = [{"name": f"Author {i}"} for i in range(10)]
        with patch("src.scraper.http_get", return_value=_mock_resp(json_data=[entry])):
            items = scrape_huggingface()
        assert len(items[0].authors) <= 5


# ── scrape_hackernews ─────────────────────────────────────────────────────────

class TestScrapeHackernews:
    def _hn_hit(self, title: str, points: int = 50, obj_id: str = "1") -> dict:
        return {
            "title": title,
            "url": f"https://example.com/{obj_id}",
            "story_text": "Some story text.",
            "author": "hnuser",
            "created_at": dt.date.today().isoformat(),
            "points": points,
            "objectID": obj_id,
        }

    def test_returns_news_items(self):
        data = {"hits": [self._hn_hit("OpenAI launches new thing")]}
        with patch("src.scraper.http_get", return_value=_mock_resp(json_data=data)):
            items = scrape_hackernews()
        assert len(items) == 1
        assert items[0].source == "HackerNews"

    def test_score_calculated_from_points(self):
        data = {"hits": [self._hn_hit("OpenAI news title", points=100)]}
        with patch("src.scraper.http_get", return_value=_mock_resp(json_data=data)):
            items = scrape_hackernews()
        assert abs(items[0].score - 1.0) < 0.01

    def test_score_capped_at_2(self):
        data = {"hits": [self._hn_hit("Viral post", points=100_000)]}
        with patch("src.scraper.http_get", return_value=_mock_resp(json_data=data)):
            items = scrape_hackernews()
        assert items[0].score <= 2.0

    def test_falls_back_to_hn_url_when_no_url(self):
        hit = self._hn_hit("Post without external URL")
        hit["url"] = None
        data = {"hits": [hit]}
        with patch("src.scraper.http_get", return_value=_mock_resp(json_data=data)):
            items = scrape_hackernews()
        assert "ycombinator.com" in items[0].url

    def test_network_error_returns_empty(self):
        import requests
        with patch("src.scraper.http_get",
                   side_effect=requests.exceptions.ConnectionError()):
            items = scrape_hackernews()
        assert items == []

    def test_empty_hits_returns_empty(self):
        with patch("src.scraper.http_get", return_value=_mock_resp(json_data={"hits": []})):
            items = scrape_hackernews()
        assert items == []


# ── scrape_all ────────────────────────────────────────────────────────────────

class TestScrapeAll:
    def _make_item(self, title: str, score: float, source: str = "Test") -> NewsItem:
        return NewsItem(source=source, title=title,
                        url=f"https://example.com/{title[:10]}",
                        summary="", authors=[],
                        published=dt.date.today().isoformat(),
                        score=score)

    def test_deduplicates_by_title_prefix(self):
        items = [
            self._make_item("OpenAI releases GPT-5 today finally", 5.0),
            self._make_item("OpenAI releases GPT-5 today finally", 4.0),  # duplicate
        ]
        with patch("src.scraper.scrape_official_sources", return_value=items[:1]), \
             patch("src.scraper.scrape_huggingface", return_value=[items[1]]), \
             patch("src.scraper.scrape_arxiv", return_value=[]), \
             patch("src.scraper.scrape_hackernews", return_value=[]):
            result = scrape_all(top_n=10, cache_dir=False)
        titles = [r.title for r in result]
        assert len(titles) == len(set(t.lower()[:60] for t in titles))

    def test_priority_items_come_first(self):
        official = [self._make_item(f"Official story {i} about AI", OFFICIAL_SCORE)
                    for i in range(5)]
        community = [self._make_item(f"Community paper {i} about ML", 1.0)
                     for i in range(5)]
        with patch("src.scraper.scrape_official_sources", return_value=official), \
             patch("src.scraper.scrape_huggingface", return_value=community), \
             patch("src.scraper.scrape_arxiv", return_value=[]), \
             patch("src.scraper.scrape_hackernews", return_value=[]):
            result = scrape_all(top_n=10, cache_dir=False)
        scores = [r.score for r in result]
        # At least one official item should appear
        assert any(s >= 4.0 for s in scores)

    def test_reserves_community_slots(self):
        official = [self._make_item(f"Official story number {i}", OFFICIAL_SCORE)
                    for i in range(10)]
        community = [self._make_item(f"Community paper number {i}", 1.0)
                     for i in range(5)]
        with patch("src.scraper.scrape_official_sources", return_value=official), \
             patch("src.scraper.scrape_huggingface", return_value=community), \
             patch("src.scraper.scrape_arxiv", return_value=[]), \
             patch("src.scraper.scrape_hackernews", return_value=[]):
            result = scrape_all(top_n=10, cache_dir=False)
        community_count = sum(1 for r in result if r.score < 4.0)
        assert community_count >= 3

    def test_respects_top_n(self):
        items = [self._make_item(f"Story number {i} about AI", float(i))
                 for i in range(20)]
        with patch("src.scraper.scrape_official_sources", return_value=items), \
             patch("src.scraper.scrape_huggingface", return_value=[]), \
             patch("src.scraper.scrape_arxiv", return_value=[]), \
             patch("src.scraper.scrape_hackernews", return_value=[]):
            result = scrape_all(top_n=7, cache_dir=False)
        assert len(result) <= 7

    def test_returns_list_when_all_sources_empty(self):
        with patch("src.scraper.scrape_official_sources", return_value=[]), \
             patch("src.scraper.scrape_huggingface", return_value=[]), \
             patch("src.scraper.scrape_arxiv", return_value=[]), \
             patch("src.scraper.scrape_hackernews", return_value=[]):
            result = scrape_all(cache_dir=False)
        assert result == []


# ── ScraperCache ─────────────────────────────────────────────────────────────

class TestScraperCache:
    def _item(self, title: str = "Test paper about AI") -> NewsItem:
        return NewsItem(source="HuggingFace", title=title,
                        url="https://example.com/paper",
                        summary="A summary.", authors=["Author"],
                        published=dt.date.today().isoformat(), score=1.0)

    def test_miss_on_empty_dir(self, tmp_path):
        cache = ScraperCache(tmp_path / "cache")
        assert cache.get("huggingface") is None

    def test_set_and_get_returns_same_items(self, tmp_path):
        cache = ScraperCache(tmp_path)
        items = [self._item("Paper one"), self._item("Paper two")]
        cache.set("huggingface", items)
        result = cache.get("huggingface")
        assert result is not None
        assert len(result) == 2
        assert result[0].title == "Paper one"

    def test_roundtrip_preserves_all_fields(self, tmp_path):
        cache = ScraperCache(tmp_path)
        original = self._item("Detailed paper title here")
        cache.set("arxiv", [original])
        loaded = cache.get("arxiv")[0]
        assert loaded.source    == original.source
        assert loaded.title     == original.title
        assert loaded.url       == original.url
        assert loaded.score     == original.score
        assert loaded.authors   == original.authors
        assert loaded.published == original.published

    def test_stale_cache_returns_none(self, tmp_path):
        cache = ScraperCache(tmp_path)
        # Write a cache file with yesterday's date
        yesterday = (dt.date.today() - dt.timedelta(days=1)).isoformat()
        path = tmp_path / "huggingface.json"
        path.write_text(json.dumps({"date": yesterday, "items": []}))
        assert cache.get("huggingface") is None

    def test_today_cache_is_fresh(self, tmp_path):
        cache = ScraperCache(tmp_path)
        items = [self._item()]
        cache.set("hackernews", items)
        result = cache.get("hackernews")
        assert result is not None and len(result) == 1

    def test_cache_file_is_valid_json(self, tmp_path):
        cache = ScraperCache(tmp_path)
        cache.set("arxiv", [self._item()])
        data = json.loads((tmp_path / "arxiv.json").read_text())
        assert "date" in data
        assert "items" in data
        assert data["date"] == dt.date.today().isoformat()

    def test_creates_cache_dir_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "cache"
        cache = ScraperCache(nested)
        cache.set("test", [self._item()])
        assert (nested / "test.json").exists()

    def test_scrape_all_uses_cache(self, tmp_path):
        """Second call should not invoke scrape functions again."""
        items = [self._item(f"Story number {i} about AI news") for i in range(3)]
        cache = ScraperCache(tmp_path)
        for key in ("official_sources", "huggingface", "arxiv", "hackernews"):
            cache.set(key, items if key == "official_sources" else [])

        with patch("src.scraper.scrape_official_sources") as mock_official, \
             patch("src.scraper.scrape_huggingface") as mock_hf, \
             patch("src.scraper.scrape_arxiv") as mock_ax, \
             patch("src.scraper.scrape_hackernews") as mock_hn:
            result = scrape_all(top_n=10, cache_dir=tmp_path)

        mock_official.assert_not_called()
        mock_hf.assert_not_called()
        mock_ax.assert_not_called()
        mock_hn.assert_not_called()
        assert len(result) == len(items)

    def test_scrape_all_populates_cache(self, tmp_path):
        """First call with empty cache should write results to disk."""
        items = [self._item(f"Official AI news story {i}") for i in range(3)]
        with patch("src.scraper.scrape_official_sources", return_value=items), \
             patch("src.scraper.scrape_huggingface", return_value=[]), \
             patch("src.scraper.scrape_arxiv", return_value=[]), \
             patch("src.scraper.scrape_hackernews", return_value=[]):
            scrape_all(top_n=10, cache_dir=tmp_path)

        assert (tmp_path / "official_sources.json").exists()
        cached = ScraperCache(tmp_path).get("official_sources")
        assert cached is not None and len(cached) == 3

    def test_corrupted_cache_file_returns_none(self, tmp_path):
        path = tmp_path / "huggingface.json"
        path.write_text("{corrupted json[[[")
        cache = ScraperCache(tmp_path)
        assert cache.get("huggingface") is None


# ── NewsItem ──────────────────────────────────────────────────────────────────

class TestNewsItem:
    def test_to_dict_has_all_fields(self):
        item = NewsItem(source="arXiv", title="Test", url="https://x.com",
                        summary="s", authors=["A"], published="2025-01-01", score=1.0)
        d = item.to_dict()
        assert d["source"] == "arXiv"
        assert d["score"] == 1.0
        assert d["authors"] == ["A"]

    def test_default_score_is_zero(self):
        item = NewsItem(source="x", title="t", url="u", summary="s",
                        authors=[], published="2025-01-01")
        assert item.score == 0.0


# ── Concurrent scraping ───────────────────────────────────────────────────────

class TestScrapeAllConcurrent:
    """Verify that scrape_all() fetches all sources and assembles results correctly
    when sources run in parallel via ThreadPoolExecutor."""

    def _item(self, title: str, score: float = 1.0) -> NewsItem:
        return NewsItem(source="Test", title=title, url=f"https://x.com/{title[:8]}",
                        summary="", authors=[], published=dt.date.today().isoformat(),
                        score=score)

    def test_all_sources_are_called(self):
        """Each source function must be called exactly once."""
        calls: list[str] = []

        def _src(name):
            def inner():
                calls.append(name)
                return []
            return inner

        with patch("src.scraper.scrape_official_sources", side_effect=_src("official")), \
             patch("src.scraper.scrape_huggingface",       side_effect=_src("hf")), \
             patch("src.scraper.scrape_arxiv",             side_effect=_src("arxiv")), \
             patch("src.scraper.scrape_hackernews",        side_effect=_src("hn")):
            scrape_all(cache_dir=False)

        assert sorted(calls) == ["arxiv", "hf", "hn", "official"]

    def test_results_from_all_sources_included(self):
        official   = [self._item("Official AI story", score=5.0)]
        hf_items   = [self._item("HuggingFace paper", score=2.0)]
        arxiv_items = [self._item("arXiv preprint X", score=1.5)]
        hn_items   = [self._item("HackerNews thread", score=1.0)]

        with patch("src.scraper.scrape_official_sources", return_value=official), \
             patch("src.scraper.scrape_huggingface",       return_value=hf_items), \
             patch("src.scraper.scrape_arxiv",             return_value=arxiv_items), \
             patch("src.scraper.scrape_hackernews",        return_value=hn_items):
            result = scrape_all(top_n=10, cache_dir=False)

        titles = {r.title for r in result}
        assert "Official AI story" in titles
        assert "HuggingFace paper" in titles
        assert "arXiv preprint X" in titles
        assert "HackerNews thread" in titles

    def test_failing_source_does_not_abort_others(self):
        """A source that raises must not prevent other sources from returning data."""
        good_item = self._item("Good community item")

        with patch("src.scraper.scrape_official_sources", side_effect=RuntimeError("network")), \
             patch("src.scraper.scrape_huggingface",       return_value=[good_item]), \
             patch("src.scraper.scrape_arxiv",             return_value=[]), \
             patch("src.scraper.scrape_hackernews",        return_value=[]):
            result = scrape_all(top_n=10, cache_dir=False)

        assert any(r.title == "Good community item" for r in result)

    def test_concurrent_cache_writes_are_safe(self, tmp_path):
        """Multiple threads writing different keys to the same ScraperCache
        must not corrupt each other's files."""
        import threading

        cache = ScraperCache(tmp_path)
        errors: list[Exception] = []

        def write(key: str, n: int) -> None:
            items = [self._item(f"{key} item {i}") for i in range(n)]
            try:
                cache.set(key, items)
            except Exception as exc:
                errors.append(exc)

        keys = ["official_sources", "huggingface", "arxiv", "hackernews"]
        threads = [threading.Thread(target=write, args=(k, 3)) for k in keys]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Cache writes raised: {errors}"
        for key in keys:
            result = cache.get(key)
            assert result is not None, f"Key '{key}' missing after concurrent write"
            assert len(result) == 3

    def test_all_sources_fail_returns_empty(self):
        with patch("src.scraper.scrape_official_sources", side_effect=Exception("err")), \
             patch("src.scraper.scrape_huggingface",       side_effect=Exception("err")), \
             patch("src.scraper.scrape_arxiv",             side_effect=Exception("err")), \
             patch("src.scraper.scrape_hackernews",        side_effect=Exception("err")):
            result = scrape_all(top_n=10, cache_dir=False)
        assert result == []
