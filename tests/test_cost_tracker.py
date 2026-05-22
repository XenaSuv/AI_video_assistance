"""Tests for CostLedger."""
import json
from pathlib import Path

import pytest

from src.cost_tracker import CostLedger, get_ledger, reset_ledger


@pytest.fixture(autouse=True)
def fresh_ledger():
    """Each test gets a clean ledger."""
    reset_ledger()
    yield
    reset_ledger()


# ── record_llm ────────────────────────────────────────────────────────────────

class TestRecordLlm:
    def test_records_entry(self):
        ledger = get_ledger()
        ledger.record_llm("script", "gpt-4o-mini", 1000, 500)
        assert len(ledger._entries) == 1
        e = ledger._entries[0]
        assert e.tag == "script"
        assert e.kind == "llm"
        assert e.prompt_tokens == 1000
        assert e.completion_tokens == 500

    def test_cost_calculated_correctly_gpt4o_mini(self):
        ledger = get_ledger()
        # 1M input @ $0.15 + 1M output @ $0.60
        ledger.record_llm("x", "gpt-4o-mini", 1_000_000, 1_000_000)
        assert abs(ledger.total_usd() - 0.75) < 0.001

    def test_cached_tokens_use_lower_price(self):
        ledger = get_ledger()
        # All tokens cached: 1M @ $0.075
        ledger.record_llm("x", "gpt-4o-mini", 1_000_000, 0, cached_tokens=1_000_000)
        assert abs(ledger.total_usd() - 0.075) < 0.001

    def test_unknown_model_falls_back_to_gpt4o_mini_prices(self):
        ledger = get_ledger()
        ledger.record_llm("x", "unknown-model-xyz", 1_000_000, 0)
        assert ledger.total_usd() > 0

    def test_multiple_entries_sum_correctly(self):
        ledger = get_ledger()
        ledger.record_llm("a", "gpt-4o-mini", 1000, 500)
        ledger.record_llm("b", "gpt-4o-mini", 2000, 1000)
        assert len(ledger._entries) == 2
        assert ledger.total_usd() > 0


# ── record_tts ────────────────────────────────────────────────────────────────

class TestRecordTts:
    def test_records_entry(self):
        ledger = get_ledger()
        ledger.record_tts("tts-en", 5000, "eleven_turbo_v2_5")
        e = ledger._entries[0]
        assert e.kind == "tts"
        assert e.chars == 5000

    def test_cost_calculated_correctly(self):
        ledger = get_ledger()
        # 1000 chars @ $0.30/1K = $0.30
        ledger.record_tts("tts-en", 1000, "eleven_turbo_v2_5")
        assert abs(ledger.total_usd() - 0.30) < 0.001

    def test_unknown_model_defaults_to_0_30(self):
        ledger = get_ledger()
        ledger.record_tts("x", 1000, "unknown-tts-model")
        assert abs(ledger.total_usd() - 0.30) < 0.001


# ── record_image ──────────────────────────────────────────────────────────────

class TestRecordImage:
    def test_records_entry(self):
        ledger = get_ledger()
        ledger.record_image("image-scene_01", "dall-e-3")
        e = ledger._entries[0]
        assert e.kind == "image"
        assert e.n == 1

    def test_cost_dall_e_3(self):
        ledger = get_ledger()
        ledger.record_image("x", "dall-e-3")
        assert abs(ledger.total_usd() - 0.080) < 0.001

    def test_n_multiplies_cost(self):
        ledger = get_ledger()
        ledger.record_image("x", "dall-e-3", n=2)
        assert abs(ledger.total_usd() - 0.160) < 0.001


# ── summary ───────────────────────────────────────────────────────────────────

class TestSummary:
    def test_empty_ledger(self):
        s = get_ledger().summary()
        assert s["total_usd"] == 0.0
        assert s["entries"] == []

    def test_by_tag_aggregates(self):
        ledger = get_ledger()
        ledger.record_llm("script", "gpt-4o-mini", 1000, 500)
        ledger.record_llm("script", "gpt-4o-mini", 500, 200)
        s = ledger.summary()
        assert "script" in s["by_tag"]
        assert s["by_tag"]["script"]["usd"] > 0

    def test_by_kind_separates_llm_tts_image(self):
        ledger = get_ledger()
        ledger.record_llm("a", "gpt-4o-mini", 1000, 500)
        ledger.record_tts("b", 1000, "eleven_turbo_v2_5")
        ledger.record_image("c", "dall-e-3")
        s = ledger.summary()
        assert "llm" in s["by_kind"]
        assert "tts" in s["by_kind"]
        assert "image" in s["by_kind"]

    def test_total_equals_sum_of_entries(self):
        ledger = get_ledger()
        ledger.record_llm("a", "gpt-4o-mini", 10000, 5000)
        ledger.record_tts("b", 2000, "eleven_turbo_v2_5")
        s = ledger.summary()
        entry_total = sum(e["usd"] for e in s["entries"])
        assert abs(s["total_usd"] - entry_total) < 1e-6

    def test_cache_hit_rate_computed(self):
        ledger = get_ledger()
        # 500k cached out of 1M prompt = 50% hit rate
        ledger.record_llm("x", "gpt-4o-mini", 1_000_000, 0, cached_tokens=500_000)
        s = ledger.summary()
        assert s["by_kind"]["llm"]["cache_hit_rate"] == 50.0

    def test_cache_hit_rate_zero_when_no_cache(self):
        ledger = get_ledger()
        ledger.record_llm("x", "gpt-4o-mini", 1_000_000, 0)
        s = ledger.summary()
        assert s["by_kind"]["llm"]["cache_hit_rate"] == 0.0

    def test_cache_savings_usd_computed(self):
        ledger = get_ledger()
        # 1M cached on gpt-4o-mini: input=$0.150, cached=$0.075 → savings=$0.075
        ledger.record_llm("x", "gpt-4o-mini", 1_000_000, 0, cached_tokens=1_000_000)
        s = ledger.summary()
        assert abs(s["cache_savings_usd"] - 0.075) < 1e-6
        assert abs(s["by_kind"]["llm"]["cache_savings_usd"] - 0.075) < 1e-6

    def test_cache_savings_usd_zero_when_no_cache(self):
        ledger = get_ledger()
        ledger.record_llm("x", "gpt-4o-mini", 1_000_000, 0)
        s = ledger.summary()
        assert s["cache_savings_usd"] == 0.0

    def test_entry_includes_cache_savings_usd(self):
        ledger = get_ledger()
        ledger.record_llm("x", "gpt-4o-mini", 1_000_000, 0, cached_tokens=1_000_000)
        s = ledger.summary()
        assert "cache_savings_usd" in s["entries"][0]

    def test_cache_hit_rate_zero_for_empty_ledger(self):
        s = get_ledger().summary()
        # no llm entries → by_kind["llm"] filtered out; top-level savings = 0
        assert s["cache_savings_usd"] == 0.0


# ── save ──────────────────────────────────────────────────────────────────────

class TestSave:
    def test_writes_valid_json(self, tmp_path):
        ledger = get_ledger()
        ledger.record_llm("script", "gpt-4o-mini", 5000, 2000)
        report = ledger.save(tmp_path / "cost_report.json")
        saved = json.loads((tmp_path / "cost_report.json").read_text())
        assert saved["total_usd"] == report["total_usd"]

    def test_creates_parent_dirs(self, tmp_path):
        ledger = get_ledger()
        deep = tmp_path / "a" / "b" / "cost_report.json"
        ledger.save(deep)
        assert deep.exists()


# ── singleton ─────────────────────────────────────────────────────────────────

class TestSingleton:
    def test_get_ledger_returns_same_instance(self):
        a = get_ledger()
        b = get_ledger()
        assert a is b

    def test_reset_creates_fresh_instance(self):
        get_ledger().record_llm("x", "gpt-4o-mini", 100, 50)
        reset_ledger()
        assert get_ledger().total_usd() == 0.0

    def test_custom_prices_override_defaults(self):
        custom = CostLedger(custom_prices={"llm": {"gpt-4o-mini": {"input": 1.0, "output": 2.0, "cached": 0.5}}})
        custom.record_llm("x", "gpt-4o-mini", 1_000_000, 0)
        assert abs(custom.total_usd() - 1.0) < 0.001

    def test_custom_prices_savings_use_overridden_rates(self):
        # custom input=1.0, cached=0.5 → savings per 1M cached = $0.50
        custom = CostLedger(custom_prices={"llm": {"gpt-4o-mini": {"input": 1.0, "output": 2.0, "cached": 0.5}}})
        custom.record_llm("x", "gpt-4o-mini", 1_000_000, 0, cached_tokens=1_000_000)
        s = custom.summary()
        assert abs(s["cache_savings_usd"] - 0.5) < 1e-6
