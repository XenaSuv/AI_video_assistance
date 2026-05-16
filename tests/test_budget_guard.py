"""Tests for src/budget_guard.py — cost aggregation and budget thresholds."""
from __future__ import annotations

import datetime as dt
import json

import pytest

from src.budget_guard import BudgetGuard, BudgetStatus


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_report(run_dir, total_usd: float) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cost_report.json").write_text(
        json.dumps({"total_usd": total_usd, "entries": []})
    )


TODAY = dt.date(2026, 5, 16)
MONTH = (TODAY.year, TODAY.month)


# ── BudgetGuard.aggregate_day ─────────────────────────────────────────────────

class TestAggregateDay:
    def test_zero_when_no_reports(self, tmp_path):
        g = BudgetGuard(tmp_path)
        assert g.aggregate_day(TODAY) == 0.0

    def test_sums_single_report(self, tmp_path):
        _write_report(tmp_path / TODAY.isoformat(), 12.50)
        g = BudgetGuard(tmp_path)
        assert g.aggregate_day(TODAY) == 12.50

    def test_sums_multiple_pipelines_same_day(self, tmp_path):
        _write_report(tmp_path / TODAY.isoformat() / "daily",   10.0)
        _write_report(tmp_path / TODAY.isoformat() / "shorts",   3.5)
        _write_report(tmp_path / TODAY.isoformat() / "breaking", 1.5)
        g = BudgetGuard(tmp_path)
        assert g.aggregate_day(TODAY) == 15.0

    def test_excludes_different_day(self, tmp_path):
        other = TODAY - dt.timedelta(days=1)
        _write_report(tmp_path / other.isoformat(), 100.0)
        _write_report(tmp_path / TODAY.isoformat(),  5.0)
        g = BudgetGuard(tmp_path)
        assert g.aggregate_day(TODAY) == 5.0

    def test_skips_unreadable_file(self, tmp_path):
        run_dir = tmp_path / TODAY.isoformat()
        run_dir.mkdir(parents=True)
        (run_dir / "cost_report.json").write_text("not valid json {{")
        g = BudgetGuard(tmp_path)
        assert g.aggregate_day(TODAY) == 0.0

    def test_rounds_to_4_decimals(self, tmp_path):
        _write_report(tmp_path / TODAY.isoformat() / "a", 0.333333)
        _write_report(tmp_path / TODAY.isoformat() / "b", 0.333333)
        g = BudgetGuard(tmp_path)
        result = g.aggregate_day(TODAY)
        assert isinstance(result, float)
        assert len(str(result).split(".")[-1]) <= 4


# ── BudgetGuard.aggregate_month ───────────────────────────────────────────────

class TestAggregateMonth:
    def test_zero_when_empty(self, tmp_path):
        g = BudgetGuard(tmp_path)
        assert g.aggregate_month(*MONTH) == 0.0

    def test_sums_across_days_in_month(self, tmp_path):
        for day in [1, 10, 20]:
            d = dt.date(TODAY.year, TODAY.month, day)
            _write_report(tmp_path / d.isoformat(), 10.0)
        g = BudgetGuard(tmp_path)
        assert g.aggregate_month(*MONTH) == 30.0

    def test_excludes_previous_month(self, tmp_path):
        prev = dt.date(TODAY.year, TODAY.month - 1, 1)
        _write_report(tmp_path / prev.isoformat(), 999.0)
        _write_report(tmp_path / TODAY.isoformat(),  5.0)
        g = BudgetGuard(tmp_path)
        assert g.aggregate_month(*MONTH) == 5.0


# ── BudgetStatus properties ───────────────────────────────────────────────────

class TestBudgetStatus:
    def _status(self, day_usd=0.0, month_usd=0.0, day_limit=50.0, month_limit=500.0):
        return BudgetStatus(
            date="2026-05-16",
            day_usd=day_usd,
            month_usd=month_usd,
            day_limit=day_limit,
            month_limit=month_limit,
        )

    def test_day_not_exceeded_when_under_limit(self):
        assert not self._status(day_usd=49.99).day_exceeded

    def test_day_exceeded_when_over_limit(self):
        assert self._status(day_usd=50.01).day_exceeded

    def test_month_not_exceeded_when_under_limit(self):
        assert not self._status(month_usd=499.99).month_exceeded

    def test_month_exceeded_when_over_limit(self):
        assert self._status(month_usd=500.01).month_exceeded

    def test_any_exceeded_false_when_both_ok(self):
        assert not self._status(day_usd=10.0, month_usd=100.0).any_exceeded

    def test_any_exceeded_true_when_day_over(self):
        assert self._status(day_usd=51.0).any_exceeded

    def test_any_exceeded_true_when_month_over(self):
        assert self._status(month_usd=501.0).any_exceeded

    def test_day_limit_zero_disables_day_check(self):
        assert not self._status(day_usd=9999.0, day_limit=0.0).day_exceeded

    def test_month_limit_zero_disables_month_check(self):
        assert not self._status(month_usd=9999.0, month_limit=0.0).month_exceeded

    def test_day_pct_calculated(self):
        s = self._status(day_usd=25.0, day_limit=50.0)
        assert s.day_pct == 50.0

    def test_month_pct_calculated(self):
        s = self._status(month_usd=100.0, month_limit=500.0)
        assert s.month_pct == 20.0

    def test_day_pct_none_when_limit_zero(self):
        assert self._status(day_limit=0.0).day_pct is None

    def test_month_pct_none_when_limit_zero(self):
        assert self._status(month_limit=0.0).month_pct is None


# ── BudgetGuard.check ─────────────────────────────────────────────────────────

class TestCheck:
    def test_returns_budget_status(self, tmp_path):
        g = BudgetGuard(tmp_path, daily_limit=50.0, monthly_limit=500.0)
        s = g.check(today=TODAY)
        assert isinstance(s, BudgetStatus)
        assert s.date == TODAY.isoformat()

    def test_status_reflects_actual_spend(self, tmp_path):
        _write_report(tmp_path / TODAY.isoformat(), 12.0)
        g = BudgetGuard(tmp_path, daily_limit=50.0, monthly_limit=500.0)
        s = g.check(today=TODAY)
        assert s.day_usd == 12.0

    def test_status_not_exceeded_when_under_limits(self, tmp_path):
        _write_report(tmp_path / TODAY.isoformat(), 5.0)
        g = BudgetGuard(tmp_path, daily_limit=50.0, monthly_limit=500.0)
        s = g.check(today=TODAY)
        assert not s.any_exceeded

    def test_status_exceeded_when_over_daily(self, tmp_path):
        _write_report(tmp_path / TODAY.isoformat(), 75.0)
        g = BudgetGuard(tmp_path, daily_limit=50.0, monthly_limit=500.0)
        s = g.check(today=TODAY)
        assert s.day_exceeded
        assert s.any_exceeded
