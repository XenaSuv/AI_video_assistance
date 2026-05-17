"""Daily and monthly API cost aggregation with budget-limit alerts.

Reads cost_report.json files written by CostLedger.save() from all pipeline
run directories under output_dir, aggregates totals for the current day and
month, and returns a BudgetStatus that callers can use to fire Slack alerts.

Usage (in _finalize):
    guard = BudgetGuard(settings.output_dir, settings.daily_budget_usd,
                        settings.monthly_budget_usd)
    status = guard.check()
    if status.any_exceeded:
        notify_budget_alert(status, pipeline="daily")
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger


@dataclass
class BudgetStatus:
    date: str          # YYYY-MM-DD of the check
    day_usd: float     # total spend today (all pipelines)
    month_usd: float   # total spend this calendar month
    day_limit: float   # configured daily limit (0 = disabled)
    month_limit: float # configured monthly limit (0 = disabled)

    @property
    def day_exceeded(self) -> bool:
        return self.day_limit > 0 and self.day_usd > self.day_limit

    @property
    def month_exceeded(self) -> bool:
        return self.month_limit > 0 and self.month_usd > self.month_limit

    @property
    def any_exceeded(self) -> bool:
        return self.day_exceeded or self.month_exceeded

    @property
    def day_pct(self) -> float | None:
        """Percentage of daily budget consumed (None if limit disabled)."""
        return round(self.day_usd / self.day_limit * 100, 1) if self.day_limit else None

    @property
    def month_pct(self) -> float | None:
        return round(self.month_usd / self.month_limit * 100, 1) if self.month_limit else None


class BudgetGuard:
    """Aggregate cost_report.json files and check against budget limits."""

    def __init__(
        self,
        output_dir: Path,
        daily_limit: float = 50.0,
        monthly_limit: float = 500.0,
    ) -> None:
        self._output_dir = output_dir
        self.daily_limit  = daily_limit
        self.monthly_limit = monthly_limit

    # ── aggregation ───────────────────────────────────────────────────────────

    def aggregate_day(self, date: dt.date) -> float:
        """Sum all cost_report.json totals for *date* across all pipelines."""
        prefix = date.isoformat()           # "YYYY-MM-DD"
        return self._sum_reports(f"{prefix}*")

    def aggregate_month(self, year: int, month: int) -> float:
        """Sum all cost_report.json totals for a calendar month."""
        prefix = f"{year:04d}-{month:02d}"  # "YYYY-MM"
        return self._sum_reports(f"{prefix}-*")

    def _sum_reports(self, day_glob: str) -> float:
        total = 0.0
        for report_path in self._output_dir.glob(f"{day_glob}/**/cost_report.json"):
            try:
                data = json.loads(report_path.read_text())
                total += float(data.get("total_usd", 0.0))
            except Exception as exc:
                logger.debug(f"BudgetGuard: skipped {report_path.name}: {exc}")
        return round(total, 4)

    # ── public API ────────────────────────────────────────────────────────────

    def check(self, today: dt.date | None = None) -> BudgetStatus:
        """Return a BudgetStatus snapshot for *today* and its calendar month."""
        today = today or dt.date.today()
        return BudgetStatus(
            date        = today.isoformat(),
            day_usd     = self.aggregate_day(today),
            month_usd   = self.aggregate_month(today.year, today.month),
            day_limit   = self.daily_limit,
            month_limit = self.monthly_limit,
        )
