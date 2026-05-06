from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings
from src.performance_tracker import _load_db, _save_db


def _strategy_path() -> Path:
    return settings.data_dir / "content_strategy.json"


def _actions_path() -> Path:
    return settings.data_dir / "auto_actions.json"


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return copy.deepcopy(default)
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning(f"Failed to load {path.name}: {exc}")
        return copy.deepcopy(default)


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


@dataclass
class AutoActionEngine:
    run_type: str = "daily"
    max_actions: int = 2

    def default_strategy(self) -> dict[str, Any]:
        return {
            "pace": "normal",
            "hook_aggressiveness": 0.5,
            "packaging_style": "standard",
            "mode_locked": False,
        }

    def load_strategy(self) -> dict[str, Any]:
        store = _load_json(_strategy_path(), {"strategies": {}})
        strategy = store.get("strategies", {}).get(self.run_type)
        if not strategy:
            strategy = self.default_strategy()
            self.save_strategy(strategy)
        return strategy

    def save_strategy(self, strategy: dict[str, Any]) -> None:
        store = _load_json(_strategy_path(), {"strategies": {}})
        store.setdefault("strategies", {})[self.run_type] = strategy
        _save_json(_strategy_path(), store)

    def load_action_log(self) -> dict[str, Any]:
        return _load_json(_actions_path(), {"entries": [], "scores": {}})

    def save_action_log(self, payload: dict[str, Any]) -> None:
        _save_json(_actions_path(), payload)

    def generate_insights(self) -> list[dict[str, Any]]:
        records = [
            r for r in _load_db()
            if r.get("content_type", "daily") == self.run_type and r.get("views", 0) > 0
        ]
        if len(records) < 2:
            return []

        recent = records[-1]
        baseline = records[-5:-1] or records[:-1]
        if not baseline:
            return []

        avg_ctr = sum((r.get("ctr") or 0.0) for r in baseline) / len(baseline)
        avg_retention = sum((r.get("avg_view_percentage") or 0.0) for r in baseline) / len(baseline)
        recent_ctr = recent.get("ctr") or 0.0
        recent_retention = recent.get("avg_view_percentage") or 0.0

        insights: list[dict[str, Any]] = []
        if recent_retention and avg_retention and recent_retention < avg_retention - 5:
            insights.append({
                "type": "retention",
                "priority": "high",
                "value": recent_retention,
                "baseline": round(avg_retention, 2),
            })
        if recent_retention and recent_retention < 35:
            insights.append({
                "type": "drop",
                "priority": "high",
                "value": recent_retention,
                "baseline": 35.0,
            })
        if recent_ctr and avg_ctr and recent_ctr < avg_ctr - 0.75:
            insights.append({
                "type": "ctr",
                "priority": "high",
                "value": recent_ctr,
                "baseline": round(avg_ctr, 2),
            })
        return insights

    def performance_good(self, insights: list[dict[str, Any]]) -> bool:
        return not insights

    def evaluate_recent_actions(self) -> None:
        store = self.load_action_log()
        pending = next(
            (
                entry for entry in reversed(store.get("entries", []))
                if entry.get("run_type") == self.run_type and entry.get("evaluation_status") == "pending"
            ),
            None,
        )
        if not pending:
            return

        records = [
            r for r in _load_db()
            if r.get("content_type", "daily") == self.run_type and r.get("views", 0) > 0
        ]
        if len(records) < 2:
            return

        latest = records[-1]
        previous = records[-2]
        if latest.get("video_id") == pending.get("evaluated_video_id"):
            return

        prev_ret = previous.get("avg_view_percentage") or 0.0
        new_ret = latest.get("avg_view_percentage") or 0.0
        delta = round(new_ret - prev_ret, 4)

        scores = store.setdefault("scores", {})
        for action in pending.get("actions", []):
            action_type = action["type"]
            score = scores.get(action_type, 0)
            scores[action_type] = score + (1 if delta > 0 else -1)

        pending["evaluation_status"] = "done"
        pending["evaluated_video_id"] = latest.get("video_id")
        pending["evaluation"] = {
            "metric": "avg_view_percentage",
            "delta_retention": delta,
        }
        self.save_action_log(store)

    def _map(self, rec: dict[str, Any]) -> dict[str, Any] | None:
        if rec["type"] == "retention" and rec["priority"] == "high":
            return {"type": "increase_pacing", "reason": "high_retention", "source": rec["type"]}
        if rec["type"] == "drop":
            return {"type": "add_hook", "reason": "early_drop", "source": rec["type"]}
        if rec["type"] == "ctr":
            return {"type": "improve_packaging", "reason": "low_ctr", "source": rec["type"]}
        return None

    def _limit(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        limited: list[dict[str, Any]] = []
        for action in actions:
            if action["type"] in seen:
                continue
            seen.add(action["type"])
            limited.append(action)
            if len(limited) >= self.max_actions:
                break
        return limited

    def _apply(self, strategy: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
        updated = copy.deepcopy(strategy)
        for action in actions:
            if action["type"] == "increase_pacing":
                updated["pace"] = "fast"
            if action["type"] == "add_hook":
                updated["hook_aggressiveness"] = min(1.0, updated.get("hook_aggressiveness", 0.5) + 0.2)
            if action["type"] == "improve_packaging":
                updated["packaging_style"] = "conflict"
        return updated

    def apply(self, strategy: dict[str, Any], insights: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if strategy.get("mode_locked"):
            return strategy, []
        if self.performance_good(insights):
            return strategy, [{"type": "no_change", "reason": "performance_good", "source": "guard"}]

        actions: list[dict[str, Any]] = []
        for rec in insights:
            action = self._map(rec)
            if action:
                actions.append(action)
        actions = self._limit(actions)
        if not actions:
            return strategy, []
        return self._apply(strategy, actions), actions

    def prepare_strategy(self) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
        self.evaluate_recent_actions()
        before = self.load_strategy()
        insights = self.generate_insights()
        after, actions = self.apply(before, insights)
        if actions and actions[0]["type"] != "no_change":
            self.save_strategy(after)

        store = self.load_action_log()
        store.setdefault("entries", []).append({
            "timestamp": datetime.utcnow().isoformat(),
            "run_type": self.run_type,
            "insights": insights,
            "actions": actions,
            "before": before,
            "after": after,
            "evaluation_status": "pending" if actions and actions[0]["type"] != "no_change" else "skipped",
        })
        self.save_action_log(store)
        return after, actions, insights
