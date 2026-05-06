"""Tests for DecisionEngineV3 — unified signals → strategy pipeline.

Covers:
- UnifiedStrategy dataclass: to_dict / from_dict roundtrip
- _collect_signals(): defaults, high-risk filter, missing keys
- _select_mode(): all four modes + priority ordering
- _build_strategy(): per-mode fields (pace, hook_aggressiveness, persona)
- _select_scene_policy(): bandit-full vs blend on retention threshold
- _blend_with_default(): 50/50 mix, normalisation, missing keys
- _safe_scene_policy(): SAFE_SCENE_MIX returned exactly
- _select_packaging(): CTR < 0.05 forces "conflict"; bandit type forwarded
- _apply_stability(): mode_locked triggers at STABILITY_WINDOW, not before
- _apply_corrections(): drops → action set; predicted_risks → preemptive_fix
- _apply_corrections(): no drops + no risks → empty actions
- decide(): end-to-end happy paths for each mode
- decide(): result persisted in history via DecisionHistoryStore
- load_history(): roundtrip through JSONL
- DecisionHistoryStore: append + load_recent ordering
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.decision_engine_v3 import (
    DEFAULT_SCENE_MIX,
    SAFE_SCENE_MIX,
    DecisionEngineV3,
    DecisionHistoryStore,
    UnifiedStrategy,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


def _engine(data_dir: Path | None = None) -> DecisionEngineV3:
    return DecisionEngineV3(data_dir=data_dir or _tmp())


def _ctx(
    retention:    float      = 0.6,
    ctr:          float      = 0.07,
    drops:        list[int]  | None = None,
    retention_30s: float     = 0.0,
    scene_bandit: dict       | None = None,
    pkg_bandit:   dict       | None = None,
    risks:        list[dict] | None = None,
    history:      list[dict] | None = None,
) -> dict:
    return {
        "metrics": {
            "avg_watch_pct": retention,
            "ctr":           ctr,
            "drops":         drops or [],
            "retention_30s": retention_30s,
        },
        "bandit": {
            "scene":     scene_bandit or {},
            "packaging": pkg_bandit   or {},
        },
        "prediction": {"risks": risks or []},
        "history":    history or [],
    }


def _bandit_mix(mix: dict) -> dict:
    return {"scene_mix": mix, "preferred_policy_id": "policy_X", "best_mean": 0.7}


def _pkg_bandit(t: str = "curiosity") -> dict:
    return {"preferred_variant_type": t, "best_mean": 0.6}


def _risk(prob: float, idx: int = 0) -> dict:
    return {"scene_idx": idx, "probability": prob}


# ── UnifiedStrategy ────────────────────────────────────────────────────────────

class TestUnifiedStrategy:
    def test_to_dict_roundtrip(self):
        s = UnifiedStrategy(
            mode="growth", pace="fast", hook_aggressiveness=0.9,
            scene_policy={"avatar": 0.5, "image": 0.5},
            packaging_style="conflict", persona="energetic",
            actions=["shorten_scenes"], mode_locked=True,
        )
        d = s.to_dict()
        s2 = UnifiedStrategy.from_dict(d)
        assert s2.mode                == "growth"
        assert s2.pace                == "fast"
        assert s2.hook_aggressiveness == 0.9
        assert s2.packaging_style     == "conflict"
        assert s2.persona             == "energetic"
        assert s2.actions             == ["shorten_scenes"]
        assert s2.mode_locked         is True

    def test_from_dict_defaults(self):
        s = UnifiedStrategy.from_dict({})
        assert s.mode                == "stable"
        assert s.pace                == "normal"
        assert s.hook_aggressiveness == 0.5
        assert s.packaging_style     == "curiosity"
        assert s.persona             == "balanced"
        assert s.actions             == []
        assert s.mode_locked         is False


# ── _collect_signals ───────────────────────────────────────────────────────────

class TestCollectSignals:
    def _signals(self, **kw):
        engine = _engine()
        return engine._collect_signals(_ctx(**kw))

    def test_metrics_extracted(self):
        s = self._signals(retention=0.45, ctr=0.06, drops=[1, 2])
        assert abs(s["retention"] - 0.45) < 1e-9
        assert abs(s["ctr"]       - 0.06) < 1e-9
        assert s["drops"]     == [1, 2]

    def test_high_risk_filtered(self):
        risks = [_risk(0.8), _risk(0.3), _risk(0.65)]
        s = self._signals(risks=risks)
        assert len(s["predicted_risks"]) == 2
        for r in s["predicted_risks"]:
            assert r["probability"] >= 0.6

    def test_empty_context_returns_safe_defaults(self):
        engine = _engine()
        s = engine._collect_signals({})
        assert s["retention"]        == 0.0
        assert s["ctr"]              == 0.0
        assert s["drops"]            == []
        assert s["predicted_risks"]  == []
        assert s["scene_bandit"]     == {}
        assert s["packaging_bandit"] == {}


# ── _select_mode ───────────────────────────────────────────────────────────────

class TestSelectMode:
    def _mode(self, **kw):
        engine = _engine()
        return engine._select_mode(engine._collect_signals(_ctx(**kw)))

    def test_low_retention_gives_growth(self):
        assert self._mode(retention=0.35) == "growth"

    def test_retention_boundary_above_not_growth(self):
        # 0.40 is NOT < 0.4
        assert self._mode(retention=0.40, ctr=0.06) != "growth"

    def test_low_ctr_gives_packaging_focus(self):
        assert self._mode(retention=0.55, ctr=0.04) == "packaging_focus"

    def test_many_drops_gives_retention_fix(self):
        assert self._mode(retention=0.55, ctr=0.06, drops=[1, 2, 3]) == "retention_fix"

    def test_exactly_two_drops_not_retention_fix(self):
        # > 2 required, so exactly 2 → stable
        mode = self._mode(retention=0.55, ctr=0.06, drops=[1, 2])
        assert mode == "stable"

    def test_stable_when_all_good(self):
        assert self._mode(retention=0.65, ctr=0.08, drops=[]) == "stable"

    def test_retention_takes_priority_over_ctr(self):
        # both < threshold → retention wins (checked first)
        assert self._mode(retention=0.30, ctr=0.03) == "growth"

    def test_ctr_takes_priority_over_drops(self):
        # ctr bad AND many drops → packaging_focus wins (checked first)
        assert self._mode(retention=0.55, ctr=0.04, drops=[1, 2, 3]) == "packaging_focus"


# ── _build_strategy ────────────────────────────────────────────────────────────

class TestBuildStrategy:
    def _strategy(self, mode: str, **ctx_kw) -> UnifiedStrategy:
        engine = _engine()
        s = engine._collect_signals(_ctx(**ctx_kw))
        return engine._build_strategy(mode, s)

    def test_growth_pace_fast(self):
        st = self._strategy("growth", retention=0.3)
        assert st.pace == "fast"

    def test_growth_hook_aggressiveness(self):
        st = self._strategy("growth", retention=0.3)
        assert abs(st.hook_aggressiveness - 0.9) < 1e-9

    def test_growth_persona(self):
        assert self._strategy("growth", retention=0.3).persona == "energetic"

    def test_packaging_focus_hook_max(self):
        st = self._strategy("packaging_focus", retention=0.5, ctr=0.03)
        assert abs(st.hook_aggressiveness - 1.0) < 1e-9

    def test_packaging_focus_forces_conflict_packaging(self):
        # packaging_focus always uses conflict regardless of CTR rule
        st = self._strategy("packaging_focus", retention=0.5, ctr=0.07)
        assert st.packaging_style == "conflict"

    def test_packaging_focus_persona(self):
        assert self._strategy("packaging_focus", retention=0.5, ctr=0.03).persona == "curious"

    def test_retention_fix_safe_policy(self):
        st = self._strategy("retention_fix", retention=0.5, ctr=0.06, drops=[1, 2, 3])
        assert st.scene_policy == SAFE_SCENE_MIX

    def test_retention_fix_pace_fast(self):
        st = self._strategy("retention_fix", retention=0.5, ctr=0.06, drops=[1, 2, 3])
        assert st.pace == "fast"

    def test_retention_fix_persona(self):
        st = self._strategy("retention_fix", retention=0.5, ctr=0.06, drops=[1, 2, 3])
        assert st.persona == "clear_explainer"

    def test_stable_persona(self):
        assert self._strategy("stable").persona == "balanced"

    def test_stable_pace_normal(self):
        assert self._strategy("stable").pace == "normal"


# ── _select_scene_policy ───────────────────────────────────────────────────────

class TestSelectScenePolicy:
    def test_low_retention_uses_bandit_fully(self):
        bandit_mix = {"avatar": 0.7, "image": 0.3}
        engine = _engine()
        s = engine._collect_signals(_ctx(
            retention=0.3,
            scene_bandit=_bandit_mix(bandit_mix),
        ))
        policy = engine._select_scene_policy(s)
        assert policy == bandit_mix

    def test_normal_retention_blends(self):
        bandit_mix = {"avatar": 0.6, "text_overlay": 0.4}
        engine = _engine()
        s = engine._collect_signals(_ctx(
            retention=0.55,
            scene_bandit=_bandit_mix(bandit_mix),
        ))
        policy = engine._select_scene_policy(s)
        # Should NOT equal bandit_mix exactly — it is blended
        assert policy != bandit_mix

    def test_no_bandit_returns_default(self):
        engine = _engine()
        s = engine._collect_signals(_ctx(retention=0.3))
        policy = engine._select_scene_policy(s)
        assert policy == DEFAULT_SCENE_MIX


# ── _blend_with_default ────────────────────────────────────────────────────────

class TestBlendWithDefault:
    def _blend(self, suggestion: dict) -> dict:
        return _engine()._blend_with_default(suggestion)

    def test_empty_suggestion_returns_default(self):
        assert self._blend({}) == DEFAULT_SCENE_MIX

    def test_blended_values_sum_to_one(self):
        mix = _bandit_mix({"avatar": 0.8, "image": 0.2})
        result = self._blend(mix)
        assert abs(sum(result.values()) - 1.0) < 1e-4

    def test_50_50_average_of_overlapping_keys(self):
        # bandit: avatar=1.0, default: avatar=0.30 → blend ≈ 0.65 (before norm)
        bandit = {"avatar": 1.0}
        result = self._blend({"scene_mix": bandit})
        # avatar should be between 0.30 and 1.0
        assert 0.30 < result["avatar"] < 1.0

    def test_extra_bandit_keys_included(self):
        # bandit adds a scene type not in DEFAULT_SCENE_MIX
        bandit = {"avatar": 0.5, "b_roll": 0.5}
        result = self._blend({"scene_mix": bandit})
        assert "b_roll" in result


# ── _select_packaging ──────────────────────────────────────────────────────────

class TestSelectPackaging:
    def _pkg(self, ctr: float, bandit_type: str = "curiosity") -> str:
        engine = _engine()
        s = engine._collect_signals(_ctx(ctr=ctr, pkg_bandit=_pkg_bandit(bandit_type)))
        return engine._select_packaging(s)

    def test_low_ctr_forces_conflict(self):
        assert self._pkg(ctr=0.03, bandit_type="simple") == "conflict"

    def test_exactly_at_threshold_forces_conflict(self):
        # < 0.05 → conflict; 0.05 is NOT < 0.05
        assert self._pkg(ctr=0.05, bandit_type="simple") != "conflict"

    def test_good_ctr_uses_bandit_type(self):
        assert self._pkg(ctr=0.08, bandit_type="curiosity") == "curiosity"

    def test_good_ctr_conflict_bandit(self):
        assert self._pkg(ctr=0.08, bandit_type="conflict") == "conflict"

    def test_unknown_bandit_type_falls_back_to_curiosity(self):
        assert self._pkg(ctr=0.08, bandit_type="unknown") == "curiosity"


# ── _apply_stability ───────────────────────────────────────────────────────────

class TestApplyStability:
    WINDOW = DecisionEngineV3.STABILITY_WINDOW

    def _stable(self, history: list[dict], mode: str = "stable") -> UnifiedStrategy:
        engine = _engine()
        st = UnifiedStrategy(mode=mode)
        return engine._apply_stability(st, {"history": history})

    def test_no_history_not_locked(self):
        assert self._stable([]).mode_locked is False

    def test_below_window_not_locked(self):
        h = [{"mode": "stable"}] * (self.WINDOW - 1)
        assert self._stable(h).mode_locked is False

    def test_exactly_at_window_same_mode_locked(self):
        h = [{"mode": "stable"}] * self.WINDOW
        assert self._stable(h).mode_locked is True

    def test_mixed_modes_not_locked(self):
        h = [{"mode": "stable"}, {"mode": "growth"}, {"mode": "stable"}]
        assert self._stable(h).mode_locked is False

    def test_locked_only_when_history_matches_current_mode(self):
        # history all "growth", current strategy is "stable" → not locked
        h = [{"mode": "growth"}] * self.WINDOW
        st = self._stable(h, mode="stable")
        assert st.mode_locked is False

    def test_more_than_window_uses_last_n(self):
        # 5 entries: first 2 are different, last 3 are stable
        h = [{"mode": "growth"}, {"mode": "growth"},
             {"mode": "stable"}, {"mode": "stable"}, {"mode": "stable"}]
        assert self._stable(h, mode="stable").mode_locked is True


# ── _apply_corrections ─────────────────────────────────────────────────────────

class TestApplyCorrections:
    def _actions(self, drops=None, risks=None) -> list[str]:
        engine = _engine()
        s = engine._collect_signals(_ctx(drops=drops or [], risks=risks or []))
        st = UnifiedStrategy()
        return engine._apply_corrections(st, s).actions

    def test_no_signals_empty_actions(self):
        assert self._actions() == []

    def test_drops_trigger_action_set(self):
        actions = self._actions(drops=[1])
        assert "shorten_scenes"  in actions
        assert "add_hooks"       in actions
        assert "increase_avatar" in actions

    def test_high_risk_triggers_preemptive_fix(self):
        assert "preemptive_fix" in self._actions(risks=[_risk(0.8)])

    def test_low_risk_not_preemptive(self):
        assert "preemptive_fix" not in self._actions(risks=[_risk(0.3)])

    def test_both_signals_combined(self):
        actions = self._actions(drops=[2], risks=[_risk(0.9)])
        for a in ("shorten_scenes", "add_hooks", "increase_avatar", "preemptive_fix"):
            assert a in actions

    def test_actions_are_sorted(self):
        actions = self._actions(drops=[1], risks=[_risk(0.9)])
        assert actions == sorted(actions)

    def test_no_duplicate_actions(self):
        actions = self._actions(drops=[1], risks=[_risk(0.9)])
        assert len(actions) == len(set(actions))


# ── decide() end-to-end ───────────────────────────────────────────────────────

class TestDecideEndToEnd:
    def test_growth_mode_full(self):
        engine = _engine()
        result = engine.decide(_ctx(retention=0.3, ctr=0.07))
        assert result.mode   == "growth"
        assert result.pace   == "fast"
        assert result.persona == "energetic"

    def test_packaging_focus_mode_full(self):
        engine = _engine()
        result = engine.decide(_ctx(retention=0.55, ctr=0.04))
        assert result.mode            == "packaging_focus"
        assert result.packaging_style == "conflict"

    def test_retention_fix_mode_full(self):
        engine = _engine()
        result = engine.decide(_ctx(retention=0.55, ctr=0.07, drops=[1, 2, 3]))
        assert result.mode         == "retention_fix"
        assert result.scene_policy == SAFE_SCENE_MIX

    def test_stable_mode_full(self):
        engine = _engine()
        result = engine.decide(_ctx(retention=0.65, ctr=0.08))
        assert result.mode == "stable"

    def test_actions_populated_from_drops(self):
        engine = _engine()
        result = engine.decide(_ctx(retention=0.55, ctr=0.07, drops=[1, 2, 3]))
        assert "shorten_scenes" in result.actions

    def test_mode_locked_after_history(self):
        engine = _engine()
        history = [{"mode": "stable"}] * engine.STABILITY_WINDOW
        result = engine.decide(_ctx(retention=0.65, ctr=0.08, history=history))
        assert result.mode_locked is True

    def test_result_persisted_to_history(self):
        d = _tmp()
        engine = DecisionEngineV3(data_dir=d)
        engine.decide(_ctx())
        hist = engine.load_history()
        assert len(hist) == 1
        assert hist[0]["mode"] in {"stable", "growth", "packaging_focus", "retention_fix"}

    def test_returns_unified_strategy_instance(self):
        engine = _engine()
        result = engine.decide(_ctx())
        assert isinstance(result, UnifiedStrategy)


# ── DecisionHistoryStore ───────────────────────────────────────────────────────

class TestDecisionHistoryStore:
    def test_append_and_load(self):
        d = _tmp()
        store = DecisionHistoryStore(d)
        store.append(UnifiedStrategy(mode="growth"))
        store.append(UnifiedStrategy(mode="stable"))
        records = store.load_recent(10)
        assert len(records) == 2
        assert records[0]["mode"] == "growth"
        assert records[1]["mode"] == "stable"

    def test_load_recent_respects_n(self):
        d = _tmp()
        store = DecisionHistoryStore(d)
        for _ in range(5):
            store.append(UnifiedStrategy())
        assert len(store.load_recent(3)) == 3

    def test_load_empty_returns_empty_list(self):
        assert DecisionHistoryStore(_tmp()).load_recent() == []

    def test_records_have_timestamp(self):
        d = _tmp()
        store = DecisionHistoryStore(d)
        store.append(UnifiedStrategy())
        record = store.load_recent(1)[0]
        assert "_ts" in record

    def test_load_history_via_engine(self):
        d = _tmp()
        engine = DecisionEngineV3(data_dir=d)
        engine.decide(_ctx(retention=0.3))
        hist = engine.load_history()
        assert hist[0]["mode"] == "growth"
