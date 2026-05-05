"""AI Video System Debug Dashboard.

Launch:
    streamlit run src/debug_dashboard.py

Reads data/run_history.json written by SystemOrchestrator on every cycle
and updated by learn_from_feedback() once analytics arrive.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import streamlit as st

# ── Path setup ────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "run_history.json"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Video Debug Dashboard",
    page_icon="🎬",
    layout="wide",
)

# ── Load data ─────────────────────────────────────────────────────────────────

@st.cache_data(ttl=10)
def load_runs() -> list[dict]:
    if not DATA_FILE.exists():
        return []
    try:
        return json.loads(DATA_FILE.read_text())
    except Exception:
        return []


runs = load_runs()

st.title("🎬 AI Video System Debug Dashboard")

if not runs:
    st.warning(f"No run data found at `{DATA_FILE}`.\nRun a production cycle first.")
    st.stop()

# ── Video selector ────────────────────────────────────────────────────────────

video_ids = [r.get("video_id") or f"(no id) run {i}" for i, r in enumerate(runs)]
selected_label = st.selectbox("Select video run", video_ids, index=len(video_ids) - 1)
run = runs[video_ids.index(selected_label)]

# ── Layout: two columns ───────────────────────────────────────────────────────

left, right = st.columns([1, 2])

# ── Left column: strategy + packaging ────────────────────────────────────────

with left:
    st.header("Strategy")
    strategy = run.get("strategy") or {}
    col_a, col_b = st.columns(2)
    col_a.metric("Mode", strategy.get("mode", "—"))
    col_b.metric("Pace", strategy.get("pace", "—"))
    st.caption(f"Scene policy: **{run.get('scene_policy', '—')}**")

    st.divider()

    st.header("Packaging")
    packaging = run.get("packaging") or {}
    st.write("**Title:**", packaging.get("title") or "—")
    st.write("**Hook:**", packaging.get("hook") or "—")
    thumb = packaging.get("thumbnail") or {}
    if thumb:
        st.json(thumb)

    st.divider()

    perf = run.get("performance") or {}
    if perf:
        st.header("Performance")
        c1, c2 = st.columns(2)
        ctr_val = perf.get("ctr", 0.0)
        watch_val = perf.get("avg_watch_pct", 0.0)
        c1.metric("CTR", f"{ctr_val:.1%}")
        c2.metric("Watch %", f"{watch_val:.1%}")

        drops = perf.get("drops") or []
        if drops:
            st.error(f"Drop scenes: {drops}")
        else:
            st.success("No retention drops recorded")
    else:
        st.info("Performance data not yet available (awaiting analytics).")

# ── Right column: scenes + insights ──────────────────────────────────────────

with right:
    scenes = run.get("scenes") or []

    st.header(f"Scene Timeline ({len(scenes)} scenes)")

    if scenes:
        HIGH = 0.6
        MED  = 0.4

        for s in scenes:
            risk    = s.get("predicted_risk", 0.0)
            label   = (
                "🔴 HIGH" if risk >= HIGH
                else "🟡 MED"  if risk >= MED
                else "🟢 LOW"
            )
            perf_drops = (perf.get("drops") or []) if perf else []
            dropped = s.get("index") in perf_drops

            badge = " ⚠️ **ACTUAL DROP**" if dropped else ""
            st.write(
                f"**Scene {s.get('index')}** | "
                f"`{s.get('type', '?')}` | "
                f"intent: `{s.get('intent', '?')}` | "
                f"risk {risk:.2f} {label}{badge}"
            )

        st.divider()

        # Scene type distribution
        st.header("Scene Distribution")
        type_counts = dict(collections.Counter(s.get("type", "unknown") for s in scenes))
        st.bar_chart(type_counts)

        st.divider()

        # Prediction error analysis
        if perf:
            st.header("Prediction Errors")
            perf_drops = perf.get("drops") or []
            missed     = [s for s in scenes if s.get("index") in perf_drops and s.get("predicted_risk", 1.0) < 0.5]
            false_alarms = [s for s in scenes if s.get("index") not in perf_drops and s.get("predicted_risk", 0.0) >= HIGH]

            if not missed and not false_alarms:
                st.success("No prediction errors — model aligned with reality.")
            for s in missed:
                st.error(
                    f"Missed drop at scene {s.get('index')} "
                    f"(predicted risk {s.get('predicted_risk'):.2f}, actual drop)"
                )
            for s in false_alarms:
                st.warning(
                    f"False alarm at scene {s.get('index')} "
                    f"(predicted risk {s.get('predicted_risk'):.2f}, no drop)"
                )

        st.divider()

        # Growth insights
        st.header("Insights")
        if perf:
            ctr_val   = perf.get("ctr", 0.0)
            watch_val = perf.get("avg_watch_pct", 0.0)
            drops     = perf.get("drops") or []

            if watch_val < 0.4:
                st.warning("Low retention → increase pacing or add micro-hooks")
            if ctr_val < 0.05:
                st.warning("Weak CTR → test new titles and thumbnail concepts")
            if drops:
                st.warning(f"Drops at scenes {drops} → review scene types at those positions")

            high_risk_count = sum(1 for s in scenes if s.get("predicted_risk", 0) >= HIGH)
            if high_risk_count > len(scenes) * 0.3:
                st.warning(f"{high_risk_count}/{len(scenes)} scenes predicted high-risk — consider restructuring")

            if watch_val >= 0.6 and ctr_val >= 0.06:
                st.success("Strong performance — replicate this strategy and packaging pattern")
        else:
            st.info("Insights available after analytics arrive.")
    else:
        st.info("No scene data for this run.")

# ── Cross-learning: top combinations ─────────────────────────────────────────

st.divider()
st.header("Top Performing Combinations")

@st.cache_data(ttl=10)
def load_top_combos(min_count: int = 3) -> list[dict]:
    try:
        sys.path.insert(0, str(ROOT))
        from src.cross_learning_engine import CrossLearningEngine
        engine = CrossLearningEngine(data_dir=ROOT / "data")
        return engine.get_top_combinations(min_count=min_count)
    except Exception:
        return []

min_obs = st.slider("Min observations per combo", 1, 10, 3)
top_combos = load_top_combos(min_obs)

if not top_combos:
    st.info(
        "Not enough cross-learning data yet. "
        f"Need ≥ {min_obs} videos with the same combination before patterns emerge."
    )
else:
    for i, c in enumerate(top_combos[:10]):
        combo   = c["combo"]
        avg_ret = c["avg_retention"]
        avg_ctr = c["avg_ctr"]
        count   = c["count"]

        ret_color = "🟢" if avg_ret >= 0.6 else "🟡" if avg_ret >= 0.4 else "🔴"
        label = (
            f"{ret_color} **#{i+1}** | "
            f"hook=`{combo.get('hook_type','?')}` "
            f"scene=`{combo.get('scene_type','?')}` "
            f"persona=`{combo.get('persona','?')}` "
            f"pace=`{combo.get('pace','?')}` "
            f"topic=`{combo.get('topic','?')}` "
            f"| ret={avg_ret:.1%} ctr={avg_ctr:.1%} (n={count})"
        )
        st.write(label)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
ts = run.get("timestamp", "—")
st.caption(f"Run timestamp: {ts} | Data file: `{DATA_FILE}`")
if st.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()
