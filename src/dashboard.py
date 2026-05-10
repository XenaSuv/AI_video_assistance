"""Real-time AI pipeline monitor.

Run with:
    streamlit run src/dashboard.py

Auto-refreshes every REFRESH_SEC seconds by calling st.rerun().
Reads data/live_state.json written by LiveState during pipeline execution.
"""
from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

# ── config ────────────────────────────────────────────────────────────────────

REFRESH_SEC = 10
STATE_FILE  = Path(__file__).resolve().parent.parent / "data" / "live_state.json"

STEP_ICONS = {
    "running": "🔄",
    "done":    "✅",
    "skipped": "⏭️",
    "failed":  "❌",
    "pending": "⏳",
}

STATUS_COLORS = {
    "idle":                 "#888888",
    "running":              "#1f77b4",
    "success":              "#2ca02c",
    "failed":               "#d62728",
    "quality_gate_failed":  "#ff7f0e",
}

STEP_ORDER = [
    "scrape", "script", "voice", "subtitles",
    "video", "thumbnail", "upload_en", "upload_ru",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_state() -> dict:
    try:
        import json
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {
            "pipeline": None, "status": "idle", "started_at": None,
            "updated_at": None, "current_step": None,
            "steps_done": 0, "steps_total": 0, "progress": 0.0,
            "strategy": {}, "steps": [], "logs": [], "metrics": {}, "cost": {},
        }


def _status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#888888")
    label = status.upper().replace("_", " ")
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f'border-radius:12px;font-size:13px;font-weight:600">{label}</span>'
    )


def _fmt_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    if sec < 60:
        return f"{sec:.0f}s"
    return f"{sec / 60:.1f}m"


def _steps_map(state: dict) -> dict[str, dict]:
    """Return a name→record dict, filling in pending steps from STEP_ORDER."""
    recorded = {s["name"]: s for s in state.get("steps", [])}
    result: dict[str, dict] = {}
    for name in STEP_ORDER:
        result[name] = recorded.get(name, {"name": name, "status": "pending"})
    # Append any steps not in STEP_ORDER (e.g. extra languages)
    for name, rec in recorded.items():
        if name not in result:
            result[name] = rec
    return result


# ── page layout ───────────────────────────────────────────────────────────────

def render(state: dict) -> None:
    status   = state.get("status", "idle")
    pipeline = state.get("pipeline") or "—"

    # ── header ────────────────────────────────────────────────────────────────
    col_title, col_refresh = st.columns([4, 1])
    with col_title:
        st.title("🎬 AI Pipeline Monitor")
        st.markdown(
            f"**Pipeline:** `{pipeline}` &nbsp;&nbsp;"
            + _status_badge(status)
            + f"&nbsp;&nbsp;<small style='color:#888'>Updated: "
            f"{state.get('updated_at') or 'never'}</small>",
            unsafe_allow_html=True,
        )
    with col_refresh:
        st.caption(f"Auto-refresh: {REFRESH_SEC}s")
        if st.button("↻ Refresh now"):
            st.rerun()

    st.divider()

    # ── progress ──────────────────────────────────────────────────────────────
    progress   = float(state.get("progress", 0.0))
    steps_done = state.get("steps_done", 0)
    steps_total = state.get("steps_total", 0)
    current    = state.get("current_step")

    col_prog, col_cur = st.columns([3, 2])
    with col_prog:
        st.subheader("Progress")
        st.progress(progress)
        st.caption(f"{steps_done} / {steps_total} steps complete ({progress*100:.0f}%)")
    with col_cur:
        st.subheader("Current step")
        if current:
            st.info(f"{STEP_ICONS['running']} **{current}**")
        elif status == "success":
            st.success("Pipeline finished")
        elif status in ("failed", "quality_gate_failed"):
            st.error("Pipeline stopped")
        else:
            st.caption("—")

    st.divider()

    # ── strategy ──────────────────────────────────────────────────────────────
    strategy = state.get("strategy") or {}
    if strategy:
        st.subheader("Strategy")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Mode", strategy.get("mode", "—"))
        sc2.metric("Exploration", f"{strategy.get('exploration_rate', 0):.0%}")
        sc3.metric("Top angle", (strategy.get("top_angle") or "—").replace("_", " "))
        sc4.metric("Confidence", f"{strategy.get('confidence', 0):.0%}")
        st.divider()

    # ── steps table ───────────────────────────────────────────────────────────
    st.subheader("Steps")
    steps = _steps_map(state)
    rows = []
    for name, rec in steps.items():
        s = rec.get("status", "pending")
        icon = STEP_ICONS.get(s, "⏳")
        dur = _fmt_duration(rec.get("duration_sec"))
        err = rec.get("error", "")
        rows.append({
            "": icon,
            "Step": name,
            "Status": s,
            "Duration": dur,
            "Error": err,
        })
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "":        st.column_config.TextColumn(width="small"),
            "Step":    st.column_config.TextColumn(width="medium"),
            "Status":  st.column_config.TextColumn(width="small"),
            "Duration":st.column_config.TextColumn(width="small"),
            "Error":   st.column_config.TextColumn(width="large"),
        },
    )

    # ── metrics + cost ────────────────────────────────────────────────────────
    metrics = state.get("metrics") or {}
    cost    = state.get("cost") or {}
    if metrics or cost:
        st.divider()
        mcols = st.columns(4)
        col_i = 0
        if "video_id" in metrics:
            mcols[col_i].metric("Video ID", metrics["video_id"])
            col_i += 1
        if "views" in metrics:
            mcols[col_i].metric("Views", metrics["views"])
            col_i += 1
        if "ctr" in metrics:
            mcols[col_i].metric("CTR", f"{metrics['ctr']:.1%}")
            col_i += 1
        if "total_usd" in cost:
            mcols[col_i % 4].metric("API cost", f"${cost['total_usd']:.3f}")

    # ── event log ─────────────────────────────────────────────────────────────
    logs = state.get("logs") or []
    if logs:
        st.divider()
        with st.expander(f"Event log ({len(logs)} entries)", expanded=True):
            for entry in reversed(logs[-30:]):
                t = entry.get("time", "")
                m = entry.get("msg", "")
                st.markdown(f"`{t}` {m}")

    # ── hook patterns section ─────────────────────────────────────────────────
    st.divider()
    render_hook_patterns()


# ── hook patterns ────────────────────────────────────────────────────────────

def _load_hook_patterns() -> tuple[list[dict], list[dict]]:
    """Load top patterns and best combos from HookPatternEngine."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.hook_pattern_engine import HookPatternEngine
        engine   = HookPatternEngine()
        patterns = [p.to_dict() for p in engine.get_top_patterns(n=10)]
        combos   = [c.to_dict() for c in engine.get_best_combo(n=10)]
        return patterns, combos
    except Exception:
        return [], []


def render_hook_patterns() -> None:
    st.header("Hook Patterns")

    patterns, combos = _load_hook_patterns()

    if not patterns:
        st.caption("No hook pattern data yet — runs after Shorts collect analytics.")
        return

    # Single-feature pattern ranking
    st.subheader("Pattern Ranking")
    rows = []
    for p in patterns:
        rows.append({
            "Pattern":   p.get("pattern", ""),
            "Avg Ret 3s": f"{p.get('avg_score', 0):.2%}",
            "Count":     p.get("count", 0),
            "Top Emotion":   p.get("top_emotion", ""),
            "Top Structure": p.get("top_structure", ""),
            "Top Tone":      p.get("top_tone", ""),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

    if combos:
        st.subheader("Best Combinations (pattern + tone + structure)")
        combo_rows = []
        for c in combos:
            combo_rows.append({
                "Pattern":   c.get("pattern", ""),
                "Tone":      c.get("tone", ""),
                "Structure": c.get("structure", ""),
                "Avg Ret 3s": f"{c.get('avg_score', 0):.2%}",
                "Count":     c.get("count", 0),
            })
        st.dataframe(combo_rows, use_container_width=True, hide_index=True)


# ── entry point ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Pipeline Monitor",
    page_icon="🎬",
    layout="wide",
)

state = _load_state()
render(state)

time.sleep(REFRESH_SEC)
st.rerun()
