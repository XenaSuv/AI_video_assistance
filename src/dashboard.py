"""Real-time AI pipeline monitor.

Run with:
    streamlit run src/dashboard.py

Auto-refreshes every REFRESH_SEC seconds by calling st.rerun().
Reads data/live_state.json written by LiveState during pipeline execution.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time
from pathlib import Path

import streamlit as st

# ── config ────────────────────────────────────────────────────────────────────

REFRESH_SEC = 10
_ROOT      = Path(__file__).resolve().parent.parent
STATE_FILE = _ROOT / "data" / "live_state.json"
DATA_DIR   = _ROOT / "data"
OUTPUT_DIR = _ROOT / "output"

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

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


# ── analytics data loaders ────────────────────────────────────────────────────

def _load_performance_records() -> list[dict]:
    path = DATA_DIR / "performance.json"
    try:
        return json.loads(path.read_text()) if path.exists() else []
    except Exception:
        return []


def _load_weekly_costs() -> dict[str, float]:
    """Aggregate cost_report.json files from the last 7 days across all pipelines."""
    today  = dt.date.today()
    cutoff = today - dt.timedelta(days=6)
    daily: dict[str, float] = {}
    if not OUTPUT_DIR.exists():
        return daily
    for report_path in OUTPUT_DIR.glob("**/cost_report.json"):
        for part in report_path.parts:
            if _DATE_RE.match(part):
                try:
                    d = dt.date.fromisoformat(part)
                    if cutoff <= d <= today:
                        data = json.loads(report_path.read_text())
                        daily[part] = daily.get(part, 0.0) + data.get("total_usd", 0.0)
                except Exception:
                    pass
                break
    return dict(sorted(daily.items()))


def _load_scene_retention() -> dict[str, float]:
    path = DATA_DIR / "scene_performance.json"
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}


# ── analytics render functions ────────────────────────────────────────────────

def render_hook_trend(records: list[dict]) -> None:
    st.subheader("Hook Score Trend")
    dated = [
        (r["timestamp"][:10], r["avg_view_percentage"])
        for r in records
        if r.get("avg_view_percentage") is not None and r.get("timestamp")
    ]
    if not dated:
        st.caption("No retention data yet — populates after videos accumulate views.")
        return
    by_date: dict[str, list[float]] = {}
    for d, v in dated:
        by_date.setdefault(d, []).append(v)
    rows = [
        {"Date": d, "Avg retention %": round(sum(vs) / len(vs) * 100, 1)}
        for d, vs in sorted(by_date.items())
    ]
    st.line_chart(rows, x="Date", y="Avg retention %")
    st.caption(
        f"Based on {len(dated)} video(s). "
        "Retention % = avg_view_percentage from YouTube Analytics."
    )


def render_ctr_trend(records: list[dict]) -> None:
    st.subheader("CTR Trend")
    dated = [
        (r["timestamp"][:10], r["ctr"])
        for r in records
        if r.get("ctr") is not None and r.get("timestamp")
    ]
    if not dated:
        st.caption("No CTR data yet.")
        return
    by_date: dict[str, list[float]] = {}
    for d, v in dated:
        by_date.setdefault(d, []).append(v)
    rows = [
        {"Date": d, "Avg CTR %": round(sum(vs) / len(vs) * 100, 2)}
        for d, vs in sorted(by_date.items())
    ]
    st.line_chart(rows, x="Date", y="Avg CTR %")


def render_weekly_cost(costs: dict[str, float]) -> None:
    st.subheader("API Spend (last 7 days)")
    if not costs:
        st.caption("No cost reports found in output/ yet.")
        return
    rows = [{"Date": d, "Cost ($)": round(v, 4)} for d, v in costs.items()]
    st.bar_chart(rows, x="Date", y="Cost ($)")
    total = sum(costs.values())
    st.caption(f"Total this week: **${total:.4f}**")


def render_scene_retention(data: dict[str, float]) -> None:
    st.subheader("Retention by Scene Type")
    if not data:
        st.caption("No scene_performance.json data yet.")
        return
    rows = [
        {"Scene type": k.replace("_", " ").title(), "Avg score": round(v, 3)}
        for k, v in sorted(data.items(), key=lambda x: -x[1])
    ]
    st.bar_chart(rows, x="Scene type", y="Avg score")
    st.caption("Source: data/scene_performance.json — scene-type retention averages.")


def render_top_videos(records: list[dict]) -> None:
    st.subheader("Top Videos by Retention")
    scored = [
        r for r in records
        if r.get("avg_view_percentage") is not None and r.get("views", 0) > 0
    ]
    scored.sort(key=lambda r: r["avg_view_percentage"], reverse=True)
    if not scored:
        st.caption("No video performance records with views yet.")
        return
    rows = [
        {
            "Title":       r["title"][:55],
            "Date":        (r.get("timestamp") or "")[:10],
            "Type":        r.get("content_type", "—"),
            "Views":       r.get("views", 0),
            "CTR %":       f"{r['ctr']*100:.1f}" if r.get("ctr") else "—",
            "Retention %": f"{r['avg_view_percentage']*100:.1f}",
        }
        for r in scored[:15]
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_analytics() -> None:
    st.header("📊 Analytics")

    records        = _load_performance_records()
    weekly_costs   = _load_weekly_costs()
    scene_ret      = _load_scene_retention()

    # ── summary KPIs ──────────────────────────────────────────────────────────
    with_ret = [r for r in records if r.get("avg_view_percentage") is not None]
    with_ctr = [r for r in records if r.get("ctr") is not None]
    avg_ret  = sum(r["avg_view_percentage"] for r in with_ret) / len(with_ret) if with_ret else None
    avg_ctr  = sum(r["ctr"]               for r in with_ctr) / len(with_ctr) if with_ctr else None
    week_usd = sum(weekly_costs.values())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total videos tracked", len(records))
    k2.metric("Avg retention",  f"{avg_ret*100:.1f}%" if avg_ret is not None else "—")
    k3.metric("Avg CTR",        f"{avg_ctr*100:.2f}%" if avg_ctr is not None else "—")
    k4.metric("Spend this week", f"${week_usd:.4f}")

    st.divider()

    # ── trend charts ──────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)
    with col_left:
        render_hook_trend(records)
    with col_right:
        render_ctr_trend(records)

    st.divider()

    # ── cost + scene retention ────────────────────────────────────────────────
    col_cost, col_scene = st.columns(2)
    with col_cost:
        render_weekly_cost(weekly_costs)
    with col_scene:
        render_scene_retention(scene_ret)

    st.divider()

    render_top_videos(records)


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

    # ── emotion engine section ────────────────────────────────────────────────
    st.divider()
    render_emotion_engine()

    # ── narrative conflict section ────────────────────────────────────────────
    st.divider()
    render_narrative_conflicts()

    # ── narrative identity section ────────────────────────────────────────────
    st.divider()
    render_narrative_identity()

    # ── shorts beat template section ─────────────────────────────────────────
    st.divider()
    render_shorts_beat_template()

    # ── emotional arc section ─────────────────────────────────────────────────
    st.divider()
    render_emotional_arcs()

    # ── hook patterns section ─────────────────────────────────────────────────
    st.divider()
    render_hook_patterns()


# ── shorts beat template ─────────────────────────────────────────────────────

def render_shorts_beat_template() -> None:
    st.header("Shorts Beat Template")
    st.caption("Golden rule: every 2–3 seconds → change (frame / emotion / text / pace)")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.shorts_beat_engine import (
            _BEAT_SSML_PREFIX,
            _BEAT_SSML_SUFFIX,
            BEAT_AVATAR_MODE,
            BEAT_DURATIONS,
            BEAT_EMOTIONS,
            BEAT_ROLES,
            BEAT_VISUAL_TYPE,
        )
    except Exception:
        st.caption("Shorts beat engine not available.")
        return

    rows = []
    offset = 0.0
    for role in BEAT_ROLES:
        dur     = BEAT_DURATIONS[role]
        emotion, intensity = BEAT_EMOTIONS[role]
        prefix  = _BEAT_SSML_PREFIX.get(role, "—") or "—"
        suffix  = _BEAT_SSML_SUFFIX.get(role, "—") or "—"
        rows.append({
            "Beat":         role.upper(),
            "Time":         f"{offset:.0f}–{offset + dur:.0f}s",
            "Duration":     f"{dur:.0f}s",
            "Emotion":      emotion,
            "Intensity":    f"{intensity:.0%}",
            "Avatar/Broll": BEAT_AVATAR_MODE[role],
            "Visual":       BEAT_VISUAL_TYPE[role],
            "SSML prefix":  prefix,
        })
        offset += dur

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.caption(
        "Emotion flow: **shock → doubt → neutral → surprise → curiosity**  "
        "— never the same emotion twice in a row."
    )


# ── emotional arc ────────────────────────────────────────────────────────────

def render_emotional_arcs() -> None:
    st.header("Emotional Arc Engine")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.emotional_arc_engine import (
            EMOTIONAL_ARCS,
            EmotionalArcEngine,
        )
        engine = EmotionalArcEngine()
        history = engine.load_stats(n=10)
        distribution = engine.arc_type_distribution()
    except Exception:
        st.caption("No arc data yet — builds after the first video is published.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Arc type distribution")
        dist_rows = [{"Arc": arc, "Used": count} for arc, count in distribution.items()]
        st.dataframe(dist_rows, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Arc definitions")
        for arc_name, steps in EMOTIONAL_ARCS.items():
            step_labels = " → ".join(f"{emo}({int(intensity*100)}%)" for emo, intensity in steps)
            st.write(f"**{arc_name}**: {step_labels}")

    if history:
        st.subheader("Recent arc runs")
        rows = [
            {
                "Date":      s.recorded_at[:10],
                "Arc":       s.arc_type,
                "Scenes":    s.scene_count,
                "Avg intens":f"{s.avg_intensity:.2f}",
                "Emotions":  " → ".join(s.emotion_sequence[:6]),
            }
            for s in reversed(history)
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No arc history yet — records after each video upload.")


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

# ── emotion engine ───────────────────────────────────────────────────────────

def render_emotion_engine() -> None:
    st.header("Emotion Engine")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.emotion_engine import (
            EMOTION_MAP,
            SHORT_EMOTION_FLOW,
            EmotionEngine,
        )
        engine = EmotionEngine()
        history = engine.load_stats(n=10)
    except Exception:
        st.caption("No emotion data yet — builds after the first video is published.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Intent → Emotion map")
        rows = [
            {"Intent": intent, "Emotion": emo, "Base intensity": f"{intensity:.0%}"}
            for intent, (emo, intensity) in EMOTION_MAP.items()
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Shorts emotion arc")
        for i, emo in enumerate(SHORT_EMOTION_FLOW):
            st.write(f"{i+1}. **{emo}**")

    if history:
        st.subheader("Emotion curve — recent videos")
        # Build intensity curves for the last runs
        chart_data: dict[str, list[float]] = {}
        for stat in history[-5:]:
            label = stat.title[:30]
            # Reconstruct intensity from sequence length (no per-scene intensities stored)
            chart_data[label] = [0.0] * stat.scene_count

        # Show summary table instead (per-scene intensity isn't stored in stats)
        stat_rows = [
            {
                "Title":     s.title[:50],
                "Scenes":    s.scene_count,
                "Dominant":  s.dominant_emotion,
                "Avg intens":f"{s.avg_intensity:.2f}",
                "Arc":       " → ".join(s.emotion_sequence[:6]),
                "Date":      s.recorded_at[:10],
            }
            for s in reversed(history)
        ]
        st.dataframe(stat_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No emotion history yet — records after each video upload.")


# ── narrative conflicts ───────────────────────────────────────────────────────

def render_narrative_conflicts() -> None:
    st.header("Narrative Conflicts")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.narrative_conflict_engine import (
            CONFLICT_TYPES,
            NarrativeConflictEngine,
        )
        engine = NarrativeConflictEngine()
        stats  = engine.stats()
        recent = engine._memory.load_recent(n=10)
    except Exception:
        st.caption("No conflict data yet — builds after the first few videos.")
        return

    if not stats["total"]:
        st.caption("No conflict data yet — builds after the first few videos.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Total conflicts",   stats["total"])
    col2.metric("Dominant type",     stats["dominant_type"])
    col3.metric("Avg intensity",     f"{stats['avg_intensity']:.2f}")

    st.subheader("Conflict type distribution")
    type_rows = [
        {"Type": t, "Count": stats["type_counts"].get(t, 0)}
        for t in CONFLICT_TYPES
    ]
    st.dataframe(type_rows, use_container_width=True, hide_index=True)

    if recent:
        st.subheader("Recent conflicts")
        rows = [
            {
                "Date":      e.recorded_at[:10],
                "Type":      e.conflict_type,
                "Intensity": f"{e.intensity:.2f}",
                "Topic":     e.topic[:50],
                "Line":      e.line[:80],
            }
            for e in reversed(recent)
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


# ── narrative identity ────────────────────────────────────────────────────────

def render_narrative_identity() -> None:
    st.header("Narrative Identity")

    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from src.narrative_identity_engine import (
            BELIEFS,
            CALLBACKS,
            THEMES,
            NarrativeIdentityEngine,
        )
        engine = NarrativeIdentityEngine()
        arc_data = engine.arc_status()
        recent   = engine._memory.load_recent(n=10)
    except Exception:
        st.caption("Narrative identity data not available yet.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Core Beliefs")
        for belief, stance in BELIEFS.items():
            st.write(f"**{belief.replace('_', ' ')}** → {stance.replace('_', ' ')}")

    with col2:
        st.subheader("Recurring Themes")
        for theme in THEMES:
            st.write(f"• {theme.replace('_', ' ')}")

    st.subheader("Narrative Arcs")
    arc_rows = []
    for _arc_key, info in arc_data.items():
        arc_rows.append({
            "Arc":      info["label"],
            "Status":   info["status"],
            "Episodes": info["episode_count"],
            "Last":     (info["last_episode"] or "")[:10],
            "Summary":  info["summary"],
        })
    st.dataframe(arc_rows, use_container_width=True, hide_index=True)

    if recent:
        st.subheader("Recent Memory (last 10 entries)")
        mem_rows = [
            {
                "Date":  e.recorded_at[:10],
                "Type":  e.entry_type,
                "Topic": e.topic[:60],
                "Theme": e.theme,
                "Arc":   e.arc_key,
            }
            for e in reversed(recent)
        ]
        st.dataframe(mem_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("No narrative memory yet — builds after the first few videos are published.")

    with st.expander("Signature Callbacks"):
        for cb in CALLBACKS:
            st.write(f"• {cb}")


tab_monitor, tab_analytics = st.tabs(["🎬 Live Monitor", "📊 Analytics"])

with tab_monitor:
    state = _load_state()
    render(state)

with tab_analytics:
    render_analytics()

time.sleep(REFRESH_SEC)
st.rerun()
