"""Feedback analyzer for YouTube/TikTok metrics and editorial insights."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

sys_path_insert = __import__("sys").path.insert
from pathlib import Path as _Path

import sys
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from config import settings
from src.youtube_analytics import get_video_metrics, get_retention_curve
from src.tiktok_analytics import get_video_metrics as get_tiktok_video_metrics
from src.shared_types import PerformanceStats
from src.performance_tracker import get_unanalyzed, mark_analyzed


@dataclass
class VideoMetrics:
    """Video performance metrics."""
    video_id: str
    platform: str
    views: int
    avg_view_duration_sec: int
    avg_view_percentage: float
    retention_curve: list[float]  # Percentage retained at each scene break


@dataclass
class RetentionAnalysis:
    """Retention analysis results."""
    drop_point: str | None
    retention_30s: float | None
    best_segment: str | None


class FeedbackAnalyzer:
    """Analyzes video performance metrics and generates editorial feedback."""

    def __init__(self):
        self.youtube = None
        self.feedback_db = Path(settings.data_dir) / "feedback_history.json"
        self.feedback_db.parent.mkdir(parents=True, exist_ok=True)

    def analyze(
        self,
        video_id: str,
        platform: str,
        editorial_plan: dict[str, Any],
        published_at: datetime | None = None,
    ) -> dict[str, Any]:
        """Analyze video performance and generate recommendations."""
        logger.info(f"Analyzing {platform} video {video_id}")

        try:
            if platform == "youtube":
                metrics = self._get_youtube_metrics(video_id)
            elif platform == "tiktok":
                metrics = self._get_tiktok_metrics(video_id)
            else:
                logger.warning(f"Unsupported platform: {platform}")
                return {}

            if not metrics:
                logger.warning(f"Could not retrieve metrics for {video_id}")
                return {}

            retention = self._analyze_retention(metrics.retention_curve)
            hook_score = self._score_hook(metrics.retention_curve)

            result = {
                "video_id": video_id,
                "platform": platform,
                "published_at": published_at.isoformat() if published_at else None,
                "views": metrics.views,
                "hook_score": round(hook_score, 2),
                "avg_view_duration": metrics.avg_view_duration_sec,
                "avg_view_percentage": round(metrics.avg_view_percentage, 2),
                "drop_point": retention.drop_point,
                "retention_30s": round(retention.retention_30s, 2) if retention.retention_30s else None,
                "best_segment": retention.best_segment,
                "angle": editorial_plan.get("angle", "unknown"),
                "format": editorial_plan.get("format", "unknown"),
                "persona": editorial_plan.get("persona", {}),
            }

            result["recommendations"] = self._generate_recommendations(result)
            self._save_feedback(result)

            return result
        except Exception as exc:
            logger.error(f"Feedback analysis failed: {exc}")
            return {}

    def _get_youtube_metrics(self, video_id: str) -> VideoMetrics | None:
        """Fetch metrics from YouTube Analytics API."""
        try:
            # Get basic metrics
            basic_metrics = get_video_metrics(video_id)
            if not basic_metrics:
                logger.warning(f"No basic metrics for {video_id}")
                return None

            # Get retention curve
            retention_curve = get_retention_curve(video_id)
            if not retention_curve:
                logger.warning(f"No retention curve for {video_id}")
                # Fall back to estimation if no curve available
                retention_curve = self._estimate_retention_curve(basic_metrics["avg_view_percentage"])

            return VideoMetrics(
                video_id=video_id,
                platform="youtube",
                views=basic_metrics["views"],
                avg_view_duration_sec=basic_metrics["avg_view_duration"],
                avg_view_percentage=basic_metrics["avg_view_percentage"],
                retention_curve=retention_curve,
            )
        except Exception as exc:
            logger.error(f"YouTube Analytics error: {exc}")
            return None

    def _get_tiktok_metrics(self, video_id: str) -> VideoMetrics | None:
        """Fetch metrics from TikTok Content API v2 (video.list scope required)."""
        raw = get_tiktok_video_metrics(video_id, settings.tiktok_token_file)
        if not raw:
            return None

        avg_pct = raw["avg_view_percentage"]
        retention_curve = self._estimate_retention_curve(avg_pct)

        avg_duration_sec = 0
        if raw["duration_sec"] > 0:
            avg_duration_sec = round(raw["duration_sec"] * avg_pct)

        return VideoMetrics(
            video_id=video_id,
            platform="tiktok",
            views=raw["views"],
            avg_view_duration_sec=avg_duration_sec,
            avg_view_percentage=avg_pct,
            retention_curve=retention_curve,
        )

    def _estimate_retention_curve(self, avg_percentage: float) -> list[float]:
        """Estimate retention curve from average view percentage."""
        # Simplified model: assume retention starts at 1.0 and decays to avg_percentage
        # In production, use actual YouTube Analytics retention data
        return [
            1.0,  # Start (100%)
            avg_percentage + 0.15,  # 30s mark
            avg_percentage + 0.10,  # Scene 2
            avg_percentage + 0.05,  # Scene 3
            avg_percentage,  # End
        ]

    def _analyze_retention(self, curve: list[float]) -> RetentionAnalysis:
        """Analyze retention curve for drop points."""
        drop_point = None
        retention_30s = None
        best_segment = None

        if len(curve) < 2:
            return RetentionAnalysis(drop_point, retention_30s, best_segment)

        retention_30s = curve[1]

        # Find significant drop points (drop > 15%)
        for i in range(1, len(curve)):
            if curve[i] < curve[i - 1] - 0.15:
                drop_point = f"scene_{i}"
                logger.info(f"Retention drop detected at {drop_point}: {curve[i-1]:.2f} → {curve[i]:.2f}")
                break

        # Find best-retained segment
        if len(curve) > 1:
            best_idx = max(range(1, len(curve)), key=lambda i: curve[i])
            best_segment = f"scene_{best_idx}"

        return RetentionAnalysis(drop_point, retention_30s, best_segment)

    def _score_hook(self, curve: list[float]) -> float:
        """Score hook effectiveness (retention after intro)."""
        if len(curve) < 2:
            return 0.0

        # Hook score = retention at 30s mark
        return curve[1]

    def _generate_recommendations(self, metrics: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on metrics."""
        recs = []

        # Hook analysis
        if metrics["hook_score"] < 0.6:
            recs.append("Hook underperformed — try more aggressive opening or curiosity gap")
        elif metrics["hook_score"] > 0.8:
            recs.append("Hook very strong — replicate this style in future videos")

        # Retention analysis
        if metrics["drop_point"] == "scene_0" or metrics["drop_point"] == "scene_1":
            recs.append("Early drop detected — consider shorter intro or add micro-hook earlier")
        elif metrics["drop_point"] == "scene_2":
            recs.append("Mid-video drop — increase pacing or add conflict/twist around scene 2")

        # Overall watch time
        if metrics["avg_view_percentage"] < 0.5:
            recs.append("Below 50% average watch — consider shorter runtime or higher energy")
        elif metrics["avg_view_percentage"] > 0.75:
            recs.append("Strong average watch time — extend runtime or add sequel/series")

        # Views
        if metrics["views"] < 100:
            recs.append("Low view count — improve thumbnail or adjust upload time")
        elif metrics["views"] > 1000:
            recs.append("Strong performance — create similar content more frequently")

        # Best segment
        if metrics["best_segment"] and metrics["best_segment"] != "scene_0":
            recs.append(f"Best retention at {metrics['best_segment']} — analyze what worked there")

        return recs

    def _save_feedback(self, result: dict[str, Any]) -> None:
        """Save feedback to history database."""
        try:
            history = []
            if self.feedback_db.exists():
                history = json.loads(self.feedback_db.read_text())

            history.append({
                "timestamp": datetime.now().isoformat(),
                "video_id": result["video_id"],
                "platform": result["platform"],
                "hook_score": result["hook_score"],
                "avg_view_percentage": result["avg_view_percentage"],
                "angle": result["angle"],
                "format": result["format"],
                "drop_point": result["drop_point"],
            })

            self.feedback_db.write_text(json.dumps(history, indent=2))
            logger.info(f"Feedback saved for {result['video_id']}")
        except Exception as exc:
            logger.warning(f"Failed to save feedback: {exc}")

    def collect_deferred_feedback(self, min_age_hours: float = 24.0) -> int:
        """Fetch metrics for published videos that haven't been analyzed yet.

        Called at the start of each pipeline run so that yesterday's video
        has had time to accumulate real YouTube/TikTok Analytics data.
        Returns the number of videos successfully analyzed.
        """
        pending = get_unanalyzed(min_age_hours)
        if not pending:
            logger.info("No pending videos for deferred feedback collection")
            return 0

        logger.info(f"Collecting deferred feedback for {len(pending)} video(s)")
        collected = 0
        for record in pending:
            video_id = record["video_id"]
            platform = record.get("platform", "youtube")
            editorial_plan = {
                "angle": record.get("angle", "unknown"),
                "format": record.get("format", "unknown"),
            }
            result = self.analyze(video_id, platform, editorial_plan)
            if result:
                mark_analyzed(video_id)
                collected += 1
                logger.info(
                    f"Deferred feedback collected: {video_id} "
                    f"hook={result.get('hook_score')} "
                    f"retention={result.get('avg_view_percentage')}"
                )
            else:
                logger.warning(f"Deferred feedback failed for {video_id} — will retry next run")

        return collected

    def load_feedback_history(self) -> list[dict[str, Any]]:
        """Load saved feedback history from storage."""
        try:
            if self.feedback_db.exists():
                return json.loads(self.feedback_db.read_text())
        except Exception as exc:
            logger.warning(f"Failed to load feedback history: {exc}")
        return []

    def get_angle_performance(self, angle: str) -> PerformanceStats | None:
        """Get performance metrics for a specific angle."""
        try:
            history = []
            if self.feedback_db.exists():
                history = json.loads(self.feedback_db.read_text())

            records = [h for h in history if h.get("angle") == angle]
            if not records:
                return None

            avg_hook_score = sum(h["hook_score"] for h in records) / len(records)
            avg_watch_time = sum(h["avg_view_percentage"] for h in records) / len(records)
            return PerformanceStats(
                category=angle,
                sample_size=len(records),
                avg_hook_score=round(avg_hook_score, 2),
                avg_watch_time=round(avg_watch_time, 2),
                success_rate=len([h for h in records if h["hook_score"] > 0.6]) / len(records),
            )
        except Exception as exc:
            logger.warning(f"Failed to get angle performance: {exc}")
            return None

    def get_format_performance(self, fmt: str) -> PerformanceStats | None:
        """Get performance metrics for a specific format."""
        try:
            history = []
            if self.feedback_db.exists():
                history = json.loads(self.feedback_db.read_text())

            records = [h for h in history if h.get("format") == fmt]
            if not records:
                return None

            avg_hook_score = sum(h["hook_score"] for h in records) / len(records)
            avg_watch_time = sum(h["avg_view_percentage"] for h in records) / len(records)
            return PerformanceStats(
                category=fmt,
                sample_size=len(records),
                avg_hook_score=round(avg_hook_score, 2),
                avg_watch_time=round(avg_watch_time, 2),
                success_rate=len([h for h in records if h["hook_score"] > 0.6]) / len(records),
            )
        except Exception as exc:
            logger.warning(f"Failed to get format performance: {exc}")
            return None
