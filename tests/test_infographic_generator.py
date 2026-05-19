"""Tests for src/infographic_generator.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.infographic_generator import (
    ANIM_SHARE,
    FPS,
    _anim_t,
    _bar_chart_frame,
    _comparison_frame,
    _ease_out,
    _parse_numeric,
    _render_frames,
    _stat_card_frame,
    _timeline_frame,
    generate_infographic_clip,
)

# ── _ease_out ─────────────────────────────────────────────────────────────────

class TestEaseOut:
    def test_zero_maps_to_zero(self):
        assert _ease_out(0.0) == 0.0

    def test_one_maps_to_one(self):
        assert _ease_out(1.0) == 1.0

    def test_clamped_below_zero(self):
        assert _ease_out(-1.0) == 0.0

    def test_clamped_above_one(self):
        assert _ease_out(2.0) == 1.0

    def test_midpoint_cubic_formula(self):
        # 1 - (1 - 0.5)^3 = 1 - 0.125 = 0.875
        assert abs(_ease_out(0.5) - 0.875) < 1e-9

    def test_monotonically_increasing(self):
        values = [_ease_out(i / 10) for i in range(11)]
        assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))

    def test_concave_at_start_convex_at_end(self):
        # Cubic ease-out accelerates slowly: value past midpoint > 0.5
        assert _ease_out(0.5) > 0.5


# ── _anim_t ──────────────────────────────────────────────────────────────────

class TestAnimT:
    def test_zero_frame_time_gives_zero(self):
        assert _anim_t(0.0) == 0.0

    def test_at_anim_share_saturates_to_one(self):
        assert _anim_t(ANIM_SHARE) == 1.0

    def test_past_anim_share_stays_at_one(self):
        assert _anim_t(1.0) == 1.0

    def test_half_of_anim_share_maps_through_ease_out(self):
        # frame_t/ANIM_SHARE = 0.5 → ease_out(0.5) = 0.875
        assert abs(_anim_t(ANIM_SHARE / 2) - 0.875) < 1e-9

    def test_returns_float_in_unit_interval(self):
        for frame_t in [0.0, 0.1, 0.2, 0.5, 0.9, 1.0]:
            v = _anim_t(frame_t)
            assert 0.0 <= v <= 1.0


# ── _parse_numeric ────────────────────────────────────────────────────────────

class TestParseNumeric:
    def test_plain_integer(self):
        num, suffix, prefix = _parse_numeric("42")
        assert num == 42.0
        assert suffix == ""
        assert prefix == ""

    def test_dollar_prefix_and_B_suffix(self):
        num, suffix, prefix = _parse_numeric("$4.2B")
        assert abs(num - 4.2) < 1e-9
        assert suffix == "B"
        assert prefix == "$"

    def test_percent_suffix(self):
        num, suffix, prefix = _parse_numeric("100%")
        assert num == 100.0
        assert suffix == "%"
        assert prefix == ""

    def test_non_numeric_returns_zero_and_original(self):
        num, suffix, prefix = _parse_numeric("N/A")
        assert num == 0.0
        assert suffix == "N/A"
        assert prefix == ""

    def test_float_value(self):
        num, suffix, prefix = _parse_numeric("3.14")
        assert abs(num - 3.14) < 1e-9
        assert suffix == ""
        assert prefix == ""

    def test_letter_prefix_no_suffix(self):
        num, suffix, prefix = _parse_numeric("x5")
        assert num == 5.0
        assert suffix == ""
        assert prefix == "x"

    def test_strips_surrounding_whitespace(self):
        num, suffix, prefix = _parse_numeric("  99  ")
        assert num == 99.0


# ── frame renderer smoke tests ────────────────────────────────────────────────

class TestFrameRenderers:
    """Verify renderers run without error under PIL MagicMock.

    PIL is stubbed in conftest.py — tests verify control flow, not pixels.
    """

    def _bar_data(self):
        return {
            "type": "bar_chart",
            "title": "Model Performance",
            "unit": "%",
            "items": [
                {"label": "GPT-4",  "value": 85},
                {"label": "Claude", "value": 90},
            ],
        }

    def _timeline_data(self):
        return {
            "type": "timeline",
            "title": "AI Milestones",
            "events": [
                {"date": "2022", "label": "ChatGPT"},
                {"date": "2023", "label": "GPT-4"},
            ],
        }

    def _stat_data(self):
        return {
            "type": "stat_card",
            "value": "$4.2B",
            "label": "Funding raised",
            "context": "In Q1 2024 alone",
        }

    def _comparison_data(self):
        return {
            "type": "comparison",
            "title": "GPT-4 vs Claude",
            "left":  {"label": "GPT-4",  "value": 80},
            "right": {"label": "Claude", "value": 90},
            "unit": "%",
        }

    # bar_chart
    def test_bar_chart_at_t_zero(self):
        _bar_chart_frame(self._bar_data(), 0.0)

    def test_bar_chart_at_t_half(self):
        _bar_chart_frame(self._bar_data(), 0.5)

    def test_bar_chart_at_t_one(self):
        _bar_chart_frame(self._bar_data(), 1.0)

    def test_bar_chart_empty_items(self):
        _bar_chart_frame({"type": "bar_chart", "items": []}, 0.5)

    # timeline
    def test_timeline_at_t_zero(self):
        _timeline_frame(self._timeline_data(), 0.0)

    def test_timeline_at_t_one(self):
        _timeline_frame(self._timeline_data(), 1.0)

    def test_timeline_empty_events(self):
        _timeline_frame({"type": "timeline", "events": []}, 0.5)

    # stat_card
    def test_stat_card_numeric_value(self):
        _stat_card_frame(self._stat_data(), 0.5)

    def test_stat_card_integer_value(self):
        _stat_card_frame({"type": "stat_card", "value": "1000", "label": "Users"}, 0.5)

    def test_stat_card_non_numeric_value(self):
        _stat_card_frame({"type": "stat_card", "value": "N/A", "label": "Status"}, 0.5)

    def test_stat_card_context_visible_past_half(self):
        # anim > 0.5 when t=1.0: context branch executes
        _stat_card_frame(self._stat_data(), 1.0)

    # comparison
    def test_comparison_at_t_zero(self):
        _comparison_frame(self._comparison_data(), 0.0)

    def test_comparison_at_t_one(self):
        _comparison_frame(self._comparison_data(), 1.0)

    def test_comparison_right_wins_when_higher(self):
        data = {**self._comparison_data(), "left": {"label": "A", "value": 20}, "right": {"label": "B", "value": 80}}
        _comparison_frame(data, 1.0)

    def test_comparison_equal_values(self):
        data = {**self._comparison_data(), "left": {"label": "A", "value": 50}, "right": {"label": "B", "value": 50}}
        _comparison_frame(data, 0.5)


# ── _render_frames ────────────────────────────────────────────────────────────

class TestRenderFrames:
    def test_frame_count_matches_fps_times_seconds(self):
        data = {"type": "stat_card", "value": "42", "label": "Score"}
        frames = _render_frames(data, 1.0)
        assert len(frames) == max(1, int(1.0 * FPS))

    def test_minimum_one_frame_for_tiny_duration(self):
        data = {"type": "stat_card", "value": "0", "label": "X"}
        frames = _render_frames(data, 0.01)
        assert len(frames) == 1

    def test_returns_list(self):
        data = {"type": "stat_card", "value": "7", "label": "Y"}
        frames = _render_frames(data, 0.5)
        assert isinstance(frames, list)

    def test_unknown_type_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown infographic type"):
            _render_frames({"type": "pie_chart"}, 1.0)

    def test_all_known_types_produce_frames(self):
        datasets = [
            {"type": "bar_chart",  "items": [{"label": "A", "value": 1}]},
            {"type": "timeline",   "events": [{"date": "2024", "label": "E"}]},
            {"type": "stat_card",  "value": "5", "label": "L"},
            {"type": "comparison", "left": {"label": "A", "value": 1}, "right": {"label": "B", "value": 2}},
        ]
        for data in datasets:
            frames = _render_frames(data, 0.5)
            assert len(frames) >= 1, f"No frames for type={data['type']!r}"


# ── generate_infographic_clip ─────────────────────────────────────────────────

class TestGenerateInfographicClip:
    def _data(self):
        return {"type": "stat_card", "value": "99", "label": "Score"}

    def test_cache_hit_skips_render(self, tmp_path):
        out = tmp_path / "infographic.mp4"
        out.touch()
        with patch("src.infographic_generator._render_frames") as mock_render:
            result = generate_infographic_clip(self._data(), out)
        mock_render.assert_not_called()
        assert result == out

    def test_cache_hit_returns_same_path(self, tmp_path):
        out = tmp_path / "cached.mp4"
        out.touch()
        assert generate_infographic_clip(self._data(), out) is out

    def test_renders_and_writes_when_missing(self, tmp_path):
        out = tmp_path / "new.mp4"
        with (
            patch("src.infographic_generator._render_frames", return_value=[]) as mock_render,
            patch("src.infographic_generator._frames_to_mp4", return_value=out) as mock_write,
        ):
            result = generate_infographic_clip(self._data(), out)
        mock_render.assert_called_once_with(self._data(), 8.0)
        mock_write.assert_called_once()
        assert result == out

    def test_custom_duration_forwarded_to_render(self, tmp_path):
        out = tmp_path / "custom.mp4"
        with (
            patch("src.infographic_generator._render_frames", return_value=[]) as mock_render,
            patch("src.infographic_generator._frames_to_mp4", return_value=out),
        ):
            generate_infographic_clip(self._data(), out, duration_sec=5.0)
        mock_render.assert_called_once_with(self._data(), 5.0)

    def test_unknown_type_propagates_error(self, tmp_path):
        out = tmp_path / "bad.mp4"
        with pytest.raises(ValueError, match="Unknown infographic type"):
            generate_infographic_clip({"type": "unknown_type"}, out)
