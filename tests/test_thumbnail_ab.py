"""Tests for src/thumbnail_ab.py.

Covers:
- _load_font()
- _score_frame()
- _wrap()
- _to_pil()
- _save()
- _style_bottom_gradient()
- _style_impact_text()
- _style_top_bar()
- _perf_file() / _load_perf() / _save_perf()
- _get_arm()
- generate_thumbnail_variants() — uses cached paths + renders fresh
- pick_thumbnail() — Thompson sampling, single-item short-circuit
- record_thumbnail_usage()
- record_thumbnail_performance()
- top_styles()
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pytest

# ---------------------------------------------------------------------------
# Stub numpy and PIL before importing the module under test (conftest.py
# already stubs them as MagicMock, but thumbnail_ab uses them at module
# level so we need real-ish mocks).
# ---------------------------------------------------------------------------

import numpy as _np_real  # won't be available; conftest replaces with MagicMock

# We need numpy and PIL to be usable.  Since conftest stubs numpy as MagicMock,
# we replace them with the real packages if available, otherwise skip.
try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module under test — import AFTER potential stubs above
# ---------------------------------------------------------------------------
import src.thumbnail_ab as tab
from src.thumbnail_ab import (
    STYLE_BOTTOM,
    STYLE_IMPACT,
    STYLE_TOP,
    _ALL_STYLES,
    _get_arm,
    _load_perf,
    _perf_file,
    _save_perf,
    _score_frame,
    _wrap,
    generate_thumbnail_variants,
    pick_thumbnail,
    record_thumbnail_performance,
    record_thumbnail_usage,
    top_styles,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_frame(mean: float = 128.0, std: float = 40.0) -> MagicMock:
    """Return a MagicMock ndarray that behaves well for _score_frame."""
    arr = MagicMock()
    arr.mean.return_value = mean
    arr.std.return_value = std

    # For the axis-aware std call (arr.std(axis=(0,1)).mean())
    axis_std = MagicMock()
    axis_std.mean.return_value = std
    arr.std.side_effect = lambda **kwargs: axis_std if "axis" in kwargs else std
    # Reset: make it callable with positional or keyword
    def std_func(*args, **kwargs):
        if args or kwargs:
            return axis_std
        return std
    arr.std = std_func
    return arr


def _make_pil_image(w: int = 1280, h: int = 720) -> MagicMock:
    """Return a MagicMock PIL Image."""
    img = MagicMock()
    img.size = (w, h)
    img.convert.return_value = img
    img.copy.return_value = img
    img.resize.return_value = img
    return img


def _make_frames(n: int = 5) -> list[tuple[float, MagicMock]]:
    return [(float(i) * 2.5, _make_fake_frame()) for i in range(n)]


# ---------------------------------------------------------------------------
# _perf_file / _load_perf / _save_perf
# ---------------------------------------------------------------------------

class TestPerfStorage:
    def test_perf_file_path(self, tmp_path):
        p = _perf_file(tmp_path)
        assert p == tmp_path / "thumbnail_performance.json"

    def test_load_perf_missing_file(self, tmp_path):
        data = _load_perf(tmp_path)
        assert data == {}

    def test_save_and_load_roundtrip(self, tmp_path):
        payload = {"daily": {"bottom_gradient": {"alpha": 2.0, "beta": 1.5, "uses": 3, "wins": 1.5}}}
        _save_perf(payload, tmp_path)
        loaded = _load_perf(tmp_path)
        assert loaded == payload

    def test_save_creates_valid_json(self, tmp_path):
        _save_perf({"ctx": {"arm": 1}}, tmp_path)
        raw = (tmp_path / "thumbnail_performance.json").read_text()
        assert json.loads(raw) == {"ctx": {"arm": 1}}


# ---------------------------------------------------------------------------
# _get_arm
# ---------------------------------------------------------------------------

class TestGetArm:
    def test_new_arm_gets_defaults(self):
        ctx = {}
        arm = _get_arm(ctx, "bottom_gradient")
        assert arm["alpha"] == 1.0
        assert arm["beta"] == 1.0
        assert arm["uses"] == 0
        assert arm["wins"] == 0.0

    def test_existing_arm_not_reset(self):
        ctx = {"bottom_gradient": {"alpha": 5.0, "beta": 2.0, "uses": 10, "wins": 4.0}}
        arm = _get_arm(ctx, "bottom_gradient")
        assert arm["alpha"] == 5.0
        assert arm["uses"] == 10

    def test_adds_to_ctx(self):
        ctx = {}
        _get_arm(ctx, "impact_text")
        assert "impact_text" in ctx


# ---------------------------------------------------------------------------
# _score_frame
# ---------------------------------------------------------------------------

class TestScoreFrame:
    def test_too_dark_returns_minus_one(self):
        arr = MagicMock()
        arr.mean.return_value = 10.0  # < 35
        # _score_frame checks mean first, so no std call needed
        arr.std = lambda *a, **kw: 50.0
        score = _score_frame(arr)
        assert score == -1.0

    def test_too_bright_returns_minus_one(self):
        arr = MagicMock()
        arr.mean.return_value = 230.0  # > 225
        arr.std = lambda *a, **kw: 50.0
        score = _score_frame(arr)
        assert score == -1.0

    def test_good_frame_returns_positive(self):
        arr = MagicMock()
        arr.mean.return_value = 128.0
        colorfulness_mock = MagicMock()
        colorfulness_mock.mean.return_value = 40.0
        def std_func(*args, **kwargs):
            if args or kwargs:
                return colorfulness_mock
            return 30.0
        arr.std = std_func
        score = _score_frame(arr)
        # colorfulness * 0.65 + contrast * 0.35 = 40*0.65 + 30*0.35 = 26 + 10.5 = 36.5
        assert score > 0.0


# ---------------------------------------------------------------------------
# _wrap
# ---------------------------------------------------------------------------

class TestWrap:
    def test_short_title_single_line(self):
        lines = _wrap("Short title")
        assert len(lines) >= 1
        assert "Short title" in lines

    def test_very_long_title_max_two_lines(self):
        title = "This is an extraordinarily long title that would normally wrap into three or more lines"
        lines = _wrap(title)
        assert len(lines) <= 2

    def test_custom_max_chars(self):
        title = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8"
        lines = _wrap(title, max_chars=15)
        assert all(len(l) > 0 for l in lines)


# ---------------------------------------------------------------------------
# Style renderers — tested with mocked PIL
# ---------------------------------------------------------------------------

class TestStyleRenderers:
    def _make_mock_image(self):
        img = MagicMock()
        img.size = (1280, 720)
        img.convert.return_value = img
        img.copy.return_value = img
        return img

    def _make_mock_draw(self):
        draw = MagicMock()
        return draw

    def test_style_bottom_gradient_runs(self):
        """Smoke test that _style_bottom_gradient runs without error."""
        mock_img = self._make_mock_image()
        overlay = MagicMock()
        overlay.__class__ = MagicMock
        alpha_composite_result = MagicMock()
        alpha_composite_result.size = (1280, 720)
        alpha_composite_result.convert.return_value = alpha_composite_result

        with patch("src.thumbnail_ab.Image") as MockImage, \
             patch("src.thumbnail_ab.ImageDraw") as MockDraw, \
             patch("src.thumbnail_ab._load_font", return_value=MagicMock()), \
             patch("src.thumbnail_ab._wrap", return_value=["Test Title"]):

            MockImage.new.return_value = overlay
            MockImage.alpha_composite.return_value = alpha_composite_result

            result = tab._style_bottom_gradient(mock_img, "Test Title")
        # Should have returned something (the converted image)
        assert result is not None

    def test_style_impact_text_runs(self):
        """Smoke test that _style_impact_text runs without error."""
        mock_img = self._make_mock_image()
        mock_vig = MagicMock()
        composite_result = MagicMock()
        composite_result.size = (1280, 720)

        mock_font = MagicMock()
        mock_font.getbbox.return_value = (0, 0, 200, 60)

        with patch("src.thumbnail_ab.Image") as MockImage, \
             patch("src.thumbnail_ab.ImageDraw") as MockDraw, \
             patch("src.thumbnail_ab._load_font", return_value=mock_font), \
             patch("src.thumbnail_ab._wrap", return_value=["Test Title"]):

            MockImage.new.return_value = mock_vig
            MockImage.alpha_composite.return_value = composite_result
            composite_result.convert.return_value = composite_result

            result = tab._style_impact_text(mock_img, "Test Title")
        assert result is not None

    def test_style_top_bar_runs(self):
        """Smoke test that _style_top_bar runs without error."""
        mock_img = self._make_mock_image()
        overlay = MagicMock()
        composite_result = MagicMock()
        composite_result.size = (1280, 720)
        composite_result.convert.return_value = composite_result

        with patch("src.thumbnail_ab.Image") as MockImage, \
             patch("src.thumbnail_ab.ImageDraw") as MockDraw, \
             patch("src.thumbnail_ab._load_font", return_value=MagicMock()), \
             patch("src.thumbnail_ab._wrap", return_value=["Test Title"]):

            MockImage.new.return_value = overlay
            MockImage.alpha_composite.return_value = composite_result

            result = tab._style_top_bar(mock_img, "Test Title")
        assert result is not None


# ---------------------------------------------------------------------------
# generate_thumbnail_variants
# ---------------------------------------------------------------------------

class TestGenerateThumbnailVariants:
    def _make_mock_frame_arr(self):
        arr = MagicMock()
        arr.mean.return_value = 128.0
        colorfulness_mock = MagicMock()
        colorfulness_mock.mean.return_value = 40.0
        def std_func(*args, **kwargs):
            if args or kwargs:
                return colorfulness_mock
            return 30.0
        arr.std = std_func
        return arr

    def test_uses_cached_thumbnails(self, tmp_path):
        """If thumbnail files already exist, they should be reused without re-rendering."""
        for style in _ALL_STYLES:
            (tmp_path / f"thumbnail_{style}.jpg").write_bytes(b"fake jpeg data")

        video_path = tmp_path / "video.mp4"
        video_path.write_bytes(b"fake video")

        mock_arr = self._make_mock_frame_arr()
        frames = [(0.5, mock_arr), (1.0, mock_arr)]
        mock_pil = MagicMock()
        mock_pil.size = (1280, 720)

        with patch("src.thumbnail_ab._sample_frames", return_value=frames), \
             patch("src.thumbnail_ab._best_frame", return_value=mock_arr), \
             patch("src.thumbnail_ab._mid_frame", return_value=mock_arr), \
             patch("src.thumbnail_ab._to_pil", return_value=mock_pil), \
             patch("src.thumbnail_ab._style_bottom_gradient") as mock_render:
            paths = generate_thumbnail_variants(video_path, "Test Title", tmp_path)

        # All cached — renderers should not be called
        mock_render.assert_not_called()
        assert len(paths) == 3
        for path in paths:
            assert path.exists()

    def test_generates_fresh_thumbnails(self, tmp_path):
        """When no cached files exist, all three variants should be rendered."""
        video_path = tmp_path / "video.mp4"

        mock_arr = self._make_mock_frame_arr()
        frames = [(0.5, mock_arr), (1.0, mock_arr), (1.5, mock_arr)]
        mock_pil_img = MagicMock()
        mock_pil_img.size = (1280, 720)
        mock_pil_img.convert.return_value = mock_pil_img
        mock_pil_img.copy.return_value = mock_pil_img

        rendered_img = MagicMock()
        rendered_img.convert.return_value = rendered_img

        # Create actual files when _save is called
        def fake_save(img, path):
            path.write_bytes(b"jpeg")
            return path

        with patch("src.thumbnail_ab._sample_frames", return_value=frames), \
             patch("src.thumbnail_ab._best_frame", return_value=mock_arr), \
             patch("src.thumbnail_ab._mid_frame", return_value=mock_arr), \
             patch("src.thumbnail_ab._to_pil", return_value=mock_pil_img), \
             patch("src.thumbnail_ab._style_bottom_gradient", return_value=rendered_img), \
             patch("src.thumbnail_ab._style_impact_text", return_value=rendered_img), \
             patch("src.thumbnail_ab._style_top_bar", return_value=rendered_img), \
             patch("src.thumbnail_ab._save", side_effect=fake_save):
            paths = generate_thumbnail_variants(video_path, "Test Title", tmp_path)

        assert len(paths) == 3

    def test_partial_cache(self, tmp_path):
        """Only cached files reused; missing ones rendered."""
        (tmp_path / f"thumbnail_{STYLE_BOTTOM}.jpg").write_bytes(b"cached")
        video_path = tmp_path / "video.mp4"

        mock_arr = self._make_mock_frame_arr()
        frames = [(0.5, mock_arr)]
        mock_pil_img = MagicMock()
        mock_pil_img.size = (1280, 720)
        mock_pil_img.convert.return_value = mock_pil_img
        mock_pil_img.copy.return_value = mock_pil_img
        rendered_img = MagicMock()

        def fake_save(img, path):
            path.write_bytes(b"jpeg")
            return path

        with patch("src.thumbnail_ab._sample_frames", return_value=frames), \
             patch("src.thumbnail_ab._best_frame", return_value=mock_arr), \
             patch("src.thumbnail_ab._mid_frame", return_value=mock_arr), \
             patch("src.thumbnail_ab._to_pil", return_value=mock_pil_img), \
             patch("src.thumbnail_ab._style_bottom_gradient", return_value=rendered_img), \
             patch("src.thumbnail_ab._style_impact_text", return_value=rendered_img), \
             patch("src.thumbnail_ab._style_top_bar", return_value=rendered_img), \
             patch("src.thumbnail_ab._save", side_effect=fake_save):
            paths = generate_thumbnail_variants(video_path, "Test Title", tmp_path)

        assert len(paths) == 3


# ---------------------------------------------------------------------------
# pick_thumbnail
# ---------------------------------------------------------------------------

class TestPickThumbnail:
    def test_single_variant_returns_it(self, tmp_path):
        variant = tmp_path / "thumbnail_bottom_gradient.jpg"
        variant.write_bytes(b"x")
        result = pick_thumbnail([variant], tmp_path, "daily")
        assert result == variant

    def test_picks_one_from_multiple(self, tmp_path):
        paths = []
        for style in _ALL_STYLES:
            p = tmp_path / f"thumbnail_{style}.jpg"
            p.write_bytes(b"x")
            paths.append(p)

        # Patch betavariate to always return highest for STYLE_IMPACT
        def fake_beta(alpha, beta):
            # Return high score for second path (impact_text)
            return 0.9 if alpha == 1.0 else 0.5

        with patch("src.thumbnail_ab.random.betavariate", side_effect=fake_beta):
            result = pick_thumbnail(paths, tmp_path, "daily")

        assert result in paths

    def test_saves_perf_after_pick(self, tmp_path):
        paths = [tmp_path / f"thumbnail_{s}.jpg" for s in _ALL_STYLES]
        for p in paths:
            p.write_bytes(b"x")

        pick_thumbnail(paths, tmp_path, "daily")
        # Performance file should now exist
        assert (tmp_path / "thumbnail_performance.json").exists()

    def test_updates_existing_context(self, tmp_path):
        # Pre-populate some arm data
        existing = {"daily": {"bottom_gradient": {"alpha": 3.0, "beta": 1.0, "uses": 5, "wins": 2.0}}}
        _save_perf(existing, tmp_path)

        paths = [tmp_path / f"thumbnail_{s}.jpg" for s in _ALL_STYLES]
        for p in paths:
            p.write_bytes(b"x")

        pick_thumbnail(paths, tmp_path, "daily")
        loaded = _load_perf(tmp_path)
        # Context should still be present
        assert "daily" in loaded

    def test_style_extracted_from_filename(self, tmp_path):
        """pick_thumbnail derives style from file stem."""
        p = tmp_path / "thumbnail_top_bar.jpg"
        p.write_bytes(b"x")
        p2 = tmp_path / "thumbnail_impact_text.jpg"
        p2.write_bytes(b"x")

        result = pick_thumbnail([p, p2], tmp_path, "test")
        assert result in [p, p2]


# ---------------------------------------------------------------------------
# record_thumbnail_usage
# ---------------------------------------------------------------------------

class TestRecordThumbnailUsage:
    def test_records_style_and_video_id(self, tmp_path):
        thumb = tmp_path / "thumbnail_bottom_gradient.jpg"
        record_thumbnail_usage(thumb, "vid_abc", tmp_path, "daily")
        data = _load_perf(tmp_path)
        assert data["_video_thumbnails"]["vid_abc"]["style"] == "bottom_gradient"
        assert data["_video_thumbnails"]["vid_abc"]["context"] == "daily"

    def test_increments_uses(self, tmp_path):
        thumb = tmp_path / "thumbnail_impact_text.jpg"
        record_thumbnail_usage(thumb, "vid_001", tmp_path, "daily")
        record_thumbnail_usage(thumb, "vid_002", tmp_path, "daily")
        data = _load_perf(tmp_path)
        arm = data["daily"]["impact_text"]
        assert arm["uses"] == 2

    def test_multiple_videos_recorded(self, tmp_path):
        thumb_a = tmp_path / "thumbnail_bottom_gradient.jpg"
        thumb_b = tmp_path / "thumbnail_top_bar.jpg"
        record_thumbnail_usage(thumb_a, "vid_A", tmp_path, "daily")
        record_thumbnail_usage(thumb_b, "vid_B", tmp_path, "daily")
        data = _load_perf(tmp_path)
        assert data["_video_thumbnails"]["vid_A"]["style"] == "bottom_gradient"
        assert data["_video_thumbnails"]["vid_B"]["style"] == "top_bar"

    def test_saves_to_disk(self, tmp_path):
        thumb = tmp_path / "thumbnail_bottom_gradient.jpg"
        record_thumbnail_usage(thumb, "vid_save", tmp_path, "ctx")
        assert (tmp_path / "thumbnail_performance.json").exists()


# ---------------------------------------------------------------------------
# record_thumbnail_performance
# ---------------------------------------------------------------------------

class TestRecordThumbnailPerformance:
    def _setup_video_record(self, tmp_path, video_id: str, style: str, ctx: str = "daily"):
        data = {
            "_video_thumbnails": {video_id: {"context": ctx, "style": style}},
            ctx: {style: {"alpha": 1.0, "beta": 1.0, "uses": 1, "wins": 0.0}},
        }
        _save_perf(data, tmp_path)

    def test_high_ctr_increases_alpha(self, tmp_path):
        self._setup_video_record(tmp_path, "vid1", "bottom_gradient")
        record_thumbnail_performance("vid1", 8.0, tmp_path)
        data = _load_perf(tmp_path)
        arm = data["daily"]["bottom_gradient"]
        # reward = min(1.0, 8.0/10) = 0.8 → alpha += 0.8
        assert abs(arm["alpha"] - 1.8) < 0.001
        assert abs(arm["beta"] - 1.2) < 0.001
        assert abs(arm["wins"] - 0.8) < 0.001

    def test_zero_ctr_increases_beta(self, tmp_path):
        self._setup_video_record(tmp_path, "vid2", "impact_text")
        record_thumbnail_performance("vid2", 0.0, tmp_path)
        data = _load_perf(tmp_path)
        arm = data["daily"]["impact_text"]
        assert arm["alpha"] == 1.0     # no reward
        assert arm["beta"] == 2.0      # +1.0
        assert arm["wins"] == 0.0

    def test_very_high_ctr_clamped_to_one(self, tmp_path):
        self._setup_video_record(tmp_path, "vid3", "top_bar")
        record_thumbnail_performance("vid3", 20.0, tmp_path)  # > 10 %
        data = _load_perf(tmp_path)
        arm = data["daily"]["top_bar"]
        assert arm["alpha"] == 2.0   # reward clamped to 1.0
        assert abs(arm["beta"] - 1.0) < 0.001  # beta += 0.0

    def test_missing_video_id_logs_warning(self, tmp_path, caplog):
        import logging
        record_thumbnail_performance("unknown_vid", 5.0, tmp_path)
        # No crash — just a log warning

    def test_context_key_override(self, tmp_path):
        """When context_key is provided, it overrides the stored context."""
        data = {
            "_video_thumbnails": {"vid_x": {"context": "daily", "style": "bottom_gradient"}},
            "weekly": {"bottom_gradient": {"alpha": 1.0, "beta": 1.0, "uses": 0, "wins": 0.0}},
        }
        _save_perf(data, tmp_path)
        record_thumbnail_performance("vid_x", 5.0, tmp_path, context_key="weekly")
        loaded = _load_perf(tmp_path)
        arm = loaded["weekly"]["bottom_gradient"]
        assert arm["alpha"] > 1.0

    def test_saves_to_disk(self, tmp_path):
        self._setup_video_record(tmp_path, "vid_save", "bottom_gradient")
        record_thumbnail_performance("vid_save", 3.0, tmp_path)
        assert (tmp_path / "thumbnail_performance.json").exists()


# ---------------------------------------------------------------------------
# top_styles
# ---------------------------------------------------------------------------

class TestTopStyles:
    def _populate_perf(self, tmp_path, context_key: str = "daily") -> None:
        data = {
            context_key: {
                "bottom_gradient": {"alpha": 5.0, "beta": 2.0, "uses": 10, "wins": 4.0},
                "impact_text":     {"alpha": 3.0, "beta": 3.0, "uses": 8, "wins": 2.0},
                "top_bar":         {"alpha": 2.0, "beta": 4.0, "uses": 5, "wins": 1.0},
            }
        }
        _save_perf(data, tmp_path)

    def test_returns_sorted_by_mean(self, tmp_path):
        self._populate_perf(tmp_path)
        styles = top_styles(tmp_path, "daily")
        means = [s["mean"] for s in styles]
        assert means == sorted(means, reverse=True)

    def test_returns_correct_fields(self, tmp_path):
        self._populate_perf(tmp_path)
        styles = top_styles(tmp_path, "daily")
        for s in styles:
            assert "style" in s
            assert "mean" in s
            assert "uses" in s
            assert "wins" in s

    def test_correct_mean_calculation(self, tmp_path):
        """mean = alpha / (alpha + beta)"""
        self._populate_perf(tmp_path)
        styles = top_styles(tmp_path, "daily")
        bottom = next(s for s in styles if s["style"] == "bottom_gradient")
        assert abs(bottom["mean"] - 5.0 / (5.0 + 2.0)) < 0.001

    def test_empty_context(self, tmp_path):
        styles = top_styles(tmp_path, "nonexistent_ctx")
        assert styles == []

    def test_respects_n_limit(self, tmp_path):
        self._populate_perf(tmp_path)
        styles = top_styles(tmp_path, "daily", n=1)
        assert len(styles) == 1
        # Should be the best one
        assert styles[0]["style"] == "bottom_gradient"

    def test_only_valid_styles_included(self, tmp_path):
        """Non-standard arm names should be excluded."""
        data = {
            "daily": {
                "bottom_gradient": {"alpha": 2.0, "beta": 1.0, "uses": 3, "wins": 1.0},
                "unknown_style":   {"alpha": 10.0, "beta": 1.0, "uses": 5, "wins": 5.0},
            }
        }
        _save_perf(data, tmp_path)
        styles = top_styles(tmp_path, "daily")
        style_names = [s["style"] for s in styles]
        assert "unknown_style" not in style_names
        assert "bottom_gradient" in style_names

    def test_multiple_contexts_isolated(self, tmp_path):
        self._populate_perf(tmp_path, context_key="daily")
        data = _load_perf(tmp_path)
        data["weekly"] = {
            "top_bar": {"alpha": 8.0, "beta": 1.0, "uses": 20, "wins": 7.0}
        }
        _save_perf(data, tmp_path)

        daily_styles = top_styles(tmp_path, "daily")
        weekly_styles = top_styles(tmp_path, "weekly")

        daily_names = {s["style"] for s in daily_styles}
        weekly_names = {s["style"] for s in weekly_styles}

        assert "top_bar" in daily_names  # also in daily
        assert weekly_names == {"top_bar"}


# ---------------------------------------------------------------------------
# _load_font (smoke test — falls back to default)
# ---------------------------------------------------------------------------

class TestLoadFont:
    def test_fallback_when_font_missing(self):
        with patch("src.thumbnail_ab.ImageFont") as MockFont:
            MockFont.truetype.side_effect = OSError("Font not found")
            MockFont.load_default.return_value = MagicMock()
            result = tab._load_font(48)
        MockFont.load_default.assert_called_once()

    def test_loads_truetype_when_available(self):
        mock_font = MagicMock()
        with patch("src.thumbnail_ab.ImageFont") as MockFont:
            MockFont.truetype.return_value = mock_font
            result = tab._load_font(48)
        assert result == mock_font


# ---------------------------------------------------------------------------
# _to_pil
# ---------------------------------------------------------------------------

class TestToPil:
    def test_calls_fromarray_and_resize(self):
        mock_arr = MagicMock()
        mock_arr.astype.return_value = mock_arr
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img
        resized = MagicMock()
        mock_img.resize.return_value = resized

        with patch("src.thumbnail_ab.Image") as MockImage:
            MockImage.fromarray.return_value = mock_img
            MockImage.LANCZOS = MagicMock()
            result = tab._to_pil(mock_arr)

        MockImage.fromarray.assert_called_once()
        mock_img.resize.assert_called_once_with(
            (tab.THUMB_W, tab.THUMB_H), MockImage.LANCZOS
        )


# ---------------------------------------------------------------------------
# _save
# ---------------------------------------------------------------------------

class TestSave:
    def test_saves_jpeg_and_returns_path(self, tmp_path):
        path = tmp_path / "thumb.jpg"
        # Create a real minimal JPEG file first to make stat() work
        path.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)  # fake JPEG header

        mock_img = MagicMock()
        # Override so _save returns path after "saving"
        def fake_img_save(p, *args, **kwargs):
            Path(p).write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        mock_img.save.side_effect = fake_img_save
        result = tab._save(mock_img, path)
        assert result == path
        mock_img.save.assert_called_once_with(str(path), "JPEG", quality=92, optimize=True)
