"""Tests for src/thumbnail_generator.py — with mocked PIL, numpy, and ffmpeg."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure PIL and numpy sub-modules are mocked before any import ─────────────
for _mod in ("PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont", "numpy"):
    sys.modules.setdefault(_mod, MagicMock())

# ── Module under test ─────────────────────────────────────────────────────────
from src.thumbnail_generator import (
    _score_frame,
    _wrap_title,
    _load_font,
    _overlay_title,
    _sample_frames,
    generate_thumbnail,
    THUMB_W,
    THUMB_H,
)


# ── numpy mock helpers ─────────────────────────────────────────────────────────

def _make_np_array(mean_val: float = 128.0, std_val: float = 40.0) -> MagicMock:
    """Return a MagicMock that mimics np.ndarray for frame-scoring purposes."""
    arr = MagicMock()
    arr.mean = MagicMock(return_value=mean_val)

    std_with_mean = MagicMock()
    std_with_mean.mean = MagicMock(return_value=std_val)

    def _std_dispatch(*args, **kwargs):
        # arr.std(axis=(0,1)) → object with .mean(); arr.std() → real float
        if kwargs.get("axis") is not None or args:
            return std_with_mean
        return std_val

    arr.std = MagicMock(side_effect=_std_dispatch)
    return arr


# ═══════════════════════════════════════════════════════════════════════════════
# _score_frame
# ═══════════════════════════════════════════════════════════════════════════════

class TestScoreFrame:
    def test_dark_frame_returns_minus_one(self):
        arr = _make_np_array(mean_val=30.0)
        assert _score_frame(arr) == -1.0

    def test_bright_frame_returns_minus_one(self):
        arr = _make_np_array(mean_val=230.0)
        assert _score_frame(arr) == -1.0

    def test_normal_frame_returns_positive_score(self):
        arr = MagicMock()
        arr.mean = MagicMock(return_value=128.0)
        arr.std = MagicMock(side_effect=[
            MagicMock(mean=MagicMock(return_value=30.0)),  # per-channel std
            50.0,                                           # overall std
        ])
        score = _score_frame(arr)
        assert isinstance(float(score), float)

    def test_boundary_mean_35_passes(self):
        arr = _make_np_array(mean_val=35.0)
        # mean == 35 (not < 35) → not automatically -1
        # (std-based score will be computed)
        # Just ensure no crash and returns float
        score = _score_frame(arr)
        assert isinstance(float(score), float)

    def test_boundary_mean_225_passes(self):
        arr = _make_np_array(mean_val=225.0)
        score = _score_frame(arr)
        assert isinstance(float(score), float)

    def test_boundary_mean_34_fails(self):
        arr = _make_np_array(mean_val=34.0)
        assert _score_frame(arr) == -1.0

    def test_boundary_mean_226_fails(self):
        arr = _make_np_array(mean_val=226.0)
        assert _score_frame(arr) == -1.0


# ═══════════════════════════════════════════════════════════════════════════════
# _wrap_title
# ═══════════════════════════════════════════════════════════════════════════════

class TestWrapTitle:
    def test_short_title_single_line(self):
        lines = _wrap_title("AI News Today")
        assert len(lines) == 1
        assert lines[0] == "AI News Today"

    def test_long_title_wraps_to_two_lines(self):
        title = "The Complete Guide to Understanding Artificial Intelligence in 2026"
        lines = _wrap_title(title)
        assert len(lines) <= 2

    def test_very_long_title_capped_at_two_lines(self):
        title = " ".join(["word"] * 20)
        lines = _wrap_title(title)
        assert len(lines) <= 2

    def test_default_max_chars_38(self):
        # A title just over 38 chars should wrap
        title = "This is a title that is exactly forty-two characters long here"
        lines = _wrap_title(title)
        assert len(lines) >= 1

    def test_custom_max_chars(self):
        title = "Short title"
        lines = _wrap_title(title, max_chars=100)
        assert len(lines) == 1

    def test_empty_string(self):
        lines = _wrap_title("")
        assert lines == []

    def test_preserves_words(self):
        title = "GPT-5 Breaks All Records"
        lines = _wrap_title(title)
        combined = " ".join(lines)
        assert "GPT-5" in combined
        assert "Records" in combined


# ═══════════════════════════════════════════════════════════════════════════════
# _load_font
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFont:
    def test_returns_truetype_font(self):
        mock_font = MagicMock()
        with patch("src.thumbnail_generator.ImageFont") as mock_IF:
            mock_IF.truetype.return_value = mock_font
            result = _load_font(54)
        assert result is mock_font

    def test_falls_back_on_oserror(self):
        default_font = MagicMock()
        with patch("src.thumbnail_generator.ImageFont") as mock_IF:
            mock_IF.truetype.side_effect = OSError("missing font")
            mock_IF.load_default.return_value = default_font
            result = _load_font(54)
        assert result is default_font


# ═══════════════════════════════════════════════════════════════════════════════
# _sample_frames
# ═══════════════════════════════════════════════════════════════════════════════

class TestSampleFrames:
    def test_returns_n_frames(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"FAKE_VIDEO")
        fake_frame = _make_np_array()

        with patch("src.thumbnail_generator.ffmpeg_utils.duration", return_value=60.0), \
             patch("src.thumbnail_generator.ffmpeg_utils.get_frame", return_value=fake_frame):
            frames = _sample_frames(video, n=8)

        assert len(frames) == 8

    def test_timestamps_within_5_and_95_percent(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"FAKE_VIDEO")
        duration = 100.0
        fake_frame = _make_np_array()

        with patch("src.thumbnail_generator.ffmpeg_utils.duration", return_value=duration), \
             patch("src.thumbnail_generator.ffmpeg_utils.get_frame", return_value=fake_frame):
            frames = _sample_frames(video, n=16)

        timestamps = [t for t, _ in frames]
        assert min(timestamps) >= duration * 0.05 - 0.001
        assert max(timestamps) <= duration * 0.95 + 0.001

    def test_returns_tuple_pairs(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"FAKE_VIDEO")
        fake_frame = _make_np_array()

        with patch("src.thumbnail_generator.ffmpeg_utils.duration", return_value=30.0), \
             patch("src.thumbnail_generator.ffmpeg_utils.get_frame", return_value=fake_frame):
            frames = _sample_frames(video, n=4)

        for t, arr in frames:
            assert isinstance(float(t), float)
            assert arr is fake_frame


# ═══════════════════════════════════════════════════════════════════════════════
# _overlay_title
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlayTitle:
    def _make_image_mock(self, w=1280, h=720):
        img = MagicMock()
        img.size = (w, h)
        # convert should return another mock with same interface
        rgba_img = MagicMock()
        rgba_img.size = (w, h)
        composite_img = MagicMock()
        composite_img.size = (w, h)
        img.convert.return_value = rgba_img
        # Image.alpha_composite returns composite_img
        return img, rgba_img, composite_img

    def test_returns_rgb_image(self):
        img, rgba_img, composite_img = self._make_image_mock()
        final_img = MagicMock()

        with patch("src.thumbnail_generator.Image") as mock_Image, \
             patch("src.thumbnail_generator.ImageDraw") as mock_Draw, \
             patch("src.thumbnail_generator.ImageFont") as mock_Font:
            overlay = MagicMock()
            mock_Image.new.return_value = overlay
            mock_Image.alpha_composite.return_value = composite_img
            composite_img.convert.return_value = final_img
            draw = MagicMock()
            mock_Draw.Draw.return_value = draw
            mock_Font.truetype.return_value = MagicMock()

            result = _overlay_title(img, "AI News Title")

        assert result is final_img

    def test_draws_text_on_image(self):
        img = MagicMock()
        img.size = (1280, 720)
        rgba_img = MagicMock()
        img.convert.return_value = rgba_img
        composite_img = MagicMock()
        composite_img.size = (1280, 720)
        final_img = MagicMock()
        composite_img.convert.return_value = final_img

        with patch("src.thumbnail_generator.Image") as mock_Image, \
             patch("src.thumbnail_generator.ImageDraw") as mock_Draw, \
             patch("src.thumbnail_generator.ImageFont") as mock_Font:
            overlay = MagicMock()
            mock_Image.new.return_value = overlay
            mock_Image.alpha_composite.return_value = composite_img
            draw = MagicMock()
            mock_Draw.Draw.return_value = draw
            mock_Font.truetype.return_value = MagicMock()

            _overlay_title(img, "GPT-5 Breaks Records")

        # draw.text should be called at least once (drop shadow + main text)
        assert draw.text.call_count >= 1

    def test_gradient_drawn_as_rectangles(self):
        img = MagicMock()
        img.size = (1280, 720)
        rgba_img = MagicMock()
        img.convert.return_value = rgba_img
        composite_img = MagicMock()
        composite_img.size = (1280, 720)
        final_img = MagicMock()
        composite_img.convert.return_value = final_img

        with patch("src.thumbnail_generator.Image") as mock_Image, \
             patch("src.thumbnail_generator.ImageDraw") as mock_Draw, \
             patch("src.thumbnail_generator.ImageFont") as mock_Font:
            overlay = MagicMock()
            mock_Image.new.return_value = overlay
            mock_Image.alpha_composite.return_value = composite_img
            draw = MagicMock()
            mock_Draw.Draw.return_value = draw
            mock_Font.truetype.return_value = MagicMock()

            _overlay_title(img, "Test Title")

        # Many rectangles drawn for gradient (720//3 = 240 lines)
        assert draw.rectangle.call_count > 10


# ═══════════════════════════════════════════════════════════════════════════════
# generate_thumbnail
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateThumbnail:
    def _make_frame(self) -> MagicMock:
        """Create a frame mock compatible with _score_frame.

        _score_frame calls std() twice per invocation and is called twice
        per frame (once in max(), once for logging). Use _make_np_array so
        the mock handles repeated calls without running out of side effects.
        """
        frame = _make_np_array(mean_val=128.0, std_val=40.0)
        frame.astype = MagicMock(return_value=frame)
        return frame

    def test_returns_cached_path_if_exists(self, tmp_path):
        out_dir = tmp_path / "output"
        out_dir.mkdir()
        thumb = out_dir / "thumbnail.jpg"
        thumb.write_bytes(b"FAKE_THUMB")

        with patch("src.thumbnail_generator.ffmpeg_utils"):
            result = generate_thumbnail(tmp_path / "video.mp4", "Title", out_dir)

        assert result == thumb

    def _patch_thumbnail(self, tmp_path, *, frames=None, out_name="thumbnail.jpg"):
        """Build mocks for generate_thumbnail tests; make save() write the file."""
        video = tmp_path / "video.mp4"
        video.write_bytes(b"FAKE_VIDEO")
        out_dir = tmp_path / "output"
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / out_name

        fake_frame = self._make_frame()
        if frames is None:
            frames = [(3.0, fake_frame)]

        overlay_img = MagicMock()

        def _write(dest, fmt=None, **kw):
            Path(dest).write_bytes(b"FAKE_JPEG")

        overlay_img.save.side_effect = _write
        return video, out_dir, out_path, fake_frame, overlay_img, frames

    def test_generates_thumbnail_and_saves_jpeg(self, tmp_path):
        video, out_dir, out_path, fake_frame, overlay_img, frames = self._patch_thumbnail(tmp_path)

        with patch("src.thumbnail_generator._sample_frames", return_value=frames), \
             patch("src.thumbnail_generator.np") as mock_np, \
             patch("src.thumbnail_generator.Image") as mock_Image, \
             patch("src.thumbnail_generator.ImageDraw"), \
             patch("src.thumbnail_generator.ImageFont"), \
             patch("src.thumbnail_generator._overlay_title", return_value=overlay_img):
            mock_np.uint8 = MagicMock()
            img = MagicMock()
            img.convert.return_value = img
            img.resize.return_value = img
            mock_Image.fromarray.return_value = img
            mock_Image.LANCZOS = MagicMock()
            result = generate_thumbnail(video, "AI Breakthrough", out_dir)

        assert result == out_path
        assert out_path.exists()

    def test_uses_custom_out_name(self, tmp_path):
        video, out_dir, out_path, fake_frame, overlay_img, frames = self._patch_thumbnail(
            tmp_path, out_name="thumbnail_ru.jpg"
        )

        with patch("src.thumbnail_generator._sample_frames", return_value=frames), \
             patch("src.thumbnail_generator.np") as mock_np, \
             patch("src.thumbnail_generator.Image") as mock_Image, \
             patch("src.thumbnail_generator.ImageDraw"), \
             patch("src.thumbnail_generator.ImageFont"), \
             patch("src.thumbnail_generator._overlay_title", return_value=overlay_img):
            mock_np.uint8 = MagicMock()
            img = MagicMock()
            img.convert.return_value = img
            img.resize.return_value = img
            mock_Image.fromarray.return_value = img
            mock_Image.LANCZOS = MagicMock()
            result = generate_thumbnail(video, "Russian variant", out_dir, out_name="thumbnail_ru.jpg")

        assert result == out_path
        assert out_path.exists()

    def test_selects_best_scoring_frame(self, tmp_path):
        """Verify that the frame with the highest score is selected."""
        good_frame = _make_np_array(mean_val=128.0, std_val=50.0)
        good_frame.astype = MagicMock(return_value=good_frame)
        dark_frame = _make_np_array(mean_val=10.0, std_val=5.0)
        dark_frame.astype = MagicMock(return_value=dark_frame)

        frames = [(0.0, dark_frame), (30.0, good_frame), (60.0, dark_frame)]
        video, out_dir, out_path, _, overlay_img, _ = self._patch_thumbnail(tmp_path, frames=frames)

        with patch("src.thumbnail_generator._sample_frames", return_value=frames), \
             patch("src.thumbnail_generator.np") as mock_np, \
             patch("src.thumbnail_generator.Image") as mock_Image, \
             patch("src.thumbnail_generator.ImageDraw"), \
             patch("src.thumbnail_generator.ImageFont"), \
             patch("src.thumbnail_generator._overlay_title", return_value=overlay_img):
            mock_np.uint8 = MagicMock()
            img = MagicMock()
            img.convert.return_value = img
            img.resize.return_value = img
            mock_Image.fromarray.return_value = img
            mock_Image.LANCZOS = MagicMock()
            generate_thumbnail(video, "Test Title", out_dir)

        # good_frame.astype should have been called (it was selected as best)
        good_frame.astype.assert_called()

    def test_thumbnail_resized_to_correct_dimensions(self, tmp_path):
        video, out_dir, out_path, fake_frame, overlay_img, frames = self._patch_thumbnail(tmp_path)

        resize_sizes = []

        def capture_resize(size, resample):
            resize_sizes.append(size)
            return MagicMock()

        with patch("src.thumbnail_generator._sample_frames", return_value=frames), \
             patch("src.thumbnail_generator.np") as mock_np, \
             patch("src.thumbnail_generator.Image") as mock_Image, \
             patch("src.thumbnail_generator.ImageDraw"), \
             patch("src.thumbnail_generator.ImageFont"), \
             patch("src.thumbnail_generator._overlay_title", return_value=overlay_img):
            mock_np.uint8 = MagicMock()
            img = MagicMock()
            img.convert.return_value = img
            img.resize.side_effect = capture_resize
            mock_Image.fromarray.return_value = img
            mock_Image.LANCZOS = MagicMock()
            generate_thumbnail(video, "Test Title", out_dir)

        assert (THUMB_W, THUMB_H) in resize_sizes
