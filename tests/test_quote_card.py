"""Tests for src/quote_card.py — render_quote_card_png with mocked PIL."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ── Ensure PIL sub-modules are mocked before any import ──────────────────────
for _mod in ("PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageFont"):
    sys.modules.setdefault(_mod, MagicMock())

from src.script_generator import Scene
from src.quote_card import (
    render_quote_card_png,
    _load_font,
    _wrap,
    _quote_font_size,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _scene_with_quote(
    idx: int = 0,
    source_quote: str = "The future belongs to those who believe in the beauty of their dreams.",
    quote_attribution: str = "Eleanor Roosevelt · Statesperson",
    narration: str = "This quote captures the spirit of innovation.",
    heading: str = "Quote Scene",
    duration_sec: int = 5,
) -> Scene:
    s = Scene(
        idx=idx,
        heading=heading,
        narration=narration,
        visual_prompt="",
        duration_sec=duration_sec,
    )
    s.source_quote = source_quote
    s.quote_attribution = quote_attribution
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# _quote_font_size
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuoteFontSize:
    def test_very_short_quote_returns_52(self):
        quote = "AI is the future"  # 4 words
        assert _quote_font_size(quote) == 52

    def test_exactly_12_words_returns_52(self):
        quote = " ".join(["word"] * 12)
        assert _quote_font_size(quote) == 52

    def test_13_words_returns_46(self):
        quote = " ".join(["word"] * 13)
        assert _quote_font_size(quote) == 46

    def test_exactly_18_words_returns_46(self):
        quote = " ".join(["word"] * 18)
        assert _quote_font_size(quote) == 46

    def test_19_words_returns_40(self):
        quote = " ".join(["word"] * 19)
        assert _quote_font_size(quote) == 40

    def test_long_quote_returns_40(self):
        quote = " ".join(["word"] * 50)
        assert _quote_font_size(quote) == 40


# ═══════════════════════════════════════════════════════════════════════════════
# _wrap
# ═══════════════════════════════════════════════════════════════════════════════

class TestWrap:
    def _mock_font_with_width(self, char_width: int = 10) -> MagicMock:
        """Returns a mock font where each char is char_width pixels wide."""
        font = MagicMock()
        # getbbox returns (left, top, right, bottom); right = len(text)*char_width
        font.getbbox = lambda text: (0, 0, len(text) * char_width, 20)
        return font

    def test_single_word_no_wrap(self):
        font = self._mock_font_with_width(10)
        lines = _wrap("Hello", font, 500)
        assert lines == ["Hello"]

    def test_long_text_wraps_to_multiple_lines(self):
        font = self._mock_font_with_width(10)
        # each word is 5 chars → 50px; max_px=80 → fits 1 word per line (50<80 but 2 words=100>80)
        text = "Apple Banana Cherry Date Elderberry"
        lines = _wrap(text, font, 80)
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 80  # rough check

    def test_empty_string_returns_empty_list(self):
        font = self._mock_font_with_width(10)
        lines = _wrap("", font, 500)
        assert lines == []

    def test_all_fits_on_one_line(self):
        font = self._mock_font_with_width(5)
        lines = _wrap("Short text here", font, 1000)
        assert len(lines) == 1

    def test_last_word_appended(self):
        font = self._mock_font_with_width(10)
        lines = _wrap("A B C", font, 1000)
        # All fit → one line
        assert "C" in lines[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# _load_font
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoadFont:
    def test_returns_truetype_font_when_available(self):
        mock_font = MagicMock()
        with patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_ImageFont.truetype.return_value = mock_font
            result = _load_font("/some/font.ttf", 48)
        assert result is mock_font

    def test_falls_back_to_default_on_oserror(self):
        default_font = MagicMock()
        with patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_ImageFont.truetype.side_effect = OSError("font not found")
            mock_ImageFont.load_default.return_value = default_font
            result = _load_font("/nonexistent/font.ttf", 48)
        assert result is default_font


# ═══════════════════════════════════════════════════════════════════════════════
# render_quote_card_png
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderQuoteCardPng:
    def _make_pil_mocks(self):
        """Create a consistent set of PIL mocks for render tests."""
        mock_img = MagicMock()
        mock_draw = MagicMock()
        mock_font = MagicMock()
        # getbbox returns (left, top, right, bottom)
        mock_font.getbbox = MagicMock(return_value=(0, 0, 200, 30))
        return mock_img, mock_draw, mock_font

    def test_returns_path_if_file_already_exists(self, tmp_path):
        out = tmp_path / "quote.png"
        out.write_bytes(b"PNG")
        scene = _scene_with_quote()
        result = render_quote_card_png(scene, out)
        assert result == out

    def test_raises_if_no_source_quote(self, tmp_path):
        out = tmp_path / "quote.png"
        scene = _scene_with_quote(source_quote="")
        with pytest.raises(ValueError, match="no source_quote"):
            render_quote_card_png(scene, out)

    def test_creates_parent_directory(self, tmp_path):
        out = tmp_path / "subdir" / "quote.png"
        assert not out.parent.exists()
        scene = _scene_with_quote()

        with patch("src.quote_card.Image") as mock_Image, \
             patch("src.quote_card.ImageDraw") as mock_ImageDraw, \
             patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_img = MagicMock()
            mock_Image.new.return_value = mock_img
            mock_draw = MagicMock()
            mock_ImageDraw.Draw.return_value = mock_draw
            mock_font = MagicMock()
            mock_font.getbbox = MagicMock(return_value=(0, 0, 200, 30))
            mock_ImageFont.truetype.return_value = mock_font
            # Make _wrap work by patching out.parent.mkdir:
            mock_img.save = MagicMock()

            render_quote_card_png(scene, out)

        assert out.parent.exists()

    def test_renders_and_saves_png(self, tmp_path):
        out = tmp_path / "quote.png"
        scene = _scene_with_quote()

        with patch("src.quote_card.Image") as mock_Image, \
             patch("src.quote_card.ImageDraw") as mock_ImageDraw, \
             patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_img = MagicMock()
            mock_Image.new.return_value = mock_img
            mock_draw = MagicMock()
            mock_ImageDraw.Draw.return_value = mock_draw
            mock_font = MagicMock()
            mock_font.getbbox = MagicMock(return_value=(0, 0, 200, 30))
            mock_ImageFont.truetype.return_value = mock_font
            mock_img.save = MagicMock()

            result = render_quote_card_png(scene, out)

        assert result == out
        mock_img.save.assert_called_once()
        save_args = mock_img.save.call_args[0]
        assert "PNG" in save_args or str(out) in str(save_args)

    def test_renders_with_attribution(self, tmp_path):
        out = tmp_path / "quote_attr.png"
        scene = _scene_with_quote(
            source_quote="AI will change everything we know.",
            quote_attribution="Sam Altman · OpenAI · openai.com",
        )

        with patch("src.quote_card.Image") as mock_Image, \
             patch("src.quote_card.ImageDraw") as mock_ImageDraw, \
             patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_img = MagicMock()
            mock_Image.new.return_value = mock_img
            mock_draw = MagicMock()
            mock_ImageDraw.Draw.return_value = mock_draw
            mock_font = MagicMock()
            mock_font.getbbox = MagicMock(return_value=(0, 0, 300, 30))
            mock_ImageFont.truetype.return_value = mock_font
            mock_img.save = MagicMock()

            result = render_quote_card_png(scene, out)

        # attribution text should be drawn (draw.text called multiple times)
        assert mock_draw.text.call_count >= 2

    def test_renders_without_attribution(self, tmp_path):
        out = tmp_path / "quote_no_attr.png"
        scene = _scene_with_quote(
            source_quote="Simple quote with no attribution.",
            quote_attribution="",
        )

        with patch("src.quote_card.Image") as mock_Image, \
             patch("src.quote_card.ImageDraw") as mock_ImageDraw, \
             patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_img = MagicMock()
            mock_Image.new.return_value = mock_img
            mock_draw = MagicMock()
            mock_ImageDraw.Draw.return_value = mock_draw
            mock_font = MagicMock()
            mock_font.getbbox = MagicMock(return_value=(0, 0, 200, 30))
            mock_ImageFont.truetype.return_value = mock_font
            mock_img.save = MagicMock()

            result = render_quote_card_png(scene, out)

        mock_img.save.assert_called_once()
        assert result == out

    def test_image_created_with_correct_dimensions(self, tmp_path):
        out = tmp_path / "quote.png"
        scene = _scene_with_quote()

        with patch("src.quote_card.Image") as mock_Image, \
             patch("src.quote_card.ImageDraw") as mock_ImageDraw, \
             patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_img = MagicMock()
            mock_Image.new.return_value = mock_img
            mock_draw = MagicMock()
            mock_ImageDraw.Draw.return_value = mock_draw
            mock_font = MagicMock()
            mock_font.getbbox = MagicMock(return_value=(0, 0, 200, 30))
            mock_ImageFont.truetype.return_value = mock_font

            render_quote_card_png(scene, out)

        # Image.new called with (mode, (W, H), color)
        call_args = mock_Image.new.call_args
        assert call_args[0][1] == (1280, 720)

    def test_font_oserror_falls_back_to_default(self, tmp_path):
        out = tmp_path / "quote.png"
        scene = _scene_with_quote()

        with patch("src.quote_card.Image") as mock_Image, \
             patch("src.quote_card.ImageDraw") as mock_ImageDraw, \
             patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_img = MagicMock()
            mock_Image.new.return_value = mock_img
            mock_draw = MagicMock()
            mock_ImageDraw.Draw.return_value = mock_draw
            default_font = MagicMock()
            default_font.getbbox = MagicMock(return_value=(0, 0, 100, 20))
            mock_ImageFont.truetype.side_effect = OSError("No font file")
            mock_ImageFont.load_default.return_value = default_font
            mock_img.save = MagicMock()

            # Should not raise even without real font files
            result = render_quote_card_png(scene, out)

        assert result == out
        mock_img.save.assert_called_once()

    def test_short_quote_uses_large_font(self, tmp_path):
        out = tmp_path / "short_quote.png"
        # Very short quote (≤12 words)
        scene = _scene_with_quote(source_quote="AI is the future of humanity today")

        font_sizes = []

        def capture_truetype(path, size):
            font_sizes.append(size)
            m = MagicMock()
            m.getbbox = MagicMock(return_value=(0, 0, 200, 30))
            return m

        with patch("src.quote_card.Image") as mock_Image, \
             patch("src.quote_card.ImageDraw") as mock_ImageDraw, \
             patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_Image.new.return_value = MagicMock()
            mock_ImageDraw.Draw.return_value = MagicMock()
            mock_ImageFont.truetype.side_effect = capture_truetype

            render_quote_card_png(scene, out)

        # The quote font should be 52 for short quotes
        assert 52 in font_sizes

    def test_long_quote_uses_smaller_font(self, tmp_path):
        out = tmp_path / "long_quote.png"
        # 19+ words → should pick size 40
        scene = _scene_with_quote(
            source_quote=" ".join(["word"] * 20)
        )

        font_sizes = []

        def capture_truetype(path, size):
            font_sizes.append(size)
            m = MagicMock()
            m.getbbox = MagicMock(return_value=(0, 0, 200, 30))
            return m

        with patch("src.quote_card.Image") as mock_Image, \
             patch("src.quote_card.ImageDraw") as mock_ImageDraw, \
             patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_Image.new.return_value = MagicMock()
            mock_ImageDraw.Draw.return_value = MagicMock()
            mock_ImageFont.truetype.side_effect = capture_truetype

            render_quote_card_png(scene, out)

        assert 40 in font_sizes

    def test_separator_bar_drawn(self, tmp_path):
        out = tmp_path / "quote.png"
        scene = _scene_with_quote()

        with patch("src.quote_card.Image") as mock_Image, \
             patch("src.quote_card.ImageDraw") as mock_ImageDraw, \
             patch("src.quote_card.ImageFont") as mock_ImageFont:
            mock_img = MagicMock()
            mock_Image.new.return_value = mock_img
            mock_draw = MagicMock()
            mock_ImageDraw.Draw.return_value = mock_draw
            mock_font = MagicMock()
            mock_font.getbbox = MagicMock(return_value=(0, 0, 200, 30))
            mock_ImageFont.truetype.return_value = mock_font

            render_quote_card_png(scene, out)

        # draw.rectangle should be called for the separator bar
        mock_draw.rectangle.assert_called()
