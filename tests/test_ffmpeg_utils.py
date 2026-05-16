"""Tests for ffmpeg_utils._esc() — the drawtext filter text escaper."""
import sys
import unittest.mock as mock

# Stub numpy before importing the module (heavy dep not needed for these tests)
import sys
sys.modules.setdefault("numpy", mock.MagicMock())

from src.ffmpeg_utils import _esc


class TestEsc:
    def test_clean_text_unchanged(self):
        assert _esc("Hello World") == "Hello World"

    def test_backslash_escaped_first(self):
        # Backslash must be doubled; must happen before other replacements
        # so that escape sequences introduced by later steps aren't re-escaped.
        assert _esc("a\\b") == "a\\\\b"

    def test_single_quote_escaped(self):
        # FFmpeg mirrors POSIX: ' → '\''
        assert _esc("it's") == r"it'\''s"

    def test_colon_escaped(self):
        assert _esc("foo:bar") == r"foo\:bar"

    def test_square_brackets_escaped(self):
        assert _esc("a[b]c") == r"a\[b\]c"

    def test_percent_doubled_for_drawtext(self):
        # % is the drawtext printf-expansion prefix (%{pts}, %{expr:…})
        assert _esc("100%") == "100%%"
        # { and } are not filter-graph special chars — only % is doubled,
        # so %%{pts} is rendered literally as %{pts} (expansion broken)
        assert _esc("%{pts}") == "%%{pts}"

    def test_percent_after_backslash_not_double_escaped(self):
        # backslash before %, then % doubled — the backslash must not be re-escaped
        result = _esc("50\\%")
        assert result == "50\\\\%%"

    def test_newline_stripped(self):
        assert _esc("line1\nline2") == "line1line2"

    def test_carriage_return_stripped(self):
        assert _esc("foo\rbar") == "foobar"

    def test_null_byte_stripped(self):
        assert _esc("foo\x00bar") == "foobar"

    def test_tab_stripped(self):
        assert _esc("a\tb") == "ab"

    def test_other_control_chars_stripped(self):
        # Bell, form-feed, etc.
        assert _esc("a\x07\x0cb") == "ab"

    def test_empty_string(self):
        assert _esc("") == ""

    def test_real_world_news_headline(self):
        # Typical RSS headline with no special chars
        result = _esc("OpenAI releases GPT-5 with record benchmarks")
        assert result == "OpenAI releases GPT-5 with record benchmarks"

    def test_headline_with_percent(self):
        # Common in financial/statistics headlines
        result = _esc("AI adoption up 35% year-over-year")
        assert result == "AI adoption up 35%% year-over-year"

    def test_headline_with_colon(self):
        result = _esc("Breaking: New model surpasses GPT-4")
        assert r"\:" in result

    def test_headline_with_single_quote(self):
        result = _esc("What's next for AI regulation?")
        assert r"'\''" in result

    def test_combined_special_chars(self):
        # Worst-case headline with multiple special chars
        result = _esc("AI's impact: [50%] increase\nin efficiency")
        # Single quote escaped
        assert r"'\''" in result
        # Colon escaped
        assert r"\:" in result
        # Percent doubled
        assert "%%" in result
        # Brackets escaped
        assert r"\[" in result
        assert r"\]" in result
        # Newline stripped — no literal newline in output
        assert "\n" not in result
