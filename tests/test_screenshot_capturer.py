"""Tests for src/screenshot_capturer.py — pure helper functions."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.screenshot_capturer import (
    _LIBRARY,
    CAPTURE_H,
    CAPTURE_W,
    available_keys,
    capture,
    capture_url,
    has_key,
    url_for,
)


class TestAvailableKeys:
    def test_returns_keys_for_known_tool(self):
        keys = available_keys("claude")
        assert isinstance(keys, list)
        assert len(keys) > 0

    def test_includes_homepage_for_claude(self):
        assert "homepage" in available_keys("claude")

    def test_returns_empty_for_unknown_tool(self):
        assert available_keys("unknown_tool_xyz") == []

    def test_returns_list(self):
        assert isinstance(available_keys("chatgpt"), list)

    def test_all_library_tools_have_keys(self):
        for tool in _LIBRARY:
            assert len(available_keys(tool)) > 0


class TestHasKey:
    def test_true_for_valid_key(self):
        assert has_key("claude", "homepage") is True

    def test_false_for_invalid_key(self):
        assert has_key("claude", "nonexistent_key_xyz") is False

    def test_false_for_unknown_tool(self):
        assert has_key("unknown_tool", "homepage") is False

    def test_false_for_none_key(self):
        assert has_key("claude", None) is False

    def test_false_for_empty_string_key(self):
        assert has_key("claude", "") is False

    def test_true_for_chatgpt_homepage(self):
        assert has_key("chatgpt", "homepage") is True


class TestUrlFor:
    def test_returns_url_string(self):
        url = url_for("claude", "homepage")
        assert isinstance(url, str)
        assert url.startswith("https://")

    def test_claude_homepage_url(self):
        url = url_for("claude", "homepage")
        assert "claude" in url.lower() or "anthropic" in url.lower()

    def test_raises_on_unknown_tool(self):
        with pytest.raises((KeyError, TypeError)):
            url_for("unknown_tool", "homepage")

    def test_raises_on_unknown_key(self):
        with pytest.raises(KeyError):
            url_for("claude", "nonexistent_key_xyz")


class TestCapture:
    def test_raises_for_unknown_key(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown screenshot key"):
            capture("claude", "nonexistent_key_xyz", tmp_path / "out.jpg")

    def test_raises_for_unknown_tool(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown screenshot key"):
            capture("unknown_tool", "homepage", tmp_path / "out.jpg")

    def test_returns_cached_if_exists(self, tmp_path):
        out = tmp_path / "claude_homepage.jpg"
        out.write_bytes(b"FAKE_JPEG")
        result = capture("claude", "homepage", out)
        assert result == out

    def test_returns_path_object(self, tmp_path):
        out = tmp_path / "cached.jpg"
        out.write_bytes(b"FAKE")
        result = capture("claude", "homepage", out)
        assert isinstance(result, Path)


class TestCaptureUrl:
    def test_returns_cached_if_exists(self, tmp_path):
        out = tmp_path / "cached.jpg"
        out.write_bytes(b"FAKE_JPEG")
        result = capture_url("https://example.com", out)
        assert result == out

    def test_raises_when_playwright_not_installed(self, tmp_path):
        out = tmp_path / "new.jpg"
        with patch.dict("sys.modules", {"playwright": None, "playwright.sync_api": None}):
            with pytest.raises((RuntimeError, ImportError, TypeError)):
                capture_url("https://example.com", out)

    def test_creates_parent_dir_before_capture(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "screenshot.jpg"

        mock_page = MagicMock()
        mock_context = MagicMock()
        mock_browser = MagicMock()
        mock_pw = MagicMock()
        mock_pw.chromium.launch.return_value = mock_browser
        mock_browser.new_context.return_value = mock_context
        mock_context.new_page.return_value = mock_page

        # Write a fake file so capture_url's trailing stat() doesn't fail
        from pathlib import Path as _Path

        def _fake_screenshot(**kwargs):
            _Path(kwargs["path"]).write_bytes(b"FAKE")

        mock_page.screenshot.side_effect = _fake_screenshot

        mock_sp_instance = MagicMock()
        mock_sp_instance.__enter__ = MagicMock(return_value=mock_pw)
        mock_sp_instance.__exit__ = MagicMock(return_value=False)

        mock_sync_api = MagicMock()
        mock_sync_api.sync_playwright = MagicMock(return_value=mock_sp_instance)

        with patch.dict("sys.modules", {
            "playwright": MagicMock(),
            "playwright.sync_api": mock_sync_api,
        }):
            capture_url("https://example.com", out)

        assert out.parent.exists()
        assert out.exists()


class TestConstants:
    def test_capture_dimensions(self):
        assert CAPTURE_W == 1792
        assert CAPTURE_H == 1024

    def test_library_has_multiple_tools(self):
        assert len(_LIBRARY) >= 2

    def test_each_library_entry_has_url(self):
        for tool, keys in _LIBRARY.items():
            for key, config in keys.items():
                assert "url" in config, f"Missing url for {tool}/{key}"
