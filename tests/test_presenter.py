"""Tests for src/presenter.py — D-ID Talks API integration."""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.presenter import _b64_uri, generate_presenter_clip

# ── helpers ───────────────────────────────────────────────────────────────────

def _make_resp(json_data: dict, status_code: int = 200):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_data
    r.iter_content = MagicMock(return_value=[b"video_bytes"])
    return r


@pytest.fixture
def avatar(tmp_path) -> Path:
    p = tmp_path / "avatar.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    return p


@pytest.fixture
def hook_audio(tmp_path) -> Path:
    p = tmp_path / "hook.mp3"
    p.write_bytes(b"ID3" + b"\x00" * 32)
    return p


@pytest.fixture
def out_path(tmp_path) -> Path:
    return tmp_path / "presenter_hook.mp4"


# ── _b64_uri ──────────────────────────────────────────────────────────────────

class TestB64Uri:
    def test_png_mime_type(self, tmp_path):
        p = tmp_path / "img.png"
        p.write_bytes(b"data")
        uri = _b64_uri(p)
        assert uri.startswith("data:image/png;base64,")

    def test_jpg_mime_type(self, tmp_path):
        p = tmp_path / "img.jpg"
        p.write_bytes(b"data")
        assert _b64_uri(p).startswith("data:image/jpeg;base64,")

    def test_mp3_mime_type(self, tmp_path):
        p = tmp_path / "audio.mp3"
        p.write_bytes(b"data")
        assert _b64_uri(p).startswith("data:audio/mpeg;base64,")

    def test_content_is_base64_encoded(self, tmp_path):
        p = tmp_path / "file.png"
        p.write_bytes(b"hello")
        uri = _b64_uri(p)
        encoded_part = uri.split(",", 1)[1]
        assert base64.b64decode(encoded_part) == b"hello"

    def test_unknown_extension_fallback(self, tmp_path):
        p = tmp_path / "file.bin"
        p.write_bytes(b"data")
        assert "application/octet-stream" in _b64_uri(p)


# ── generate_presenter_clip ───────────────────────────────────────────────────

class TestGeneratePresenterClip:
    def test_returns_none_when_no_api_key(self, hook_audio, out_path, avatar):
        result = generate_presenter_clip(
            hook_audio, out_path, api_key="", avatar_path=avatar
        )
        assert result is None

    def test_returns_none_when_avatar_missing(self, hook_audio, out_path, tmp_path):
        result = generate_presenter_clip(
            hook_audio, out_path,
            api_key="testkey",
            avatar_path=tmp_path / "nonexistent.png",
        )
        assert result is None

    def test_returns_cached_if_exists(self, hook_audio, out_path, avatar):
        out_path.write_bytes(b"cached")
        result = generate_presenter_clip(
            hook_audio, out_path, api_key="testkey", avatar_path=avatar
        )
        assert result == out_path

    def test_happy_path_returns_out_path(self, hook_audio, out_path, avatar):
        create_resp = _make_resp({"id": "talk_abc123"})
        done_resp   = _make_resp({"status": "done", "result_url": "https://cdn.d-id.com/result.mp4"})
        dl_resp     = _make_resp({})
        dl_resp.iter_content = MagicMock(return_value=[b"mp4data"])

        with patch("src.presenter.http_post", return_value=create_resp), \
             patch("src.presenter.http_get", side_effect=[done_resp, dl_resp]):
            result = generate_presenter_clip(
                hook_audio, out_path, api_key="testkey", avatar_path=avatar
            )

        assert result == out_path
        assert out_path.read_bytes() == b"mp4data"

    def test_create_talk_sends_correct_payload(self, hook_audio, out_path, avatar):
        create_resp = _make_resp({"id": "talk_xyz"})
        done_resp   = _make_resp({"status": "done", "result_url": "https://cdn.d-id.com/r.mp4"})
        dl_resp     = _make_resp({})
        dl_resp.iter_content = MagicMock(return_value=[b"data"])

        with patch("src.presenter.http_post", return_value=create_resp) as mock_post, \
             patch("src.presenter.http_get", side_effect=[done_resp, dl_resp]):
            generate_presenter_clip(
                hook_audio, out_path, api_key="mykey", avatar_path=avatar
            )

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        assert payload["config"]["stitch"] is True
        assert payload["config"]["result_format"] == "mp4"
        assert payload["script"]["type"] == "audio"
        assert kwargs["headers"]["Authorization"] == "Basic mykey"

    def test_polls_until_done(self, hook_audio, out_path, avatar):
        create_resp  = _make_resp({"id": "talk_poll"})
        pending_resp = _make_resp({"status": "created"})
        done_resp    = _make_resp({"status": "done", "result_url": "https://cdn.d-id.com/r.mp4"})
        dl_resp      = _make_resp({})
        dl_resp.iter_content = MagicMock(return_value=[b"data"])

        with patch("src.presenter.http_post", return_value=create_resp), \
             patch("src.presenter.http_get", side_effect=[pending_resp, pending_resp, done_resp, dl_resp]), \
             patch("time.sleep"):
            result = generate_presenter_clip(
                hook_audio, out_path, api_key="testkey", avatar_path=avatar
            )

        assert result == out_path

    def test_returns_none_on_did_error_status(self, hook_audio, out_path, avatar):
        create_resp = _make_resp({"id": "talk_err"})
        error_resp  = _make_resp({"status": "error", "error": {"description": "render failed"}})

        with patch("src.presenter.http_post", return_value=create_resp), \
             patch("src.presenter.http_get", return_value=error_resp), \
             patch("time.sleep"):
            result = generate_presenter_clip(
                hook_audio, out_path, api_key="testkey", avatar_path=avatar
            )

        assert result is None
        assert not out_path.exists()

    def test_returns_none_on_timeout(self, hook_audio, out_path, avatar):
        create_resp = _make_resp({"id": "talk_slow"})
        pending_resp = _make_resp({"status": "created"})

        # Simulate timeout by making monotonic exceed _POLL_TIMEOUT immediately
        with patch("src.presenter.http_post", return_value=create_resp), \
             patch("src.presenter.http_get", return_value=pending_resp), \
             patch("time.sleep"), \
             patch("time.monotonic", side_effect=[0.0, 0.0, 999.0]):
            result = generate_presenter_clip(
                hook_audio, out_path, api_key="testkey", avatar_path=avatar
            )

        assert result is None

    def test_returns_none_on_network_error(self, hook_audio, out_path, avatar):
        import requests
        with patch("src.presenter.http_post",
                   side_effect=requests.exceptions.ConnectionError("network down")):
            result = generate_presenter_clip(
                hook_audio, out_path, api_key="testkey", avatar_path=avatar
            )
        assert result is None

    def test_cleans_up_partial_download_on_failure(self, hook_audio, out_path, avatar):
        create_resp = _make_resp({"id": "talk_partial"})
        done_resp   = _make_resp({"status": "done", "result_url": "https://cdn.d-id.com/r.mp4"})
        # dl raises after out_path is created (simulate partial write)
        import requests as req

        def _bad_dl(*args, **kwargs):
            out_path.write_bytes(b"partial")
            raise req.exceptions.ConnectionError("dropped")

        with patch("src.presenter.http_post", return_value=create_resp), \
             patch("src.presenter.http_get", side_effect=[done_resp, _bad_dl]), \
             patch("time.sleep"):
            result = generate_presenter_clip(
                hook_audio, out_path, api_key="testkey", avatar_path=avatar
            )

        assert result is None
        assert not out_path.exists()

    def test_creates_parent_dirs(self, hook_audio, avatar, tmp_path):
        deep_out = tmp_path / "a" / "b" / "presenter.mp4"
        create_resp = _make_resp({"id": "talk_dir"})
        done_resp   = _make_resp({"status": "done", "result_url": "https://cdn.d-id.com/r.mp4"})
        dl_resp     = _make_resp({})
        dl_resp.iter_content = MagicMock(return_value=[b"data"])

        with patch("src.presenter.http_post", return_value=create_resp), \
             patch("src.presenter.http_get", side_effect=[done_resp, dl_resp]):
            result = generate_presenter_clip(
                hook_audio, deep_out, api_key="testkey", avatar_path=avatar
            )

        assert result == deep_out
        assert deep_out.exists()
