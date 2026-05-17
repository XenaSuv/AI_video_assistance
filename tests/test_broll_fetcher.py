"""Tests for src/broll_fetcher.py — HTTP calls mocked, ffmpeg mocked."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.script_generator import Scene


# ── helpers ────────────────────────────────────────────────────────────────────

def _scene(idx: int = 0, video_query: str = "robot arm") -> Scene:
    s = Scene(
        idx=idx, heading="AI", narration="robots", visual_prompt="robot",
        duration_sec=30,
    )
    s.video_query = video_query
    return s


def _pexels_video(width: int = 1280, height: int = 720, quality: str = "hd") -> dict:
    return {
        "id": 123,
        "width": width,
        "height": height,
        "duration": 15,
        "url": "https://pexels.com/video/123",
        "user": {"name": "TestPhotog"},
        "video_files": [
            {"quality": quality, "width": width, "height": height,
             "link": "https://cdn.pexels.com/video.mp4"},
        ],
    }


def _pixabay_video(url: str = "https://cdn.pixabay.com/video.mp4") -> dict:
    return {
        "id": 456,
        "videos": {
            "large":  {"url": url, "width": 1920, "height": 1080},
            "medium": {"url": url, "width": 1280, "height": 720},
        },
    }


# ── import after stubs are in place ─────────────────────────────────────────

from src.broll_fetcher import (
    _pexels_search,
    _pexels_best_file,
    _pixabay_search,
    _pixabay_best_url,
    _download,
    _process_clip,
    save_pexels_credit,
    get_pexels_credits,
    fetch_broll_clip,
    fetch_broll_for_short,
)


# ═══════════════════════════════════════════════════════════════════════════════
# _pexels_search
# ═══════════════════════════════════════════════════════════════════════════════

class TestPexelsSearch:
    def test_returns_empty_when_no_key(self):
        with patch("src.broll_fetcher.settings") as ms:
            ms.pexels_api_key = ""
            result = _pexels_search("robot")
        assert result == []

    def test_calls_api_with_correct_params(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"videos": [_pexels_video()]}
        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher.http_get", return_value=mock_resp) as mock_get:
            ms.pexels_api_key = "pexels_key_123"
            result = _pexels_search("robot arm")
        mock_get.assert_called_once()
        params = mock_get.call_args[1]["params"]
        assert params["query"] == "robot arm"
        assert result[0]["id"] == 123

    def test_returns_videos_list(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"videos": [_pexels_video(), _pexels_video()]}
        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher.http_get", return_value=mock_resp):
            ms.pexels_api_key = "key"
            result = _pexels_search("test")
        assert len(result) == 2

    def test_returns_empty_on_missing_key_in_response(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher.http_get", return_value=mock_resp):
            ms.pexels_api_key = "key"
            result = _pexels_search("test")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# _pexels_best_file
# ═══════════════════════════════════════════════════════════════════════════════

class TestPexelsBestFile:
    def test_prefers_hd_quality(self):
        video = {
            "video_files": [
                {"quality": "sd", "width": 640, "link": "sd.mp4"},
                {"quality": "hd", "width": 1280, "link": "hd.mp4"},
            ]
        }
        result = _pexels_best_file(video)
        assert result["link"] == "hd.mp4"

    def test_falls_back_to_sd_when_no_hd(self):
        video = {
            "video_files": [
                {"quality": "sd", "width": 640, "link": "sd.mp4"},
            ]
        }
        result = _pexels_best_file(video)
        assert result is not None
        assert result["link"] == "sd.mp4"

    def test_returns_none_when_no_files(self):
        assert _pexels_best_file({"video_files": []}) is None
        assert _pexels_best_file({}) is None

    def test_selects_widest_among_hd(self):
        video = {
            "video_files": [
                {"quality": "hd", "width": 1280, "link": "hd1280.mp4"},
                {"quality": "hd", "width": 1920, "link": "hd1920.mp4"},
            ]
        }
        result = _pexels_best_file(video)
        assert result["link"] == "hd1920.mp4"


# ═══════════════════════════════════════════════════════════════════════════════
# _pixabay_search
# ═══════════════════════════════════════════════════════════════════════════════

class TestPixabaySearch:
    def test_returns_empty_when_no_key(self):
        with patch("src.broll_fetcher.settings") as ms:
            ms.pixabay_api_key = ""
            result = _pixabay_search("robot")
        assert result == []

    def test_calls_api_and_returns_hits(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"hits": [_pixabay_video()]}
        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher.http_get", return_value=mock_resp):
            ms.pixabay_api_key = "pixabay_key"
            result = _pixabay_search("nature")
        assert len(result) == 1

    def test_returns_empty_when_hits_missing(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher.http_get", return_value=mock_resp):
            ms.pixabay_api_key = "key"
            result = _pixabay_search("test")
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════════
# _pixabay_best_url
# ═══════════════════════════════════════════════════════════════════════════════

class TestPixabayBestUrl:
    def test_prefers_large(self):
        video = _pixabay_video("https://large.mp4")
        result = _pixabay_best_url(video)
        assert result == "https://large.mp4"

    def test_falls_back_to_medium(self):
        video = {"videos": {"medium": {"url": "medium.mp4", "width": 1280}}}
        result = _pixabay_best_url(video)
        assert result == "medium.mp4"

    def test_falls_back_to_small(self):
        video = {"videos": {"small": {"url": "small.mp4", "width": 640}}}
        result = _pixabay_best_url(video)
        assert result == "small.mp4"

    def test_returns_none_when_no_urls(self):
        assert _pixabay_best_url({"videos": {}}) is None
        assert _pixabay_best_url({}) is None


# ═══════════════════════════════════════════════════════════════════════════════
# _download
# ═══════════════════════════════════════════════════════════════════════════════

class TestDownload:
    def test_writes_chunks_to_file(self, tmp_path):
        out = tmp_path / "video.mp4"
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"chunk1", b"chunk2", b""]

        with patch("src.broll_fetcher.http_get", return_value=mock_resp):
            result = _download("https://example.com/video.mp4", out)

        assert result == out
        assert out.read_bytes() == b"chunk1chunk2"

    def test_skips_empty_chunks(self, tmp_path):
        out = tmp_path / "video.mp4"
        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = [b"data", b"", b"more"]

        with patch("src.broll_fetcher.http_get", return_value=mock_resp):
            _download("https://example.com/v.mp4", out)

        assert out.read_bytes() == b"datamore"


# ═══════════════════════════════════════════════════════════════════════════════
# _process_clip
# ═══════════════════════════════════════════════════════════════════════════════

class TestProcessClip:
    def test_resizes_if_wrong_size(self, tmp_path):
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"video")
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.ffmpeg_utils.video_size", return_value=(640, 360)), \
             patch("src.broll_fetcher.ffmpeg_utils.resize_video") as mock_resize, \
             patch("src.broll_fetcher.ffmpeg_utils.loop_and_trim") as mock_trim:
            # Create the resized file so unlink works
            resized = raw.with_stem(raw.stem + "_resized")
            mock_resize.side_effect = lambda *a, **kw: resized.write_bytes(b"resized")
            result = _process_clip(raw, 30.0, out)

        mock_resize.assert_called_once()
        mock_trim.assert_called_once()

    def test_skips_resize_when_correct_size(self, tmp_path):
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"video")
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.ffmpeg_utils.video_size", return_value=(1280, 720)), \
             patch("src.broll_fetcher.ffmpeg_utils.resize_video") as mock_resize, \
             patch("src.broll_fetcher.ffmpeg_utils.loop_and_trim") as mock_trim:
            result = _process_clip(raw, 30.0, out)

        mock_resize.assert_not_called()
        mock_trim.assert_called_once_with(raw, out, target_sec=30.0)

    def test_cleans_up_raw_file(self, tmp_path):
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"video")
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.ffmpeg_utils.video_size", return_value=(1280, 720)), \
             patch("src.broll_fetcher.ffmpeg_utils.loop_and_trim"):
            _process_clip(raw, 30.0, out)

        assert not raw.exists()


# ═══════════════════════════════════════════════════════════════════════════════
# save_pexels_credit / get_pexels_credits
# ═══════════════════════════════════════════════════════════════════════════════

class TestPexelsCredits:
    def test_save_creates_credits_file(self, tmp_path):
        save_pexels_credit(tmp_path, "TestPhotog", "https://pexels.com/video/123")
        credits = get_pexels_credits(tmp_path)
        assert len(credits) == 1
        assert "TestPhotog" in credits[0]

    def test_no_duplicates(self, tmp_path):
        save_pexels_credit(tmp_path, "Photog", "https://url")
        save_pexels_credit(tmp_path, "Photog", "https://url")
        credits = get_pexels_credits(tmp_path)
        assert len(credits) == 1

    def test_appends_multiple_credits(self, tmp_path):
        save_pexels_credit(tmp_path, "Photog1", "https://url1")
        save_pexels_credit(tmp_path, "Photog2", "https://url2")
        credits = get_pexels_credits(tmp_path)
        assert len(credits) == 2

    def test_get_returns_empty_when_no_file(self, tmp_path):
        credits = get_pexels_credits(tmp_path)
        assert credits == []


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_broll_clip
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchBrollClip:
    def test_returns_none_when_no_video_query(self, tmp_path):
        scene = _scene(0, video_query="")
        scene.video_query = ""
        result = fetch_broll_clip(scene, tmp_path / "out.mp4")
        assert result is None

    def test_returns_none_when_no_api_keys(self, tmp_path):
        scene = _scene(0)
        with patch("src.broll_fetcher.settings") as ms:
            ms.pexels_api_key = ""
            ms.pixabay_api_key = ""
            result = fetch_broll_clip(scene, tmp_path / "out.mp4")
        assert result is None

    def test_pexels_success_returns_clip(self, tmp_path):
        scene = _scene(0, "robot arm")
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher._pexels_search", return_value=[_pexels_video()]), \
             patch("src.broll_fetcher._download") as mock_dl, \
             patch("src.broll_fetcher._process_clip", return_value=out) as mock_proc:
            ms.pexels_api_key = "key"
            ms.pixabay_api_key = ""
            result = fetch_broll_clip(scene, out)

        assert result == out
        mock_dl.assert_called_once()
        mock_proc.assert_called_once()

    def test_pexels_saves_credit_when_run_dir_given(self, tmp_path):
        scene = _scene(0, "robot arm")
        out = tmp_path / "out.mp4"
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher._pexels_search", return_value=[_pexels_video()]), \
             patch("src.broll_fetcher._download"), \
             patch("src.broll_fetcher._process_clip", return_value=out):
            ms.pexels_api_key = "key"
            ms.pixabay_api_key = ""
            fetch_broll_clip(scene, out, run_dir=run_dir)

        credits = get_pexels_credits(run_dir)
        assert len(credits) == 1

    def test_falls_through_to_pixabay_when_pexels_empty(self, tmp_path):
        scene = _scene(0, "nature scene")
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher._pexels_search", return_value=[]), \
             patch("src.broll_fetcher._pixabay_search", return_value=[_pixabay_video()]), \
             patch("src.broll_fetcher._download"), \
             patch("src.broll_fetcher._process_clip", return_value=out) as mock_proc:
            ms.pexels_api_key = "pkey"
            ms.pixabay_api_key = "pixkey"
            result = fetch_broll_clip(scene, out)

        assert result == out
        mock_proc.assert_called_once()

    def test_pixabay_success_returns_clip(self, tmp_path):
        scene = _scene(0, "mountain")
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher._pixabay_search", return_value=[_pixabay_video()]), \
             patch("src.broll_fetcher._download"), \
             patch("src.broll_fetcher._process_clip", return_value=out):
            ms.pexels_api_key = ""
            ms.pixabay_api_key = "key"
            result = fetch_broll_clip(scene, out)

        assert result == out

    def test_pexels_exception_falls_back_to_pixabay(self, tmp_path):
        scene = _scene(0, "office")
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher._pexels_search", side_effect=RuntimeError("network")), \
             patch("src.broll_fetcher._pixabay_search", return_value=[_pixabay_video()]), \
             patch("src.broll_fetcher._download"), \
             patch("src.broll_fetcher._process_clip", return_value=out):
            ms.pexels_api_key = "pkey"
            ms.pixabay_api_key = "pixkey"
            result = fetch_broll_clip(scene, out)

        assert result == out

    def test_returns_none_when_all_fail(self, tmp_path):
        scene = _scene(0)
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher._pexels_search", return_value=[]), \
             patch("src.broll_fetcher._pixabay_search", return_value=[]):
            ms.pexels_api_key = "pkey"
            ms.pixabay_api_key = "pixkey"
            result = fetch_broll_clip(scene, out)

        assert result is None

    def test_skips_pexels_video_without_files(self, tmp_path):
        scene = _scene(0, "robot")
        out = tmp_path / "out.mp4"
        video_no_files = {"video_files": [], "id": 1}

        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher._pexels_search", return_value=[video_no_files]), \
             patch("src.broll_fetcher._pixabay_search", return_value=[]):
            ms.pexels_api_key = "pkey"
            ms.pixabay_api_key = "pixkey"
            result = fetch_broll_clip(scene, out)

        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_broll_for_short
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchBrollForShort:
    def test_returns_none_when_no_keys(self, tmp_path):
        with patch("src.broll_fetcher.settings") as ms:
            ms.pexels_api_key = ""
            ms.pixabay_api_key = ""
            result = fetch_broll_for_short("robot", 15.0, tmp_path / "out.mp4")
        assert result is None

    def test_pexels_success(self, tmp_path):
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher._pexels_search", return_value=[_pexels_video()]), \
             patch("src.broll_fetcher._download"), \
             patch("src.broll_fetcher.ffmpeg_utils.make_vertical"), \
             patch("src.broll_fetcher.ffmpeg_utils.loop_and_trim"):
            ms.pexels_api_key = "key"
            ms.pixabay_api_key = ""
            result = fetch_broll_for_short("robot", 15.0, out)

        assert result == out

    def test_pixabay_fallback(self, tmp_path):
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher._pixabay_search", return_value=[_pixabay_video()]), \
             patch("src.broll_fetcher._download"), \
             patch("src.broll_fetcher.ffmpeg_utils.make_vertical"), \
             patch("src.broll_fetcher.ffmpeg_utils.loop_and_trim"):
            ms.pexels_api_key = ""
            ms.pixabay_api_key = "key"
            result = fetch_broll_for_short("nature", 15.0, out)

        assert result == out

    def test_pexels_error_falls_back_to_pixabay(self, tmp_path):
        out = tmp_path / "out.mp4"

        with patch("src.broll_fetcher.settings") as ms, \
             patch("src.broll_fetcher._pexels_search", side_effect=RuntimeError("down")), \
             patch("src.broll_fetcher._pixabay_search", return_value=[_pixabay_video()]), \
             patch("src.broll_fetcher._download"), \
             patch("src.broll_fetcher.ffmpeg_utils.make_vertical"), \
             patch("src.broll_fetcher.ffmpeg_utils.loop_and_trim"):
            ms.pexels_api_key = "pkey"
            ms.pixabay_api_key = "pixkey"
            result = fetch_broll_for_short("robot", 15.0, out)

        assert result == out
