"""Tests for the HeyGen presenter module.

All HTTP calls are mocked — no real API credentials needed.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from src.heygen_presenter import (
    _create_video,
    _create_video_text,
    _download,
    _poll_video,
    _upload_audio,
    generate_heygen_clip,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    m = MagicMock()
    m.status_code = status_code
    m.json.return_value = json_data
    m.text = str(json_data)
    m.iter_content.return_value = [b"fake-video-bytes"]
    return m


# ── _upload_audio ──────────────────────────────────────────────────────────────

class TestUploadAudio:
    def test_returns_url_on_success(self, tmp_path):
        audio = tmp_path / "hook.mp3"
        audio.write_bytes(b"fake-audio")
        resp = _mock_response({"data": {"url": "https://cdn.heygen.com/audio/abc.mp3"}})
        with patch("src.heygen_presenter.http_post", return_value=resp) as mock_post:
            url, asset_id = _upload_audio(audio, api_key="test-key")
        assert url == "https://cdn.heygen.com/audio/abc.mp3"
        assert asset_id == ""
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "upload.heygen.com" in args[0]
        assert "v1/asset" in args[0]
        # Must send raw binary, not multipart
        assert kwargs.get("data") == b"fake-audio"
        assert "files" not in kwargs
        assert kwargs.get("headers", {}).get("Content-Type") == "audio/mpeg"

    def test_returns_asset_id_when_provided(self, tmp_path):
        audio = tmp_path / "hook.mp3"
        audio.write_bytes(b"fake-audio")
        resp = _mock_response({"data": {"url": "https://cdn.heygen.com/audio/abc.mp3", "asset_id": "asset-xyz"}})
        with patch("src.heygen_presenter.http_post", return_value=resp):
            url, asset_id = _upload_audio(audio, api_key="test-key")
        assert url == "https://cdn.heygen.com/audio/abc.mp3"
        assert asset_id == "asset-xyz"

    def test_raises_when_no_url_or_asset_id_returned(self, tmp_path):
        audio = tmp_path / "hook.mp3"
        audio.write_bytes(b"fake-audio")
        resp = _mock_response({"data": {}})
        with patch("src.heygen_presenter.http_post", return_value=resp):
            with pytest.raises(RuntimeError, match="no URL or asset_id"):
                _upload_audio(audio, api_key="test-key")


# ── _create_video ─────────────────────────────────────────────────────────────

class TestCreateVideo:
    def test_returns_video_id(self):
        resp = _mock_response({"data": {"video_id": "vid-123"}})
        with patch("src.heygen_presenter.http_post", return_value=resp):
            vid_id = _create_video("key", "avatar-1", "https://cdn.heygen.com/a.mp3")
        assert vid_id == "vid-123"

    def test_uses_correct_aspect_ratio(self):
        resp = _mock_response({"data": {"video_id": "vid-456"}})
        with patch("src.heygen_presenter.http_post", return_value=resp) as mock_post:
            _create_video("key", "avatar-1", "https://audio.url", aspect="9:16")
        payload = mock_post.call_args[1]["json"]
        assert payload["dimension"] == {"width": 720, "height": 1280}

    def test_raises_when_no_video_id(self):
        resp = _mock_response({"data": {}})
        with patch("src.heygen_presenter.http_post", return_value=resp):
            with pytest.raises(RuntimeError, match="video creation failed"):
                _create_video("key", "avatar-1", "https://audio.url")


# ── _create_video_text ────────────────────────────────────────────────────────

class TestCreateVideoText:
    def test_returns_video_id(self):
        resp = _mock_response({"data": {"video_id": "vid-txt-1"}})
        with patch("src.heygen_presenter.http_post", return_value=resp):
            vid_id = _create_video_text("key", "avatar-1", "voice-1", "Hello world")
        assert vid_id == "vid-txt-1"

    def test_truncates_text_at_2000_chars(self):
        resp = _mock_response({"data": {"video_id": "vid-2"}})
        long_text = "x" * 3000
        with patch("src.heygen_presenter.http_post", return_value=resp) as mock_post:
            _create_video_text("key", "avatar-1", "voice-1", long_text)
        payload = mock_post.call_args[1]["json"]
        assert len(payload["video_inputs"][0]["voice"]["input_text"]) == 2000

    def test_voice_id_in_payload(self):
        resp = _mock_response({"data": {"video_id": "vid-v"}})
        with patch("src.heygen_presenter.http_post", return_value=resp) as mock_post:
            _create_video_text("key", "avatar-1", "my-voice-xyz", "Hello")
        payload = mock_post.call_args[1]["json"]
        assert payload["video_inputs"][0]["voice"]["voice_id"] == "my-voice-xyz"

    def test_avatar_id_in_payload(self):
        resp = _mock_response({"data": {"video_id": "vid-a"}})
        with patch("src.heygen_presenter.http_post", return_value=resp) as mock_post:
            _create_video_text("key", "my-avatar-id", "voice-1", "Hello")
        payload = mock_post.call_args[1]["json"]
        assert payload["video_inputs"][0]["character"]["avatar_id"] == "my-avatar-id"


# ── _poll_video ───────────────────────────────────────────────────────────────

class TestPollVideo:
    def test_returns_url_when_completed(self):
        resp = _mock_response({"data": {"status": "completed", "video_url": "https://cdn.heygen.com/v.mp4"}})
        with patch("src.heygen_presenter.http_get", return_value=resp):
            with patch("src.heygen_presenter.time.sleep"):
                url = _poll_video("key", "vid-1")
        assert url == "https://cdn.heygen.com/v.mp4"

    def test_raises_on_failed_status(self):
        resp = _mock_response({"data": {"status": "failed", "error": "render error"}})
        with patch("src.heygen_presenter.http_get", return_value=resp):
            with patch("src.heygen_presenter.time.sleep"):
                with pytest.raises(RuntimeError, match="failed"):
                    _poll_video("key", "vid-1")

    def test_raises_on_timeout(self):
        resp = _mock_response({"data": {"status": "processing"}})
        with patch("src.heygen_presenter.http_get", return_value=resp):
            with patch("src.heygen_presenter.time.sleep"):
                with patch("src.heygen_presenter._POLL_TIMEOUT", -1):
                    with pytest.raises(TimeoutError):
                        _poll_video("key", "vid-1")

    def test_polls_until_completed_on_second_call(self):
        processing = _mock_response({"data": {"status": "processing"}})
        completed  = _mock_response({"data": {"status": "completed", "video_url": "https://cdn.heygen.com/v.mp4"}})
        with patch("src.heygen_presenter.http_get", side_effect=[processing, completed]):
            with patch("src.heygen_presenter.time.sleep"):
                url = _poll_video("key", "vid-1")
        assert url == "https://cdn.heygen.com/v.mp4"


# ── generate_heygen_clip ──────────────────────────────────────────────────────

class TestGenerateHeygenClip:
    def _setup_mocks(self):
        upload_resp   = _mock_response({"data": {"url": "https://cdn.heygen.com/a.mp3"}})
        create_resp   = _mock_response({"data": {"video_id": "vid-abc"}})
        poll_resp     = _mock_response({"data": {"status": "completed", "video_url": "https://cdn.heygen.com/v.mp4"}})
        download_resp = _mock_response({})
        download_resp.iter_content.return_value = [b"video-content"]
        return upload_resp, create_resp, poll_resp, download_resp

    def test_returns_none_when_no_api_key(self, tmp_path):
        audio = tmp_path / "hook.mp3"
        audio.write_bytes(b"x")
        out   = tmp_path / "out.mp4"
        result = generate_heygen_clip(audio, out, api_key="", avatar_id="av-1")
        assert result is None

    def test_returns_none_when_no_avatar_id(self, tmp_path):
        audio = tmp_path / "hook.mp3"
        audio.write_bytes(b"x")
        out   = tmp_path / "out.mp4"
        result = generate_heygen_clip(audio, out, api_key="key", avatar_id="")
        assert result is None

    def test_returns_cached_clip_without_api_call(self, tmp_path):
        out = tmp_path / "cached.mp4"
        out.write_bytes(b"already-exists")
        with patch("src.heygen_presenter.http_post") as mock_post:
            result = generate_heygen_clip(
                tmp_path / "audio.mp3", out, api_key="key", avatar_id="av-1"
            )
        assert result == out
        mock_post.assert_not_called()

    def test_full_audio_driven_flow(self, tmp_path):
        audio = tmp_path / "hook.mp3"
        audio.write_bytes(b"fake-audio")
        out   = tmp_path / "out.mp4"

        upload_resp, create_resp, poll_resp, dl_resp = self._setup_mocks()

        with patch("src.heygen_presenter.http_post", side_effect=[upload_resp, create_resp]):
            with patch("src.heygen_presenter.http_get", side_effect=[poll_resp, dl_resp]):
                with patch("src.heygen_presenter.time.sleep"):
                    result = generate_heygen_clip(
                        audio, out, api_key="key", avatar_id="av-1"
                    )

        assert result == out
        assert out.exists()
        assert out.read_bytes() == b"video-content"

    def test_fallback_to_text_mode_when_audio_missing(self, tmp_path):
        audio = tmp_path / "missing.mp3"   # does NOT exist
        out   = tmp_path / "out.mp4"

        create_resp = _mock_response({"data": {"video_id": "vid-txt"}})
        poll_resp   = _mock_response({"data": {"status": "completed", "video_url": "https://cdn.heygen.com/v.mp4"}})
        dl_resp     = MagicMock()
        dl_resp.iter_content.return_value = [b"tts-video"]

        with patch("src.heygen_presenter.http_post", return_value=create_resp):
            with patch("src.heygen_presenter.http_get", side_effect=[poll_resp, dl_resp]):
                with patch("src.heygen_presenter.time.sleep"):
                    result = generate_heygen_clip(
                        audio, out,
                        api_key="key", avatar_id="av-1",
                        fallback_text="Hello world", voice_id="voice-1",
                    )

        assert result == out
        assert out.read_bytes() == b"tts-video"

    def test_tts_fallback_sends_voice_id_and_avatar_id_in_payload(self, tmp_path):
        """voice_id and avatar_id must appear in the TTS video generation payload."""
        audio = tmp_path / "missing.mp3"   # does NOT exist → triggers TTS path
        out   = tmp_path / "out.mp4"

        create_resp = _mock_response({"data": {"video_id": "vid-tts"}})
        poll_resp   = _mock_response({"data": {"status": "completed", "video_url": "https://cdn.heygen.com/v.mp4"}})
        dl_resp     = MagicMock()
        dl_resp.iter_content.return_value = [b"tts"]

        with patch("src.heygen_presenter.http_post", return_value=create_resp) as mock_post:
            with patch("src.heygen_presenter.http_get", side_effect=[poll_resp, dl_resp]):
                with patch("src.heygen_presenter.time.sleep"):
                    generate_heygen_clip(
                        audio, out,
                        api_key="key",
                        avatar_id="avatar-from-casual",
                        fallback_text="Hello world",
                        voice_id="heygen-voice-abc",
                    )

        payload = mock_post.call_args[1]["json"]
        voice   = payload["video_inputs"][0]["voice"]
        char    = payload["video_inputs"][0]["character"]
        assert voice["voice_id"]   == "heygen-voice-abc"
        assert char["avatar_id"]   == "avatar-from-casual"

    def test_returns_none_on_api_failure(self, tmp_path):
        audio = tmp_path / "hook.mp3"
        audio.write_bytes(b"x")
        out   = tmp_path / "out.mp4"

        with patch("src.heygen_presenter.http_post", side_effect=Exception("network error")):
            result = generate_heygen_clip(audio, out, api_key="key", avatar_id="av-1")

        assert result is None
        assert not out.exists()

    def test_cleans_up_partial_download_on_failure(self, tmp_path):
        audio = tmp_path / "hook.mp3"
        audio.write_bytes(b"x")
        out   = tmp_path / "out.mp4"

        upload_resp = _mock_response({"data": {"url": "https://cdn.heygen.com/a.mp3"}})
        create_resp = _mock_response({"data": {"video_id": "vid-abc"}})
        poll_resp   = _mock_response({"data": {"status": "completed", "video_url": "https://cdn.heygen.com/v.mp4"}})

        def _bad_download(url: str, **kwargs):
            # Write a partial file then raise to simulate a failed download
            out.write_bytes(b"partial")
            raise IOError("download interrupted")

        with patch("src.heygen_presenter.http_post", side_effect=[upload_resp, create_resp]):
            with patch("src.heygen_presenter.http_get", return_value=poll_resp):
                with patch("src.heygen_presenter.time.sleep"):
                    with patch("src.heygen_presenter._download", side_effect=_bad_download):
                        result = generate_heygen_clip(audio, out, api_key="key", avatar_id="av-1")

        assert result is None
        assert not out.exists()
