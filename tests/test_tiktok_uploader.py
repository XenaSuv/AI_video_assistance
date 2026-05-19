"""Tests for src/tiktok_uploader.py — token management and upload helpers."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.tiktok_uploader import (
    _CHUNK_SIZE,
    _TOKEN_URL,
    _do_refresh,
    _get_access_token,
    _init_upload,
    _load_tokens,
    _save_tokens,
    _upload_chunks,
    post_short,
)


def _token_data(
    access_token: str = "access_123",
    refresh_token: str = "refresh_456",
    expires_at: float | None = None,
) -> dict:
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": expires_at if expires_at is not None else (time.time() + 3600),
    }


class TestLoadTokens:
    def test_loads_json_from_file(self, tmp_path):
        token_file = tmp_path / "token.json"
        data = _token_data()
        token_file.write_text(json.dumps(data))
        result = _load_tokens(token_file)
        assert result["access_token"] == "access_123"

    def test_returns_dict(self, tmp_path):
        token_file = tmp_path / "token.json"
        token_file.write_text(json.dumps(_token_data()))
        result = _load_tokens(token_file)
        assert isinstance(result, dict)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            _load_tokens(tmp_path / "missing.json")


class TestSaveTokens:
    def test_writes_json_to_file(self, tmp_path):
        token_file = tmp_path / "token.json"
        data = _token_data()
        _save_tokens(token_file, data)
        assert token_file.exists()

    def test_roundtrip(self, tmp_path):
        token_file = tmp_path / "token.json"
        data = _token_data("abc", "def", 9999999.0)
        _save_tokens(token_file, data)
        loaded = json.loads(token_file.read_text())
        assert loaded["access_token"] == "abc"
        assert loaded["refresh_token"] == "def"

    def test_creates_readable_json(self, tmp_path):
        token_file = tmp_path / "token.json"
        _save_tokens(token_file, {"key": "val"})
        parsed = json.loads(token_file.read_text())
        assert parsed == {"key": "val"}


class TestDoRefresh:
    def test_calls_http_post(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "access_token": "new_token",
            "refresh_token": "new_refresh",
            "expires_in": 86400,
        }
        with patch("src.tiktok_uploader.http_post", return_value=mock_resp) as mock_post:
            result = _do_refresh("key", "secret", "old_refresh")
        mock_post.assert_called_once()
        assert result["access_token"] == "new_token"

    def test_returns_response_json(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"access_token": "refreshed"}
        with patch("src.tiktok_uploader.http_post", return_value=mock_resp):
            result = _do_refresh("key", "secret", "refresh")
        assert result["access_token"] == "refreshed"

    def test_sends_correct_grant_type(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        with patch("src.tiktok_uploader.http_post", return_value=mock_resp) as mock_post:
            _do_refresh("key", "secret", "old_token")
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["data"]["grant_type"] == "refresh_token"


class TestGetAccessToken:
    def test_returns_valid_token_without_refresh(self, tmp_path):
        token_file = tmp_path / "token.json"
        data = _token_data("valid_token", expires_at=time.time() + 7200)
        token_file.write_text(json.dumps(data))
        result = _get_access_token("key", "secret", token_file)
        assert result == "valid_token"

    def test_refreshes_when_expired(self, tmp_path):
        token_file = tmp_path / "token.json"
        data = _token_data("old_token", "old_refresh", expires_at=time.time() - 100)
        token_file.write_text(json.dumps(data))

        mock_refresh_response = {
            "access_token": "new_token",
            "refresh_token": "new_refresh",
            "expires_in": 86400,
        }
        with patch("src.tiktok_uploader._do_refresh", return_value=mock_refresh_response):
            result = _get_access_token("key", "secret", token_file)
        assert result == "new_token"

    def test_saves_refreshed_token(self, tmp_path):
        token_file = tmp_path / "token.json"
        data = _token_data("old_token", "old_refresh", expires_at=0)
        token_file.write_text(json.dumps(data))

        mock_refresh_response = {
            "access_token": "refreshed",
            "refresh_token": "new_refresh",
            "expires_in": 86400,
        }
        with patch("src.tiktok_uploader._do_refresh", return_value=mock_refresh_response):
            _get_access_token("key", "secret", token_file)

        saved = json.loads(token_file.read_text())
        assert saved["access_token"] == "refreshed"

    def test_expiry_minus_300_triggers_refresh(self, tmp_path):
        token_file = tmp_path / "token.json"
        # expires_at is 200 seconds from now — less than 300s buffer → refresh
        data = _token_data("near_expiry", expires_at=time.time() + 200)
        token_file.write_text(json.dumps(data))

        mock_refresh_response = {
            "access_token": "refreshed_early",
            "expires_in": 86400,
        }
        with patch("src.tiktok_uploader._do_refresh", return_value=mock_refresh_response):
            result = _get_access_token("key", "secret", token_file)
        assert result == "refreshed_early"


class TestPostShort:
    def test_calls_get_access_token(self, tmp_path):
        video = tmp_path / "short.mp4"
        video.write_bytes(b"FAKE_VIDEO" * 100)

        with patch("src.tiktok_uploader._get_access_token", return_value="token") as mock_tok:
            with patch("src.tiktok_uploader._init_upload", return_value=("pub123", "https://upload.url")):
                with patch("src.tiktok_uploader._upload_chunks"):
                    with patch("src.tiktok_uploader.settings") as mock_settings:
                        mock_settings.tiktok_client_key = "ck"
                        mock_settings.tiktok_client_secret = "cs"
                        mock_settings.tiktok_token_file = tmp_path / "tok.json"
                        mock_settings.tiktok_privacy = "PUBLIC_TO_EVERYONE"
                        result = post_short(video, "Test caption", client_key="ck", client_secret="cs",
                                            token_file=tmp_path / "tok.json", privacy="PUBLIC_TO_EVERYONE")
        assert result == "pub123"

    def test_returns_publish_id(self, tmp_path):
        video = tmp_path / "short.mp4"
        video.write_bytes(b"FAKE" * 1000)

        with patch("src.tiktok_uploader._get_access_token", return_value="tok"):
            with patch("src.tiktok_uploader._init_upload", return_value=("vid999", "https://up.url")):
                with patch("src.tiktok_uploader._upload_chunks"):
                    result = post_short(
                        video, "caption",
                        client_key="k", client_secret="s",
                        token_file=tmp_path / "t.json",
                        privacy="PUBLIC_TO_EVERYONE",
                    )
        assert result == "vid999"


class TestInitUpload:
    def test_returns_publish_id_and_upload_url(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "data": {"publish_id": "pub_abc", "upload_url": "https://upload.tiktok.com/xyz"}
        }
        with patch("src.tiktok_uploader.http_post", return_value=mock_resp):
            pub_id, upload_url = _init_upload("token", "My Caption", 1024 * 1024, "PUBLIC_TO_EVERYONE")
        assert pub_id == "pub_abc"
        assert upload_url == "https://upload.tiktok.com/xyz"

    def test_sends_authorization_header(self):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"publish_id": "p", "upload_url": "u"}}
        with patch("src.tiktok_uploader.http_post", return_value=mock_resp) as mock_post:
            _init_upload("mytoken", "caption", 1024, "PUBLIC_TO_EVERYONE")
        headers = mock_post.call_args[1]["headers"]
        assert "Bearer mytoken" in headers["Authorization"]

    def test_truncates_caption_to_2200_chars(self):
        long_caption = "x" * 3000
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"publish_id": "p", "upload_url": "u"}}
        with patch("src.tiktok_uploader.http_post", return_value=mock_resp) as mock_post:
            _init_upload("token", long_caption, 1024, "PUBLIC_TO_EVERYONE")
        sent_title = mock_post.call_args[1]["json"]["post_info"]["title"]
        assert len(sent_title) <= 2200

    def test_calculates_chunk_count(self):
        video_size = _CHUNK_SIZE * 3 + 1000  # 3 full chunks + partial
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"data": {"publish_id": "p", "upload_url": "u"}}
        with patch("src.tiktok_uploader.http_post", return_value=mock_resp) as mock_post:
            _init_upload("token", "caption", video_size, "PUBLIC_TO_EVERYONE")
        source_info = mock_post.call_args[1]["json"]["source_info"]
        assert source_info["total_chunk_count"] == 4


class TestUploadChunks:
    def test_uploads_single_chunk(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"x" * 1000)

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("src.tiktok_uploader.requests.put", return_value=mock_resp) as mock_put:
            _upload_chunks("https://upload.url", video)

        assert mock_put.call_count >= 1

    def test_sends_content_range_header(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"FAKE_VIDEO_BYTES" * 50)  # 800 bytes

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch("src.tiktok_uploader.requests.put", return_value=mock_resp) as mock_put:
            _upload_chunks("https://upload.url", video)

        headers = mock_put.call_args_list[0][1]["headers"]
        assert "Content-Range" in headers

    def test_raises_on_http_error(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"CHUNK_DATA" * 10)

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("HTTP 400")

        with patch("src.tiktok_uploader.requests.put", return_value=mock_resp):
            with pytest.raises(Exception, match="HTTP 400"):
                _upload_chunks("https://upload.url", video)


class TestMain:
    def test_no_auth_arg_does_nothing(self):
        from src.tiktok_uploader import main
        with patch("sys.argv", ["tiktok_uploader"]):
            with patch("src.tiktok_uploader.run_auth_flow") as mock_auth:
                main()
        mock_auth.assert_not_called()

    def test_auth_with_missing_credentials_exits(self, capsys):
        from src.tiktok_uploader import main
        with patch("sys.argv", ["tiktok_uploader", "--auth"]):
            with patch("src.tiktok_uploader.settings") as mock_settings:
                mock_settings.tiktok_client_key = ""
                mock_settings.tiktok_client_secret = ""
                with pytest.raises(SystemExit):
                    main()
        out = capsys.readouterr().out
        assert "TIKTOK_CLIENT_KEY" in out or "TIKTOK_CLIENT_SECRET" in out

    def test_auth_with_credentials_calls_run_auth_flow(self, tmp_path):
        from src.tiktok_uploader import main
        token_file = tmp_path / "tok.json"
        with patch("sys.argv", ["tiktok_uploader", "--auth"]):
            with patch("src.tiktok_uploader.settings") as mock_settings:
                mock_settings.tiktok_client_key = "key123"
                mock_settings.tiktok_client_secret = "sec456"
                mock_settings.tiktok_token_file = token_file
                with patch("src.tiktok_uploader.run_auth_flow") as mock_auth:
                    main()
        mock_auth.assert_called_once_with("key123", "sec456", token_file)
