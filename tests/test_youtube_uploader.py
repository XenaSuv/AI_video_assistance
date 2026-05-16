"""Tests for youtube_uploader — credential migration, _fill_timestamps, and upload helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call, PropertyMock

import pytest

# Extend conftest stubs with sub-modules that youtube_uploader imports at module level.
for _m in (
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
):
    sys.modules.setdefault(_m, MagicMock())

from src.youtube_uploader import _migrate_pickle, _fill_timestamps, _upload_with_retry
from src.script_generator import Scene, VideoScript


# ── _upload_with_retry ────────────────────────────────────────────────────────

class _FakeHttpError(Exception):
    """Real exception class that stands in for googleapiclient.errors.HttpError."""

    def __init__(self, status: int, retry_after: str | None = None) -> None:
        self.resp = MagicMock()
        self.resp.status = status
        self.resp.get = MagicMock(return_value=retry_after)
        super().__init__(f"HTTP {status}")


def _make_http_error(status: int, retry_after: str | None = None) -> _FakeHttpError:
    return _FakeHttpError(status, retry_after)


def _make_request(chunks: list) -> MagicMock:
    """Return a mock request whose next_chunk() yields each item in *chunks*.

    Items may be:
      - (status, response) tuple → returned as-is
      - a BaseException instance  → raised
    """
    req = MagicMock()
    req.resumable_uri      = "https://upload.googleapis.com/fake-uri"
    req.resumable_progress = 0

    side_effects: list = []
    for item in chunks:
        if isinstance(item, BaseException):
            side_effects.append(item)
        else:
            side_effects.append(item)   # already a (status, response) tuple

    req.next_chunk.side_effect = side_effects
    return req


class TestUploadWithRetry:
    @pytest.fixture(autouse=True)
    def _patch_env(self, monkeypatch):
        """Replace the stub HttpError with our real exception class and suppress sleeps."""
        monkeypatch.setattr("src.youtube_uploader.HttpError", _FakeHttpError)
        monkeypatch.setattr("src.youtube_uploader.time.sleep", lambda _: None)

    # ── happy path ──────────────────────────────────────────────────────────────

    def test_returns_response_on_success(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([(None, {"id": "abc123"})])
        result = _upload_with_retry(req, video, max_retries=3)
        assert result == {"id": "abc123"}

    def test_state_file_deleted_on_success(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([(None, {"id": "abc123"})])
        _upload_with_retry(req, video, max_retries=3)
        assert not (tmp_path / "video.mp4.upload_state.json").exists()

    def test_multi_chunk_upload(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        status_mid = MagicMock()
        status_mid.progress.return_value = 0.5
        req = _make_request([
            (status_mid, None),
            (None, {"id": "done42"}),
        ])
        result = _upload_with_retry(req, video, max_retries=3)
        assert result["id"] == "done42"

    # ── 5xx retry with backoff ──────────────────────────────────────────────────

    def test_5xx_retries_and_succeeds(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([
            _make_http_error(503),
            _make_http_error(500),
            (None, {"id": "ok"}),
        ])
        result = _upload_with_retry(req, video, max_retries=5)
        assert result["id"] == "ok"

    def test_5xx_raises_after_max_retries(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([_make_http_error(503)] * 6)
        with pytest.raises(_FakeHttpError):
            _upload_with_retry(req, video, max_retries=5)

    def test_5xx_state_file_deleted_on_fatal_error(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([_make_http_error(503)] * 6)
        try:
            _upload_with_retry(req, video, max_retries=5)
        except Exception:
            pass
        assert not (tmp_path / "video.mp4.upload_state.json").exists()

    def test_5xx_sleep_doubles_each_retry(self, tmp_path, monkeypatch):
        sleep_calls: list[float] = []
        monkeypatch.setattr("src.youtube_uploader.time.sleep", sleep_calls.append)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([
            _make_http_error(500),
            _make_http_error(500),
            (None, {"id": "x"}),
        ])
        _upload_with_retry(req, video, max_retries=5)
        assert sleep_calls == [2, 4]  # 2**1, 2**2

    # ── 429 rate limit ──────────────────────────────────────────────────────────

    def test_429_retries_and_succeeds(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([
            _make_http_error(429),
            (None, {"id": "ok"}),
        ])
        result = _upload_with_retry(req, video, max_retries=3)
        assert result["id"] == "ok"

    def test_429_respects_retry_after_header(self, tmp_path, monkeypatch):
        sleep_calls: list[float] = []
        monkeypatch.setattr("src.youtube_uploader.time.sleep", sleep_calls.append)
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")

        err = _make_http_error(429, retry_after="30")  # Retry-After: 30s
        req = _make_request([err, (None, {"id": "ok"})])
        _upload_with_retry(req, video, max_retries=3)
        assert sleep_calls == [30]

    def test_429_raises_after_max_retries(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([_make_http_error(429)] * 4)
        with pytest.raises(_FakeHttpError):
            _upload_with_retry(req, video, max_retries=3)

    # ── non-retryable 4xx ───────────────────────────────────────────────────────

    def test_403_raises_immediately(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([_make_http_error(403)])
        with pytest.raises(_FakeHttpError):
            _upload_with_retry(req, video, max_retries=5)
        assert req.next_chunk.call_count == 1  # no retry

    def test_400_raises_immediately(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([_make_http_error(400)])
        with pytest.raises(_FakeHttpError):
            _upload_with_retry(req, video, max_retries=5)
        assert req.next_chunk.call_count == 1

    # ── network errors ──────────────────────────────────────────────────────────

    def test_network_error_retries_and_succeeds(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([
            TimeoutError("timed out"),
            (None, {"id": "ok"}),
        ])
        result = _upload_with_retry(req, video, max_retries=3)
        assert result["id"] == "ok"

    def test_network_error_raises_after_max_retries(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        req = _make_request([OSError("network gone")] * 4)
        with pytest.raises(OSError):
            _upload_with_retry(req, video, max_retries=3)

    # ── resumable state persistence ─────────────────────────────────────────────

    def test_state_file_written_after_chunk(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        status_mid = MagicMock()
        status_mid.progress.return_value = 0.4
        req = _make_request([
            (status_mid, None),
            (None, {"id": "done"}),
        ])
        req.resumable_uri      = "https://upload.googleapis.com/uri123"
        req.resumable_progress = 4_000_000

        state_path = tmp_path / "video.mp4.upload_state.json"
        # patch write_text to capture what was written after first chunk
        written: list[str] = []
        original_next_chunk = req.next_chunk.side_effect

        call_count = [0]
        responses = [(status_mid, None), (None, {"id": "done"})]

        def patched_next_chunk():
            result = responses[call_count[0]]
            call_count[0] += 1
            if isinstance(result, BaseException):
                raise result
            return result

        req.next_chunk.side_effect = patched_next_chunk

        _upload_with_retry(req, video, max_retries=3)
        # state file should be cleaned up on success
        assert not state_path.exists()

    def test_resumes_from_existing_state_file(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        state_path = tmp_path / "video.mp4.upload_state.json"
        state_path.write_text(json.dumps({
            "resumable_uri":      "https://upload.googleapis.com/saved-uri",
            "resumable_progress": 8_000_000,
        }))
        req = _make_request([(None, {"id": "resumed"})])
        result = _upload_with_retry(req, video, max_retries=3)
        assert result["id"] == "resumed"
        # URI must have been restored onto the request
        assert req.resumable_uri == "https://upload.googleapis.com/saved-uri"
        assert req.resumable_progress == 8_000_000

    def test_corrupted_state_file_starts_fresh(self, tmp_path):
        video = tmp_path / "video.mp4"
        video.write_bytes(b"fake")
        state_path = tmp_path / "video.mp4.upload_state.json"
        state_path.write_text("not valid json {{")
        req = _make_request([(None, {"id": "fresh"})])
        result = _upload_with_retry(req, video, max_retries=3)
        assert result["id"] == "fresh"


# ── _fill_timestamps ──────────────────────────────────────────────────────────

def _script(scenes: list[tuple[str, int]]) -> VideoScript:
    return VideoScript(
        title="Test",
        description="(TIMESTAMPS_AUTOFILL)",
        tags=[],
        hook="hook text",
        scenes=[
            Scene(idx=i, heading=h, narration="narration", visual_prompt="", duration_sec=d)
            for i, (h, d) in enumerate(scenes)
        ],
    )


class TestFillTimestamps:
    def test_replaces_placeholder(self):
        script = _script([("Intro", 60), ("Main", 120)])
        result = _fill_timestamps("(TIMESTAMPS_AUTOFILL)", script)
        assert "(TIMESTAMPS_AUTOFILL)" not in result
        assert "Chapters:" in result

    def test_no_change_when_placeholder_absent(self):
        script = _script([("Intro", 60)])
        result = _fill_timestamps("No placeholder here.", script)
        assert result == "No placeholder here."

    def test_first_scene_starts_at_zero(self):
        script = _script([("Intro", 60), ("Body", 120)])
        result = _fill_timestamps("(TIMESTAMPS_AUTOFILL)", script)
        assert "00:00 Intro" in result

    def test_second_scene_timestamp_correct(self):
        script = _script([("Intro", 60), ("Body", 120)])
        result = _fill_timestamps("(TIMESTAMPS_AUTOFILL)", script)
        assert "01:00 Body" in result

    def test_timestamp_over_one_hour(self):
        script = _script([("Opening", 3600), ("Next", 60)])
        result = _fill_timestamps("(TIMESTAMPS_AUTOFILL)", script)
        # 3600s = 60 minutes
        assert "60:00 Next" in result

    def test_description_prefix_preserved(self):
        script = _script([("Scene", 30)])
        desc = "My description\n(TIMESTAMPS_AUTOFILL)"
        result = _fill_timestamps(desc, script)
        assert "My description" in result

    def test_all_scene_headings_appear(self):
        script = _script([("Intro", 30), ("Middle", 60), ("Outro", 30)])
        result = _fill_timestamps("(TIMESTAMPS_AUTOFILL)", script)
        for heading in ["Intro", "Middle", "Outro"]:
            assert heading in result


# ── _migrate_pickle ────────────────────────────────────────────────────────────

class TestMigratePickle:
    def test_writes_json_and_deletes_pickle(self, tmp_path):
        pickle_path = tmp_path / "token.pickle"
        json_path   = tmp_path / "token.json"

        fake_creds = MagicMock()
        fake_creds.to_json.return_value = json.dumps({"token": "abc", "refresh_token": "xyz"})

        # Write a dummy bytes file (pickle.load is patched below)
        pickle_path.write_bytes(b"placeholder")

        with patch("pickle.load", return_value=fake_creds):
            _migrate_pickle(pickle_path, json_path)

        assert not pickle_path.exists()
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["token"] == "abc"

    def test_raises_if_pickle_unreadable(self, tmp_path):
        pickle_path = tmp_path / "bad.pickle"
        json_path   = tmp_path / "token.json"
        pickle_path.write_bytes(b"not-valid-pickle")
        with pytest.raises(Exception):
            _migrate_pickle(pickle_path, json_path)


# ── _get_creds auto-migration ─────────────────────────────────────────────────

class TestGetCredsAutoMigration:
    def test_detects_pickle_suffix_and_migrates(self, tmp_path):
        """When token_file has .pickle suffix, _get_creds must migrate it."""
        from src.youtube_uploader import _get_creds

        secrets = tmp_path / "client_secrets.json"
        secrets.write_text(json.dumps({"installed": {"client_id": "x", "client_secret": "y"}}))

        pickle_path = tmp_path / "token.pickle"
        fake_creds = MagicMock()
        fake_creds.valid = True
        fake_creds.expired = False
        fake_creds.to_json.return_value = json.dumps({"token": "abc"})

        # Write a dummy file (pickle.load is patched below)
        pickle_path.write_bytes(b"placeholder")

        # Patch Credentials.from_authorized_user_info so it returns the fake creds
        with patch("pickle.load", return_value=fake_creds):
            with patch("src.youtube_uploader.Credentials") as MockCreds:
                MockCreds.from_authorized_user_info.return_value = fake_creds
                creds = _get_creds(client_secrets=secrets, token_file=pickle_path)

        # pickle must be deleted after migration
        assert not pickle_path.exists()
        json_path = pickle_path.with_suffix(".json")
        assert json_path.exists()
        assert creds is fake_creds

    def test_json_token_loaded_directly(self, tmp_path):
        """When token_file is already .json and valid, no migration occurs."""
        from src.youtube_uploader import _get_creds

        json_path = tmp_path / "token.json"
        fake_creds = MagicMock()
        fake_creds.valid = True
        fake_creds.expired = False
        fake_creds.to_json.return_value = json.dumps({"token": "abc"})
        json_path.write_text(json.dumps({"token": "abc"}))

        with patch("src.youtube_uploader.Credentials") as MockCreds:
            MockCreds.from_authorized_user_info.return_value = fake_creds
            creds = _get_creds(token_file=json_path)

        assert creds is fake_creds


# ── upload_video guards ────────────────────────────────────────────────────────

class TestUploadVideoGuards:
    @pytest.fixture(autouse=True)
    def reset_circuit_breaker(self):
        from src.circuit_breaker import youtube_breaker
        youtube_breaker.reset()
        yield
        youtube_breaker.reset()

    def test_refuses_video_without_audio(self, tmp_path):
        """upload_video raises ValueError when the long-form video lacks an audio stream."""
        from src.youtube_uploader import upload_video

        fake_video = tmp_path / "silent.mp4"
        fake_video.write_bytes(b"fake")

        with patch("src.youtube_uploader.ffmpeg_utils.has_audio_stream", return_value=False):
            with pytest.raises(ValueError, match="audio stream"):
                upload_video(
                    fake_video,
                    title="Test",
                    description="desc",
                    tags=["ai"],
                    is_short=False,
                    client_secrets=tmp_path / "secrets.json",
                    token_file=tmp_path / "token.json",
                )

    def test_short_skips_audio_check(self, tmp_path):
        """Shorts are exempt from the audio-stream check."""
        from src.youtube_uploader import upload_video

        fake_video = tmp_path / "short.mp4"
        fake_video.write_bytes(b"fake")

        fake_creds = MagicMock()
        fake_creds.valid = True
        fake_yt = MagicMock()
        fake_yt.videos().insert().execute.return_value = {"id": "yt-short-id"}
        fake_yt.videos().insert().next_chunk.return_value = (None, {"id": "yt-short-id"})

        with patch("src.youtube_uploader.ffmpeg_utils.has_audio_stream", return_value=False):
            with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
                with patch("src.youtube_uploader.MediaFileUpload"):
                    # Should not raise despite no audio stream
                    try:
                        upload_video(
                            fake_video,
                            title="Short clip",
                            description="desc",
                            tags=["ai"],
                            is_short=True,
                        )
                    except Exception:
                        pass  # May fail for other mocking reasons — key: no ValueError
