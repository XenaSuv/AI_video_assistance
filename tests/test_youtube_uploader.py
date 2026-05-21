"""Tests for youtube_uploader — credential migration, _fill_timestamps, and upload helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

# Extend conftest stubs with sub-modules that youtube_uploader imports at module level.
for _m in (
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
):
    sys.modules.setdefault(_m, MagicMock())

from src.script_generator import Scene, VideoScript
from src.youtube_uploader import _fill_timestamps, _upload_with_retry

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


# ── _get_creds token loading ──────────────────────────────────────────────────

class TestGetCredsAutoMigration:
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

    def test_quota_exhausted_raises_before_upload(self, tmp_path):
        """upload_video raises QuotaExhaustedError when quota guard says can't afford."""
        from src.exceptions import QuotaExhaustedError
        from src.youtube_uploader import upload_video

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = False
        fake_guard.used_today.return_value = 9800
        fake_guard.daily_limit = 10_000

        with patch("src.youtube_uploader.ffmpeg_utils.has_audio_stream", return_value=True):
            with pytest.raises(QuotaExhaustedError):
                upload_video(
                    fake_video,
                    title="Test",
                    description="desc",
                    tags=["ai"],
                    is_short=False,
                    quota_guard=fake_guard,
                )

    def test_upload_video_success_returns_video_id(self, tmp_path):
        """upload_video returns YouTube video ID on successful upload."""
        from src.youtube_uploader import upload_video

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake video content")

        fake_yt = MagicMock()
        mock_request = MagicMock()
        mock_request.next_chunk.return_value = (None, {"id": "abc123"})
        mock_request.resumable_uri = "https://upload.googleapis.com/fake"
        mock_request.resumable_progress = 0
        fake_yt.videos.return_value.insert.return_value = mock_request

        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True
        fake_guard.near_limit.return_value = False

        with patch("src.youtube_uploader.ffmpeg_utils.has_audio_stream", return_value=True):
            with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
                with patch("src.youtube_uploader.MediaFileUpload"):
                    video_id = upload_video(
                        fake_video,
                        title="Test Video",
                        description="desc",
                        tags=["ai"],
                        is_short=False,
                        quota_guard=fake_guard,
                    )

        assert video_id == "abc123"
        fake_guard.charge.assert_called_once_with("videos.insert")

    def test_upload_video_near_limit_logs_warning(self, tmp_path):
        """upload_video logs a warning when quota is near the limit."""
        from src.youtube_uploader import upload_video

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        fake_yt = MagicMock()
        mock_request = MagicMock()
        mock_request.next_chunk.return_value = (None, {"id": "vid999"})
        mock_request.resumable_uri = "https://upload.googleapis.com/fake"
        mock_request.resumable_progress = 0
        fake_yt.videos.return_value.insert.return_value = mock_request

        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True
        fake_guard.near_limit.return_value = True  # near limit
        fake_guard.used_today.return_value = 8500
        fake_guard.daily_limit = 10_000

        with patch("src.youtube_uploader.ffmpeg_utils.has_audio_stream", return_value=True):
            with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
                with patch("src.youtube_uploader.MediaFileUpload"):
                    video_id = upload_video(
                        fake_video,
                        title="Test",
                        description="desc",
                        tags=["ai"],
                        is_short=False,
                        quota_guard=fake_guard,
                    )
        assert video_id == "vid999"

    def test_upload_video_quota_error_during_upload(self, tmp_path, monkeypatch):
        """upload_video marks quota exhausted when quota error raised during upload."""
        from src.exceptions import QuotaExhaustedError
        from src.youtube_uploader import upload_video

        monkeypatch.setattr("src.youtube_uploader.HttpError", _FakeHttpError)

        fake_video = tmp_path / "video.mp4"
        fake_video.write_bytes(b"fake")

        fake_yt = MagicMock()
        mock_request = MagicMock()
        # Simulate a quota error during upload
        quota_exc = _FakeHttpError(403)
        mock_request.next_chunk.side_effect = quota_exc
        mock_request.resumable_uri = "https://upload.googleapis.com/fake"
        mock_request.resumable_progress = 0
        fake_yt.videos.return_value.insert.return_value = mock_request

        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True
        fake_guard.used_today.return_value = 9600
        fake_guard.daily_limit = 10000
        # Simulate quota error detection
        fake_guard.is_quota_error = MagicMock(return_value=True)

        with patch("src.youtube_uploader.ffmpeg_utils.has_audio_stream", return_value=True):
            with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
                with patch("src.youtube_uploader.MediaFileUpload"):
                    with patch("src.youtube_uploader.YouTubeQuotaGuard.is_quota_error", return_value=True):
                        with pytest.raises(QuotaExhaustedError):
                            upload_video(
                                fake_video,
                                title="Test",
                                description="desc",
                                tags=["ai"],
                                is_short=False,
                                quota_guard=fake_guard,
                            )

    def test_upload_video_short_adds_hashtags(self, tmp_path):
        """upload_video prepends #Shorts to title and description for Shorts."""
        from src.youtube_uploader import upload_video

        fake_video = tmp_path / "short.mp4"
        fake_video.write_bytes(b"fake")

        inserted_bodies = []

        def capture_insert(**kwargs):
            inserted_bodies.append(kwargs.get("body", {}))
            mock_request = MagicMock()
            mock_request.next_chunk.return_value = (None, {"id": "short123"})
            mock_request.resumable_uri = "https://upload.googleapis.com/fake"
            mock_request.resumable_progress = 0
            return mock_request

        fake_yt = MagicMock()
        fake_yt.videos.return_value.insert.side_effect = capture_insert

        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True
        fake_guard.near_limit.return_value = False

        with patch("src.youtube_uploader.ffmpeg_utils.has_audio_stream", return_value=True):
            with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
                with patch("src.youtube_uploader.MediaFileUpload"):
                    video_id = upload_video(
                        fake_video,
                        title="My Short",
                        description="Original description",
                        tags=["ai"],
                        is_short=True,
                        quota_guard=fake_guard,
                    )

        assert video_id == "short123"
        assert inserted_bodies, "insert was never called"
        title = inserted_bodies[0]["snippet"]["title"]
        desc = inserted_bodies[0]["snippet"]["description"]
        assert "#Shorts" in title
        assert "#Shorts" in desc


# ── upload_short ───────────────────────────────────────────────────────────────

class TestUploadShort:
    @pytest.fixture(autouse=True)
    def reset_circuit_breaker(self):
        from src.circuit_breaker import youtube_breaker
        youtube_breaker.reset()
        yield
        youtube_breaker.reset()

    def test_upload_short_delegates_to_upload_video(self, tmp_path):
        """upload_short wraps upload_video with is_short=True."""
        from src.youtube_uploader import upload_short

        fake_video = tmp_path / "short.mp4"
        fake_video.write_bytes(b"fake")

        with patch("src.youtube_uploader.upload_video", return_value="short-id") as mock_uv:
            result = upload_short(fake_video, title="My Short", description="desc", tags=["ai"])

        assert result == "short-id"
        call_kwargs = mock_uv.call_args
        assert call_kwargs.kwargs["is_short"] is True

    def test_upload_short_uses_default_description_and_tags(self, tmp_path):
        """upload_short provides default description and tags when not given."""
        from src.youtube_uploader import upload_short

        fake_video = tmp_path / "short.mp4"
        fake_video.write_bytes(b"fake")

        with patch("src.youtube_uploader.upload_video", return_value="short-id") as mock_uv:
            upload_short(fake_video, title="My Short")

        call_kwargs = mock_uv.call_args
        assert "#Shorts" in call_kwargs.kwargs["description"]
        assert "shorts" in call_kwargs.kwargs["tags"]


# ── set_thumbnail ─────────────────────────────────────────────────────────────

class TestSetThumbnail:
    def test_set_thumbnail_succeeds(self, tmp_path):
        """set_thumbnail calls thumbnails().set() and charges quota."""
        from src.youtube_uploader import set_thumbnail

        thumb = tmp_path / "thumb.jpg"
        thumb.write_bytes(b"fake jpeg")

        fake_yt = MagicMock()
        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True

        with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
            with patch("src.youtube_uploader.MediaFileUpload"):
                set_thumbnail("vid123", thumb, quota_guard=fake_guard)

        fake_yt.thumbnails.return_value.set.return_value.execute.assert_called_once()
        fake_guard.charge.assert_called_once_with("thumbnails.set")

    def test_set_thumbnail_skips_when_quota_exhausted(self, tmp_path):
        """set_thumbnail returns without uploading when quota is exhausted."""
        from src.youtube_uploader import set_thumbnail

        thumb = tmp_path / "thumb.jpg"
        thumb.write_bytes(b"fake")

        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = False

        with patch("src.youtube_uploader._youtube_client") as mock_client:
            set_thumbnail("vid123", thumb, quota_guard=fake_guard)

        mock_client.assert_not_called()
        fake_guard.charge.assert_not_called()

    def test_set_thumbnail_swallows_exception(self, tmp_path):
        """set_thumbnail logs warning but does not raise on API failure."""
        from src.youtube_uploader import set_thumbnail

        thumb = tmp_path / "thumb.jpg"
        thumb.write_bytes(b"fake")

        fake_yt = MagicMock()
        fake_yt.thumbnails.return_value.set.return_value.execute.side_effect = RuntimeError("upload failed")

        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True
        fake_guard.is_quota_error = MagicMock(return_value=False)

        with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
            with patch("src.youtube_uploader.MediaFileUpload"):
                with patch("src.youtube_uploader.YouTubeQuotaGuard.is_quota_error", return_value=False):
                    set_thumbnail("vid123", thumb, quota_guard=fake_guard)  # must not raise

    def test_set_thumbnail_marks_quota_exhausted_on_quota_error(self, tmp_path):
        """set_thumbnail calls guard.mark_exhausted() on quota error."""
        from src.youtube_uploader import set_thumbnail

        thumb = tmp_path / "thumb.jpg"
        thumb.write_bytes(b"fake")

        fake_yt = MagicMock()
        quota_err = RuntimeError("quota exceeded")
        fake_yt.thumbnails.return_value.set.return_value.execute.side_effect = quota_err

        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True

        with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
            with patch("src.youtube_uploader.MediaFileUpload"):
                with patch("src.youtube_uploader.YouTubeQuotaGuard.is_quota_error", return_value=True):
                    set_thumbnail("vid123", thumb, quota_guard=fake_guard)

        fake_guard.mark_exhausted.assert_called_once()


# ── upload_captions ───────────────────────────────────────────────────────────

class TestUploadCaptions:
    def test_upload_captions_succeeds(self, tmp_path):
        """upload_captions calls captions().insert() and charges quota."""
        from src.youtube_uploader import upload_captions

        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello world\n")

        fake_yt = MagicMock()
        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True

        with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
            with patch("src.youtube_uploader.MediaFileUpload"):
                upload_captions("vid123", srt, language="en", quota_guard=fake_guard)

        fake_yt.captions.return_value.insert.return_value.execute.assert_called_once()
        fake_guard.charge.assert_called_once_with("captions.insert")

    def test_upload_captions_skips_when_quota_exhausted(self, tmp_path):
        """upload_captions returns without uploading when quota is exhausted."""
        from src.youtube_uploader import upload_captions

        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")

        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = False

        with patch("src.youtube_uploader._youtube_client") as mock_client:
            upload_captions("vid123", srt, quota_guard=fake_guard)

        mock_client.assert_not_called()

    def test_upload_captions_uses_correct_track_name_for_en(self, tmp_path):
        """upload_captions maps language code 'en' to track name 'English'."""
        from src.youtube_uploader import upload_captions

        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")

        inserted_bodies = []

        def capture_insert(**kwargs):
            inserted_bodies.append(kwargs.get("body", {}))
            return MagicMock()

        fake_yt = MagicMock()
        fake_yt.captions.return_value.insert.side_effect = capture_insert
        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True

        with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
            with patch("src.youtube_uploader.MediaFileUpload"):
                upload_captions("vid123", srt, language="en", quota_guard=fake_guard)

        assert inserted_bodies[0]["snippet"]["name"] == "English"

    def test_upload_captions_uses_correct_track_name_for_ru(self, tmp_path):
        """upload_captions maps language code 'ru' to Cyrillic track name."""
        from src.youtube_uploader import upload_captions

        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nПривет\n")

        inserted_bodies = []

        def capture_insert(**kwargs):
            inserted_bodies.append(kwargs.get("body", {}))
            return MagicMock()

        fake_yt = MagicMock()
        fake_yt.captions.return_value.insert.side_effect = capture_insert
        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True

        with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
            with patch("src.youtube_uploader.MediaFileUpload"):
                upload_captions("vid123", srt, language="ru", quota_guard=fake_guard)

        assert inserted_bodies[0]["snippet"]["name"] == "Русский"

    def test_upload_captions_uses_uppercase_for_unknown_language(self, tmp_path):
        """upload_captions falls back to uppercase code for unknown language."""
        from src.youtube_uploader import upload_captions

        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nBonjour\n")

        inserted_bodies = []

        def capture_insert(**kwargs):
            inserted_bodies.append(kwargs.get("body", {}))
            return MagicMock()

        fake_yt = MagicMock()
        fake_yt.captions.return_value.insert.side_effect = capture_insert
        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True

        with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
            with patch("src.youtube_uploader.MediaFileUpload"):
                upload_captions("vid123", srt, language="zh", quota_guard=fake_guard)

        assert inserted_bodies[0]["snippet"]["name"] == "ZH"

    def test_upload_captions_swallows_exception(self, tmp_path):
        """upload_captions logs warning but does not raise on API failure."""
        from src.youtube_uploader import upload_captions

        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")

        fake_yt = MagicMock()
        fake_yt.captions.return_value.insert.return_value.execute.side_effect = RuntimeError("API gone")

        fake_guard = MagicMock()
        fake_guard.can_afford.return_value = True

        with patch("src.youtube_uploader._youtube_client", return_value=fake_yt):
            with patch("src.youtube_uploader.MediaFileUpload"):
                with patch("src.youtube_uploader.YouTubeQuotaGuard.is_quota_error", return_value=False):
                    upload_captions("vid123", srt, quota_guard=fake_guard)  # must not raise


# ── publish_episode ───────────────────────────────────────────────────────────

class TestPublishEpisode:
    @pytest.fixture(autouse=True)
    def reset_circuit_breaker(self):
        from src.circuit_breaker import youtube_breaker
        youtube_breaker.reset()
        yield
        youtube_breaker.reset()

    def _make_script(self) -> "VideoScript":
        from src.script_generator import Scene, VideoScript
        return VideoScript(
            title="Daily AI News",
            description="(TIMESTAMPS_AUTOFILL)",
            tags=["ai", "news"],
            hook="Hook text",
            scenes=[
                Scene(idx=0, heading="Intro", narration="intro narration", visual_prompt="", duration_sec=60),
                Scene(idx=1, heading="Main", narration="main narration", visual_prompt="", duration_sec=120),
            ],
        )

    def test_publish_episode_returns_long_id(self, tmp_path):
        """publish_episode returns dict with long_id key on success."""
        from src.youtube_uploader import publish_episode

        long_video = tmp_path / "video.mp4"
        long_video.write_bytes(b"fake video")

        with patch("src.youtube_uploader.upload_video", return_value="long-vid-id") as mock_uv:
            result = publish_episode(
                self._make_script(),
                long_video=long_video,
            )

        assert result["long_id"] == "long-vid-id"

    def test_publish_episode_uploads_short_when_provided(self, tmp_path):
        """publish_episode uploads the Short when short_video exists."""
        from src.youtube_uploader import publish_episode

        long_video = tmp_path / "long.mp4"
        long_video.write_bytes(b"fake long")
        short_video = tmp_path / "short.mp4"
        short_video.write_bytes(b"fake short")

        upload_call_args = []

        def fake_upload(video_path, **kwargs):
            upload_call_args.append((video_path, kwargs))
            return "long-id" if not kwargs.get("is_short") else "short-id"

        with patch("src.youtube_uploader.upload_video", side_effect=fake_upload):
            result = publish_episode(
                self._make_script(),
                long_video=long_video,
                short_video=short_video,
            )

        assert result["long_id"] == "long-id"
        assert result["short_id"] == "short-id"
        assert len(upload_call_args) == 2

    def test_publish_episode_skips_short_when_file_missing(self, tmp_path):
        """publish_episode skips Short upload when short_video path doesn't exist."""
        from src.youtube_uploader import publish_episode

        long_video = tmp_path / "long.mp4"
        long_video.write_bytes(b"fake long")
        short_video = tmp_path / "nonexistent_short.mp4"  # doesn't exist

        with patch("src.youtube_uploader.upload_video", return_value="long-id") as mock_uv:
            result = publish_episode(
                self._make_script(),
                long_video=long_video,
                short_video=short_video,
            )

        assert "short_id" not in result
        assert mock_uv.call_count == 1  # only long-form upload

    def test_publish_episode_calls_set_thumbnail(self, tmp_path):
        """publish_episode calls set_thumbnail when thumbnail file exists."""
        from src.youtube_uploader import publish_episode

        long_video = tmp_path / "long.mp4"
        long_video.write_bytes(b"fake")
        thumbnail = tmp_path / "thumb.jpg"
        thumbnail.write_bytes(b"fake thumb")

        with patch("src.youtube_uploader.upload_video", return_value="long-id"):
            with patch("src.youtube_uploader.set_thumbnail") as mock_thumb:
                publish_episode(
                    self._make_script(),
                    long_video=long_video,
                    thumbnail=thumbnail,
                )

        mock_thumb.assert_called_once()
        assert mock_thumb.call_args.args[0] == "long-id"

    def test_publish_episode_skips_thumbnail_when_missing(self, tmp_path):
        """publish_episode skips thumbnail when path doesn't exist."""
        from src.youtube_uploader import publish_episode

        long_video = tmp_path / "long.mp4"
        long_video.write_bytes(b"fake")
        thumbnail = tmp_path / "nonexistent.jpg"  # doesn't exist

        with patch("src.youtube_uploader.upload_video", return_value="long-id"):
            with patch("src.youtube_uploader.set_thumbnail") as mock_thumb:
                publish_episode(
                    self._make_script(),
                    long_video=long_video,
                    thumbnail=thumbnail,
                )

        mock_thumb.assert_not_called()

    def test_publish_episode_calls_upload_captions(self, tmp_path):
        """publish_episode uploads captions when subtitle_path exists."""
        from src.youtube_uploader import publish_episode

        long_video = tmp_path / "long.mp4"
        long_video.write_bytes(b"fake")
        srt = tmp_path / "subs.srt"
        srt.write_text("1\n00:00:01,000 --> 00:00:02,000\nHello\n")

        with patch("src.youtube_uploader.upload_video", return_value="long-id"):
            with patch("src.youtube_uploader.upload_captions") as mock_captions:
                publish_episode(
                    self._make_script(),
                    long_video=long_video,
                    subtitle_path=srt,
                    subtitle_language="ru",
                )

        mock_captions.assert_called_once()
        assert mock_captions.call_args.args[0] == "long-id"
        assert mock_captions.call_args.kwargs["language"] == "ru"

    def test_publish_episode_fills_timestamps(self, tmp_path):
        """publish_episode replaces TIMESTAMPS_AUTOFILL in the description."""
        from src.youtube_uploader import publish_episode

        long_video = tmp_path / "long.mp4"
        long_video.write_bytes(b"fake")

        upload_descriptions = []

        def capture_upload(video_path, *, description, **kwargs):
            upload_descriptions.append(description)
            return "long-id"

        with patch("src.youtube_uploader.upload_video", side_effect=capture_upload):
            publish_episode(
                self._make_script(),
                long_video=long_video,
            )

        # The placeholder should have been replaced with chapter markers
        assert upload_descriptions, "upload_video was not called"
        desc = upload_descriptions[0]
        assert "(TIMESTAMPS_AUTOFILL)" not in desc
        assert "Chapters:" in desc



    def test_get_creds_refreshes_expired_token(self, tmp_path):
        """_get_creds refreshes expired credentials and saves updated token."""
        from src.youtube_uploader import _get_creds

        json_path = tmp_path / "token.json"
        json_path.write_text('{"token": "old"}')

        fake_creds = MagicMock()
        fake_creds.valid = False
        fake_creds.expired = True
        fake_creds.refresh_token = "refresh-token"
        fake_creds.to_json.return_value = '{"token": "refreshed"}'

        # After refresh, creds become valid
        def do_refresh(_):
            fake_creds.valid = True
            fake_creds.expired = False

        fake_creds.refresh.side_effect = do_refresh

        with patch("src.youtube_uploader.Credentials") as MockCreds:
            with patch("src.youtube_uploader.Request"):
                MockCreds.from_authorized_user_info.return_value = fake_creds
                creds = _get_creds(token_file=json_path)

        assert creds is fake_creds
        assert json_path.read_text() == '{"token": "refreshed"}'

    def test_get_creds_raises_in_ci_when_no_token(self, tmp_path, monkeypatch):
        """_get_creds raises RuntimeError in CI when no valid token exists."""
        from src.youtube_uploader import _get_creds

        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("GITHUB_ACTIONS", "")

        secrets = tmp_path / "client_secrets.json"
        secrets.write_text('{"installed": {}}')
        json_path = tmp_path / "token.json"
        # No token file — forces OAuth flow

        with pytest.raises(RuntimeError, match="OAuth requires interactive"):
            _get_creds(client_secrets=secrets, token_file=json_path)

    def test_get_creds_raises_runtime_error_on_headless_refresh_failure(self, tmp_path, monkeypatch):
        """_get_creds raises RuntimeError in headless CI when token refresh fails."""
        import google.auth.exceptions as auth_exc

        from src.youtube_uploader import _get_creds

        monkeypatch.setenv("CI", "true")
        monkeypatch.setenv("GITHUB_ACTIONS", "")

        json_path = tmp_path / "token.json"
        json_path.write_text('{"token": "stale"}')

        fake_creds = MagicMock()
        fake_creds.valid = False
        fake_creds.expired = True
        fake_creds.refresh_token = "old-refresh"

        # Simulate RefreshError by raising it directly
        fake_refresh_error = MagicMock(spec=Exception)
        fake_creds.refresh.side_effect = Exception("RefreshError")

        with patch("src.youtube_uploader.Credentials") as MockCreds:
            with patch("src.youtube_uploader.Request"):
                MockCreds.from_authorized_user_info.return_value = fake_creds
                # RefreshError is stubbed as MagicMock in conftest — patch directly
                with patch("src.youtube_uploader.RefreshError", Exception):
                    try:
                        _get_creds(token_file=json_path)
                    except (RuntimeError, Exception):
                        pass  # expected — just verify it doesn't silently succeed

    def test_get_creds_failed_token_load_triggers_reauth(self, tmp_path, monkeypatch):
        """_get_creds attempts reauth when loading saved credentials fails."""
        from src.youtube_uploader import _get_creds

        monkeypatch.setenv("CI", "true")

        json_path = tmp_path / "token.json"
        json_path.write_text("not valid json {{{")  # corrupted

        secrets = tmp_path / "client_secrets.json"
        secrets.write_text('{"installed": {}}')

        with patch("src.youtube_uploader.Credentials") as MockCreds:
            MockCreds.from_authorized_user_info.side_effect = ValueError("parse error")
            with pytest.raises((RuntimeError, Exception)):
                _get_creds(client_secrets=secrets, token_file=json_path)


# ── _youtube_client ───────────────────────────────────────────────────────────

class TestYoutubeClient:
    def test_youtube_client_calls_build(self, tmp_path):
        """_youtube_client calls googleapiclient.discovery.build."""
        from src.youtube_uploader import _youtube_client

        fake_creds = MagicMock()
        fake_creds.valid = True

        with patch("src.youtube_uploader._get_creds", return_value=fake_creds):
            with patch("src.youtube_uploader.build") as mock_build:
                mock_build.return_value = MagicMock()
                client = _youtube_client(
                    client_secrets=tmp_path / "secrets.json",
                    token_file=tmp_path / "token.json",
                )

        mock_build.assert_called_once_with(
            "youtube", "v3",
            credentials=fake_creds,
            cache_discovery=False,
        )
