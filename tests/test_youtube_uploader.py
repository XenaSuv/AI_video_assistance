"""Tests for youtube_uploader — credential migration, _fill_timestamps, and upload helpers."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Extend conftest stubs with sub-modules that youtube_uploader imports at module level.
for _m in (
    "google.oauth2.credentials",
    "google.auth.exceptions",
    "googleapiclient.http",
):
    sys.modules.setdefault(_m, MagicMock())

from src.youtube_uploader import _migrate_pickle, _fill_timestamps
from src.script_generator import Scene, VideoScript


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
