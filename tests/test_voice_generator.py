"""Tests for voice_generator — SSML conversion, retry predicate, scene synthesis,
hook synthesis, and the async/sync script synthesis entry points."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock, call

# Stub elevenlabs before import
_elev_mock = MagicMock()
sys.modules.setdefault("elevenlabs", _elev_mock)
sys.modules.setdefault("elevenlabs.client", _elev_mock)

# Stub tenacity before import
_tenacity = MagicMock()
_tenacity.retry = lambda **kw: (lambda f: f)   # passthrough decorator
_tenacity.retry_if_exception = lambda f: f
_tenacity.stop_after_attempt = lambda n: n
_tenacity.wait_exponential = lambda **kw: None
sys.modules.setdefault("tenacity", _tenacity)

import pytest

from src.voice_generator import annotated_to_ssml, _is_retryable_elevenlabs


# ── annotated_to_ssml ─────────────────────────────────────────────────────────

class TestAnnotatedToSsml:
    def test_wraps_in_speak_tag(self):
        result = annotated_to_ssml("Hello world")
        assert result.startswith("<speak>")
        assert result.endswith("</speak>")

    def test_pause_short_replaced(self):
        result = annotated_to_ssml("Wait[PAUSE_SHORT]now")
        assert '<break time="300ms"/>' in result
        assert "[PAUSE_SHORT]" not in result

    def test_pause_long_replaced(self):
        result = annotated_to_ssml("Think[PAUSE_LONG]deeply")
        assert '<break time="700ms"/>' in result
        assert "[PAUSE_LONG]" not in result

    def test_emphasis_tags_replaced(self):
        result = annotated_to_ssml("[EMPHASIS]important[/EMPHASIS]")
        assert '<emphasis level="strong">' in result
        assert "</emphasis>" in result
        assert "[EMPHASIS]" not in result
        assert "[/EMPHASIS]" not in result

    def test_multiple_annotations_in_sequence(self):
        text = "[EMPHASIS]AI[/EMPHASIS][PAUSE_SHORT]is here[PAUSE_LONG]"
        result = annotated_to_ssml(text)
        assert '<emphasis level="strong">' in result
        assert '<break time="300ms"/>' in result
        assert '<break time="700ms"/>' in result

    def test_plain_text_unchanged_inside_speak(self):
        result = annotated_to_ssml("No annotations here")
        assert result == "<speak>No annotations here</speak>"

    def test_empty_string(self):
        result = annotated_to_ssml("")
        assert result == "<speak></speak>"


# ── _is_retryable_elevenlabs ──────────────────────────────────────────────────

class TestIsRetryableElevenlabs:
    def test_429_is_retryable(self):
        exc = Exception("rate limit")
        exc.status_code = 429
        assert _is_retryable_elevenlabs(exc)

    def test_500_is_retryable(self):
        exc = Exception("server error")
        exc.status_code = 500
        assert _is_retryable_elevenlabs(exc)

    def test_503_is_retryable(self):
        exc = Exception("service unavailable")
        exc.status_code = 503
        assert _is_retryable_elevenlabs(exc)

    def test_400_is_not_retryable(self):
        exc = Exception("bad request")
        exc.status_code = 400
        assert not _is_retryable_elevenlabs(exc)

    def test_404_is_not_retryable(self):
        exc = Exception("not found")
        exc.status_code = 404
        assert not _is_retryable_elevenlabs(exc)

    def test_connection_error_is_retryable(self):
        assert _is_retryable_elevenlabs(ConnectionError("refused"))

    def test_timeout_error_is_retryable(self):
        assert _is_retryable_elevenlabs(TimeoutError("timed out"))

    def test_os_error_is_retryable(self):
        assert _is_retryable_elevenlabs(OSError("broken pipe"))

    def test_generic_exception_without_status_code_not_retryable(self):
        exc = Exception("unexpected")
        assert not _is_retryable_elevenlabs(exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_scene(idx: int, narration: str, heading: str = "", duration_sec: int = 0):
    from src.script_generator import Scene
    return Scene(
        idx=idx,
        heading=heading or f"Scene {idx}",
        narration=narration,
        visual_prompt="",
        duration_sec=duration_sec,
    )


def _make_script(scenes=None):
    from src.script_generator import VideoScript
    if scenes is None:
        scenes = [_make_scene(0, "Hello world")]
    return VideoScript(
        title="Test Script",
        description="desc",
        tags=["ai"],
        hook="This is the hook",
        scenes=scenes,
    )


# ── _synthesize_one_scene ─────────────────────────────────────────────────────

class TestSynthesizeOneScene:
    def test_writes_audio_file_and_sets_duration(self, tmp_path):
        from src.voice_generator import _synthesize_one_scene
        scene = _make_scene(1, "This is plain narration without any SSML tags.")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        fake_chunks = [b"audio_chunk_1", b"audio_chunk_2"]
        mock_client = MagicMock()

        with patch("src.voice_generator._tts_convert", return_value=iter(fake_chunks)):
            with patch("src.voice_generator.ff_duration", return_value=15.0):
                with patch("src.voice_generator.get_ledger") as mock_ledger:
                    mock_ledger.return_value = MagicMock()
                    result = _synthesize_one_scene(
                        scene, audio_dir, "voice_id_x", "model_id_x", mock_client
                    )

        assert result == audio_dir / "scene_01.mp3"
        assert result.exists()
        assert result.read_bytes() == b"audio_chunk_1audio_chunk_2"
        assert scene.duration_sec == 16  # int(15.0) + 1

    def test_uses_ssml_when_pause_tags_present(self, tmp_path):
        from src.voice_generator import _synthesize_one_scene
        scene = _make_scene(2, "Wait[PAUSE_SHORT]here comes something important.")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        captured_calls = []

        def capture_tts(client, text, v_id, m_id, use_ssml=False):
            captured_calls.append({"text": text, "use_ssml": use_ssml})
            return iter([b"data"])

        with patch("src.voice_generator._tts_convert", side_effect=capture_tts):
            with patch("src.voice_generator.ff_duration", return_value=10.0):
                with patch("src.voice_generator.get_ledger") as mock_ledger:
                    mock_ledger.return_value = MagicMock()
                    _synthesize_one_scene(scene, audio_dir, "v", "m", MagicMock())

        assert len(captured_calls) == 1
        assert captured_calls[0]["use_ssml"] is True
        assert "<speak>" in captured_calls[0]["text"]

    def test_skips_empty_chunks(self, tmp_path):
        """Empty byte chunks should not be written to the file."""
        from src.voice_generator import _synthesize_one_scene
        scene = _make_scene(3, "Some narration here.")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        fake_chunks = [b"real_data", b"", b"more_data", b""]
        with patch("src.voice_generator._tts_convert", return_value=iter(fake_chunks)):
            with patch("src.voice_generator.ff_duration", return_value=5.0):
                with patch("src.voice_generator.get_ledger") as mock_ledger:
                    mock_ledger.return_value = MagicMock()
                    result = _synthesize_one_scene(scene, audio_dir, "v", "m", MagicMock())

        assert result.read_bytes() == b"real_datamore_data"

    def test_plain_narration_does_not_use_ssml(self, tmp_path):
        from src.voice_generator import _synthesize_one_scene
        scene = _make_scene(0, "Just plain text with no special tags.")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        captured = []
        def capture_tts(client, text, v_id, m_id, use_ssml=False):
            captured.append(use_ssml)
            return iter([b"bytes"])

        with patch("src.voice_generator._tts_convert", side_effect=capture_tts):
            with patch("src.voice_generator.ff_duration", return_value=5.0):
                with patch("src.voice_generator.get_ledger") as mock_ledger:
                    mock_ledger.return_value = MagicMock()
                    _synthesize_one_scene(scene, audio_dir, "v", "m", MagicMock())

        assert captured[0] is False


# ── synthesize_hook ───────────────────────────────────────────────────────────

class TestSynthesizeHook:
    def test_writes_hook_mp3(self, tmp_path):
        from src.voice_generator import synthesize_hook
        script = _make_script()
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        fake_chunks = [b"hook_audio"]
        with patch("src.voice_generator.ElevenLabs") as MockElevenLabs:
            mock_client = MagicMock()
            MockElevenLabs.return_value = mock_client
            with patch("src.voice_generator._tts_convert", return_value=iter(fake_chunks)):
                with patch("src.voice_generator.ff_duration", return_value=3.0):
                    with patch("src.voice_generator.get_ledger") as mock_ledger:
                        mock_ledger.return_value = MagicMock()
                        result = synthesize_hook(script, tmp_path)

        assert result.name == "hook.mp3"
        assert result.exists()
        assert result.read_bytes() == b"hook_audio"

    def test_raises_if_hook_text_empty(self, tmp_path):
        from src.voice_generator import synthesize_hook
        script = _make_script()
        script.hook = "   "  # whitespace only
        with pytest.raises(ValueError, match="no hook text"):
            synthesize_hook(script, tmp_path)

    def test_returns_cached_hook_without_synthesis(self, tmp_path):
        from src.voice_generator import synthesize_hook
        script = _make_script()
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()
        hook_path = audio_dir / "hook.mp3"
        hook_path.write_bytes(b"cached")

        with patch("src.voice_generator.ElevenLabs") as MockElevenLabs:
            with patch("src.voice_generator._tts_convert") as mock_tts:
                result = synthesize_hook(script, tmp_path)

        mock_tts.assert_not_called()
        assert result == hook_path

    def test_uses_provided_voice_id(self, tmp_path):
        from src.voice_generator import synthesize_hook
        script = _make_script()
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        captured = []
        def capture_tts(client, text, v_id, m_id, use_ssml=False):
            captured.append({"v_id": v_id, "m_id": m_id})
            return iter([b"audio"])

        with patch("src.voice_generator.ElevenLabs"):
            with patch("src.voice_generator._tts_convert", side_effect=capture_tts):
                with patch("src.voice_generator.ff_duration", return_value=2.0):
                    with patch("src.voice_generator.get_ledger") as mock_ledger:
                        mock_ledger.return_value = MagicMock()
                        synthesize_hook(script, tmp_path, voice_id="custom_voice", model_id="custom_model")

        assert captured[0]["v_id"] == "custom_voice"
        assert captured[0]["m_id"] == "custom_model"


# ── async_synthesize_script ───────────────────────────────────────────────────

class TestAsyncSynthesizeScript:
    def test_synthesizes_all_scenes(self, tmp_path):
        """async_synthesize_script returns one path per scene, sorted."""
        from src.voice_generator import async_synthesize_script
        scenes = [
            _make_scene(0, "Scene zero narration"),
            _make_scene(1, "Scene one narration"),
        ]
        script = _make_script(scenes)

        def fake_synthesize(scene, audio_dir, v_id, m_id, client):
            p = audio_dir / f"scene_{scene.idx:02d}.mp3"
            p.write_bytes(b"audio")
            scene.duration_sec = 10
            return p

        with patch("src.voice_generator.ElevenLabs"):
            with patch("src.voice_generator._synthesize_one_scene", side_effect=fake_synthesize):
                paths = asyncio.run(async_synthesize_script(script, tmp_path))

        assert len(paths) == 2
        assert paths[0].name == "scene_00.mp3"
        assert paths[1].name == "scene_01.mp3"

    def test_audio_subdir_created(self, tmp_path):
        from src.voice_generator import async_synthesize_script
        scenes = [_make_scene(0, "narration")]
        script = _make_script(scenes)

        def fake_synth(scene, audio_dir, v_id, m_id, client):
            p = audio_dir / f"scene_{scene.idx:02d}.mp3"
            p.write_bytes(b"x")
            scene.duration_sec = 5
            return p

        with patch("src.voice_generator.ElevenLabs"):
            with patch("src.voice_generator._synthesize_one_scene", side_effect=fake_synth):
                asyncio.run(async_synthesize_script(script, tmp_path, audio_subdir="audio_ru"))

        assert (tmp_path / "audio_ru").is_dir()

    def test_uses_custom_voice_and_model(self, tmp_path):
        from src.voice_generator import async_synthesize_script
        scenes = [_make_scene(0, "narration")]
        script = _make_script(scenes)

        captured = []

        def fake_synth(scene, audio_dir, v_id, m_id, client):
            captured.append({"v_id": v_id, "m_id": m_id})
            p = audio_dir / f"scene_{scene.idx:02d}.mp3"
            p.write_bytes(b"x")
            scene.duration_sec = 5
            return p

        with patch("src.voice_generator.ElevenLabs"):
            with patch("src.voice_generator._synthesize_one_scene", side_effect=fake_synth):
                asyncio.run(
                    async_synthesize_script(
                        script, tmp_path,
                        voice_id="custom_v",
                        model_id="custom_m",
                    )
                )

        assert captured[0]["v_id"] == "custom_v"
        assert captured[0]["m_id"] == "custom_m"


# ── synthesize_script (sync wrapper) ─────────────────────────────────────────

class TestSynthesizeScript:
    def test_sync_wrapper_returns_paths(self, tmp_path):
        from src.voice_generator import synthesize_script
        scenes = [_make_scene(0, "narration")]
        script = _make_script(scenes)

        def fake_synth(scene, audio_dir, v_id, m_id, client):
            p = audio_dir / f"scene_{scene.idx:02d}.mp3"
            p.write_bytes(b"audio")
            scene.duration_sec = 8
            return p

        with patch("src.voice_generator.ElevenLabs"):
            with patch("src.voice_generator._synthesize_one_scene", side_effect=fake_synth):
                paths = synthesize_script(script, tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "scene_00.mp3"
