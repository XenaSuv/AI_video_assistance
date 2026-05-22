"""Tests for voice_generator — SSML conversion, retry predicate, scene synthesis,
hook synthesis, async TTS retry/timeout, and the async/sync script entry points."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Stub elevenlabs before import
_elev_mock = MagicMock()
sys.modules.setdefault("elevenlabs", _elev_mock)
sys.modules.setdefault("elevenlabs.client", _elev_mock)

import pytest

from src.voice_generator import _is_retryable_elevenlabs, annotated_to_ssml

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
        exc.status_code = 429  # type: ignore[attr-defined]
        assert _is_retryable_elevenlabs(exc)

    def test_500_is_retryable(self):
        exc = Exception("server error")
        exc.status_code = 500  # type: ignore[attr-defined]
        assert _is_retryable_elevenlabs(exc)

    def test_503_is_retryable(self):
        exc = Exception("service unavailable")
        exc.status_code = 503  # type: ignore[attr-defined]
        assert _is_retryable_elevenlabs(exc)

    def test_400_is_not_retryable(self):
        exc = Exception("bad request")
        exc.status_code = 400  # type: ignore[attr-defined]
        assert not _is_retryable_elevenlabs(exc)

    def test_404_is_not_retryable(self):
        exc = Exception("not found")
        exc.status_code = 404  # type: ignore[attr-defined]
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


# ── _tts_convert_async ────────────────────────────────────────────────────────

class TestTtsConvertAsync:
    def test_returns_bytes_on_first_success(self):
        from src.voice_generator import _tts_convert_async
        mock_client = MagicMock()
        with patch("src.voice_generator.elevenlabs_breaker") as mock_breaker:
            mock_breaker.call.return_value = b"audio_data"
            result = asyncio.run(_tts_convert_async(mock_client, "text", "v", "m"))
        assert result == b"audio_data"
        assert mock_breaker.call.call_count == 1

    def test_empty_chunks_joined_and_filtered(self):
        """_blocking_call skips empty byte chunks before returning."""
        from src.voice_generator import _tts_convert_async
        mock_client = MagicMock()
        mock_client.text_to_speech.convert.return_value = [b"a", b"", b"b", b""]
        with patch("src.voice_generator.elevenlabs_breaker") as mock_breaker:
            mock_breaker.call.side_effect = lambda f: f()
            result = asyncio.run(_tts_convert_async(mock_client, "text", "v", "m"))
        assert result == b"ab"

    def test_retryable_error_retries_and_succeeds(self):
        from src.voice_generator import _tts_convert_async
        mock_client = MagicMock()
        calls = 0

        def side_eff(func: object) -> bytes:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise ConnectionError("network blip")
            return b"ok"

        with patch("src.voice_generator.elevenlabs_breaker") as mock_breaker:
            mock_breaker.call.side_effect = side_eff
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = asyncio.run(_tts_convert_async(mock_client, "text", "v", "m"))

        assert result == b"ok"
        assert calls == 2

    def test_non_retryable_raises_immediately(self):
        from src.voice_generator import _tts_convert_async
        mock_client = MagicMock()
        bad_exc: Exception = Exception("bad request")
        bad_exc.status_code = 400  # type: ignore[attr-defined]
        with patch("src.voice_generator.elevenlabs_breaker") as mock_breaker:
            mock_breaker.call.side_effect = bad_exc
            with pytest.raises(Exception, match="bad request"):
                asyncio.run(_tts_convert_async(mock_client, "text", "v", "m"))
        assert mock_breaker.call.call_count == 1

    def test_exhausted_attempts_raises_runtime_error(self):
        from src.voice_generator import _tts_convert_async, _TTS_MAX_ATTEMPTS
        mock_client = MagicMock()
        with patch("src.voice_generator.elevenlabs_breaker") as mock_breaker:
            mock_breaker.call.side_effect = ConnectionError("persistent")
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RuntimeError, match=f"after {_TTS_MAX_ATTEMPTS} attempts"):
                    asyncio.run(_tts_convert_async(mock_client, "text", "v", "m"))
        assert mock_breaker.call.call_count == _TTS_MAX_ATTEMPTS

    def test_timeout_is_retried_and_succeeds(self):
        """asyncio.TimeoutError from wait_for is treated as retryable."""
        from src.voice_generator import _tts_convert_async
        mock_client = MagicMock()
        call_n = 0
        original_wait_for = asyncio.wait_for

        async def patched_wait_for(aw: object, timeout: float | None = None, **kwargs: object) -> bytes:
            nonlocal call_n
            call_n += 1
            if call_n == 1:
                # Cancel the coroutine/future to avoid ResourceWarning
                if hasattr(aw, "close"):
                    aw.close()  # type: ignore[union-attr]
                raise asyncio.TimeoutError()
            return await original_wait_for(aw, timeout=timeout)  # type: ignore[arg-type]

        with patch("src.voice_generator.elevenlabs_breaker") as mock_breaker:
            mock_breaker.call.return_value = b"after_timeout"
            with patch("asyncio.wait_for", patched_wait_for):
                with patch("asyncio.sleep", new_callable=AsyncMock):
                    result = asyncio.run(_tts_convert_async(mock_client, "text", "v", "m"))

        assert result == b"after_timeout"
        assert call_n == 2

    def test_exponential_backoff_between_retries(self):
        """Sleep delay doubles on each retry, capped at _TTS_BACKOFF_MAX."""
        from src.voice_generator import (
            _tts_convert_async, _TTS_BACKOFF_BASE, _TTS_BACKOFF_MAX, _TTS_MAX_ATTEMPTS,
        )
        mock_client = MagicMock()
        sleep_calls: list[float] = []

        async def record_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("src.voice_generator.elevenlabs_breaker") as mock_breaker:
            mock_breaker.call.side_effect = ConnectionError("keep failing")
            with patch("asyncio.sleep", side_effect=record_sleep):
                with pytest.raises(RuntimeError):
                    asyncio.run(_tts_convert_async(mock_client, "text", "v", "m"))

        assert len(sleep_calls) == _TTS_MAX_ATTEMPTS - 1
        # First delay is _TTS_BACKOFF_BASE, each subsequent doubles (capped)
        expected_delay = _TTS_BACKOFF_BASE
        for actual in sleep_calls:
            assert actual == min(expected_delay, _TTS_BACKOFF_MAX)
            expected_delay *= 2.0


# ── _synthesize_one_scene ─────────────────────────────────────────────────────

class TestSynthesizeOneScene:
    def test_writes_audio_file_and_sets_duration(self, tmp_path):
        from src.voice_generator import _synthesize_one_scene
        scene = _make_scene(1, "This is plain narration without any SSML tags.")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with patch("src.voice_generator._tts_convert_async", new_callable=AsyncMock,
                   return_value=b"audio_chunk_1audio_chunk_2"):
            with patch("src.voice_generator.ff_duration", return_value=15.0):
                with patch("src.voice_generator.get_ledger") as mock_ledger:
                    mock_ledger.return_value = MagicMock()
                    result = asyncio.run(
                        _synthesize_one_scene(scene, audio_dir, "voice_id_x", "model_id_x", MagicMock())
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

        captured_text: list[str] = []

        async def capture_tts(client: object, text: str, v_id: str, m_id: str) -> bytes:
            captured_text.append(text)
            return b"data"

        with patch("src.voice_generator._tts_convert_async", side_effect=capture_tts):
            with patch("src.voice_generator.ff_duration", return_value=10.0):
                with patch("src.voice_generator.get_ledger") as mock_ledger:
                    mock_ledger.return_value = MagicMock()
                    asyncio.run(_synthesize_one_scene(scene, audio_dir, "v", "m", MagicMock()))

        assert len(captured_text) == 1
        assert "<speak>" in captured_text[0]

    def test_plain_narration_does_not_use_ssml(self, tmp_path):
        from src.voice_generator import _synthesize_one_scene
        scene = _make_scene(0, "Just plain text with no special tags.")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        captured_text: list[str] = []

        async def capture_tts(client: object, text: str, v_id: str, m_id: str) -> bytes:
            captured_text.append(text)
            return b"bytes"

        with patch("src.voice_generator._tts_convert_async", side_effect=capture_tts):
            with patch("src.voice_generator.ff_duration", return_value=5.0):
                with patch("src.voice_generator.get_ledger") as mock_ledger:
                    mock_ledger.return_value = MagicMock()
                    asyncio.run(_synthesize_one_scene(scene, audio_dir, "v", "m", MagicMock()))

        assert "<speak>" not in captured_text[0]

    def test_file_written_with_returned_bytes(self, tmp_path):
        from src.voice_generator import _synthesize_one_scene
        scene = _make_scene(3, "Some narration here.")
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with patch("src.voice_generator._tts_convert_async", new_callable=AsyncMock,
                   return_value=b"real_datamore_data"):
            with patch("src.voice_generator.ff_duration", return_value=5.0):
                with patch("src.voice_generator.get_ledger") as mock_ledger:
                    mock_ledger.return_value = MagicMock()
                    result = asyncio.run(
                        _synthesize_one_scene(scene, audio_dir, "v", "m", MagicMock())
                    )

        assert result.read_bytes() == b"real_datamore_data"


# ── synthesize_hook ───────────────────────────────────────────────────────────

class TestSynthesizeHook:
    def test_writes_hook_mp3(self, tmp_path):
        from src.voice_generator import synthesize_hook
        script = _make_script()
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        with patch("src.voice_generator.ElevenLabs"):
            with patch("src.voice_generator._tts_convert_async", new_callable=AsyncMock,
                       return_value=b"hook_audio"):
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

        with patch("src.voice_generator.ElevenLabs"):
            with patch("src.voice_generator._tts_convert_async", new_callable=AsyncMock) as mock_tts:
                result = synthesize_hook(script, tmp_path)

        mock_tts.assert_not_called()
        assert result == hook_path

    def test_uses_provided_voice_id(self, tmp_path):
        from src.voice_generator import synthesize_hook
        script = _make_script()
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        captured: list[dict[str, str]] = []

        async def capture_tts(client: object, text: str, v_id: str, m_id: str) -> bytes:
            captured.append({"v_id": v_id, "m_id": m_id})
            return b"audio"

        with patch("src.voice_generator.ElevenLabs"):
            with patch("src.voice_generator._tts_convert_async", side_effect=capture_tts):
                with patch("src.voice_generator.ff_duration", return_value=2.0):
                    with patch("src.voice_generator.get_ledger") as mock_ledger:
                        mock_ledger.return_value = MagicMock()
                        synthesize_hook(
                            script, tmp_path,
                            voice_id="custom_voice",
                            model_id="custom_model",
                        )

        assert captured[0]["v_id"] == "custom_voice"
        assert captured[0]["m_id"] == "custom_model"


# ── async_synthesize_script ───────────────────────────────────────────────────

class TestAsyncSynthesizeScript:
    def test_synthesizes_all_scenes(self, tmp_path):
        from src.voice_generator import async_synthesize_script
        scenes = [
            _make_scene(0, "Scene zero narration"),
            _make_scene(1, "Scene one narration"),
        ]
        script = _make_script(scenes)

        async def fake_synthesize(
            scene: object, audio_dir: Path, v_id: str, m_id: str, client: object
        ) -> Path:
            assert isinstance(audio_dir, Path)
            p = audio_dir / f"scene_{scene.idx:02d}.mp3"  # type: ignore[union-attr]
            p.write_bytes(b"audio")
            scene.duration_sec = 10  # type: ignore[union-attr]
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

        async def fake_synth(
            scene: object, audio_dir: Path, v_id: str, m_id: str, client: object
        ) -> Path:
            p = audio_dir / f"scene_{scene.idx:02d}.mp3"  # type: ignore[union-attr]
            p.write_bytes(b"x")
            scene.duration_sec = 5  # type: ignore[union-attr]
            return p

        with patch("src.voice_generator.ElevenLabs"):
            with patch("src.voice_generator._synthesize_one_scene", side_effect=fake_synth):
                asyncio.run(async_synthesize_script(script, tmp_path, audio_subdir="audio_ru"))

        assert (tmp_path / "audio_ru").is_dir()

    def test_uses_custom_voice_and_model(self, tmp_path):
        from src.voice_generator import async_synthesize_script
        scenes = [_make_scene(0, "narration")]
        script = _make_script(scenes)

        captured: list[dict[str, str]] = []

        async def fake_synth(
            scene: object, audio_dir: Path, v_id: str, m_id: str, client: object
        ) -> Path:
            captured.append({"v_id": v_id, "m_id": m_id})
            p = audio_dir / f"scene_{scene.idx:02d}.mp3"  # type: ignore[union-attr]
            p.write_bytes(b"x")
            scene.duration_sec = 5  # type: ignore[union-attr]
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

        async def fake_synth(
            scene: object, audio_dir: Path, v_id: str, m_id: str, client: object
        ) -> Path:
            p = audio_dir / f"scene_{scene.idx:02d}.mp3"  # type: ignore[union-attr]
            p.write_bytes(b"audio")
            scene.duration_sec = 8  # type: ignore[union-attr]
            return p

        with patch("src.voice_generator.ElevenLabs"):
            with patch("src.voice_generator._synthesize_one_scene", side_effect=fake_synth):
                paths = synthesize_script(script, tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "scene_00.mp3"
