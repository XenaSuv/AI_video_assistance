"""Tests for AvatarEngine — intent-aware HeyGen avatar orchestration."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.avatar_engine import (
    AVATAR_INTENTS,
    AvatarEngine,
    _INTENT_STYLE,
    _PERSONA_AVATAR_ATTR,
)


# ── AVATAR_INTENTS ────────────────────────────────────────────────────────────

class TestAvatarIntents:
    def test_shock_is_avatar_intent(self):
        assert "shock" in AVATAR_INTENTS

    def test_twist_is_avatar_intent(self):
        assert "twist" in AVATAR_INTENTS

    def test_hook_is_avatar_intent(self):
        assert "hook" in AVATAR_INTENTS

    def test_conflict_is_avatar_intent(self):
        assert "conflict" in AVATAR_INTENTS

    def test_warning_is_avatar_intent(self):
        assert "warning" in AVATAR_INTENTS

    def test_explain_is_not_avatar_intent(self):
        assert "explain" not in AVATAR_INTENTS

    def test_data_is_not_avatar_intent(self):
        assert "data" not in AVATAR_INTENTS

    def test_curiosity_is_not_avatar_intent(self):
        assert "curiosity" not in AVATAR_INTENTS


# ── AvatarEngine.should_use_avatar ────────────────────────────────────────────

class TestShouldUseAvatar:
    def test_shock_uses_avatar(self):
        engine = AvatarEngine()
        assert engine.should_use_avatar("shock") is True

    def test_explain_does_not_use_avatar(self):
        engine = AvatarEngine()
        assert engine.should_use_avatar("explain") is False

    def test_unknown_intent_does_not_use_avatar(self):
        engine = AvatarEngine()
        assert engine.should_use_avatar("something_new") is False


# ── AvatarEngine.avatar_style ─────────────────────────────────────────────────

class TestAvatarStyle:
    def test_shock_returns_closeup(self):
        engine = AvatarEngine()
        assert engine.avatar_style("shock") == "closeup"

    def test_twist_returns_closeup(self):
        engine = AvatarEngine()
        assert engine.avatar_style("twist") == "closeup"

    def test_hook_returns_closeup(self):
        engine = AvatarEngine()
        assert engine.avatar_style("hook") == "closeup"

    def test_conflict_returns_closeup(self):
        engine = AvatarEngine()
        assert engine.avatar_style("conflict") == "closeup"

    def test_explain_returns_normal(self):
        engine = AvatarEngine()
        assert engine.avatar_style("explain") == "normal"

    def test_warning_returns_normal(self):
        engine = AvatarEngine()
        assert engine.avatar_style("warning") == "normal"

    def test_unknown_intent_returns_normal(self):
        engine = AvatarEngine()
        assert engine.avatar_style("unknown_xyz") == "normal"

    def test_all_intents_have_style(self):
        engine = AvatarEngine()
        for intent in ("shock", "twist", "hook", "conflict", "warning",
                       "explain", "curiosity", "data", "reaction", "mistake", "insider"):
            assert engine.avatar_style(intent) in ("normal", "closeup"), intent


# ── AvatarEngine.avatar_id ────────────────────────────────────────────────────

class TestAvatarId:
    def test_returns_default_when_no_env_override(self):
        engine = AvatarEngine()
        engine._default_id = "default_avatar"
        result = engine.avatar_id("direct")
        assert result == "default_avatar"

    def test_returns_persona_override_when_set(self):
        engine = AvatarEngine()
        engine._default_id = "default_avatar"
        attr = _PERSONA_AVATAR_ATTR.get("direct", "heygen_avatar_id_direct")
        with patch("src.avatar_engine.settings") as mock_settings:
            setattr(mock_settings, attr, "direct_avatar_id")
            result = engine.avatar_id("direct")
        assert result == "direct_avatar_id"

    def test_falls_back_to_default_when_setting_empty(self):
        engine = AvatarEngine()
        engine._default_id = "default_avatar"
        attr = _PERSONA_AVATAR_ATTR.get("serious", "heygen_avatar_id_serious")
        with patch("src.avatar_engine.settings") as mock_settings:
            setattr(mock_settings, attr, "")
            result = engine.avatar_id("serious")
        assert result == "default_avatar"

    def test_unknown_persona_returns_default(self):
        engine = AvatarEngine()
        engine._default_id = "default_avatar"
        assert engine.avatar_id("completely_unknown_persona") == "default_avatar"

    def test_all_defined_personas_have_attr_entry(self):
        for persona in ("direct", "serious", "curious", "professional", "casual"):
            assert persona in _PERSONA_AVATAR_ATTR, persona


# ── AvatarEngine.render — disabled ────────────────────────────────────────────

class TestRenderDisabled:
    def test_returns_none_when_disabled(self, tmp_path):
        engine = AvatarEngine()
        engine._enabled = False
        result = engine.render(
            tmp_path / "audio.mp3",
            tmp_path / "out.mp4",
            intent="shock",
        )
        assert result is None

    def test_render_for_scene_returns_none_when_disabled(self, tmp_path):
        from src.script_generator import Scene
        engine = AvatarEngine()
        engine._enabled = False
        scene = Scene(idx=0, heading="h", narration="n", visual_prompt="v", scene_intent="shock")
        result = engine.render_for_scene(scene, tmp_path / "audio.mp3", tmp_path)
        assert result is None


# ── AvatarEngine.render — enabled with mocked HeyGen ─────────────────────────

class TestRenderEnabled:
    def _engine_with_mock(self, mock_clip_path: Path) -> AvatarEngine:
        engine = AvatarEngine()
        engine._enabled    = True
        engine._api_key    = "fake-key"
        engine._default_id = "fake-avatar"
        engine._voice_id   = "fake-voice"
        return engine

    def test_returns_none_when_heygen_returns_none(self, tmp_path):
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        with patch("src.avatar_engine.generate_heygen_clip", return_value=None):
            result = engine.render(
                tmp_path / "audio.mp3",
                tmp_path / "out.mp4",
                intent="shock",
            )
        assert result is None

    def test_calls_heygen_with_correct_style(self, tmp_path):
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        raw_path = tmp_path / "out_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path) as mock_hg:
            with patch("src.avatar_engine.ffmpeg_utils.burn_captions", side_effect=lambda v, t, o, **kw: v):
                with patch("shutil.copy2"):
                    engine.render(
                        tmp_path / "audio.mp3",
                        tmp_path / "out.mp4",
                        intent="shock",
                        persona="direct",
                        subtitle_text="test text",
                    )

        call_kwargs = mock_hg.call_args.kwargs
        assert call_kwargs["avatar_style"] == "closeup"

    def test_casual_persona_passes_casual_avatar_id(self, tmp_path):
        """heygen_avatar_id_casual setting must reach generate_heygen_clip as avatar_id."""
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        raw_path = tmp_path / "out_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.settings") as mock_settings:
            mock_settings.heygen_avatar_id_casual = "casual-avatar-xyz"
            with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path) as mock_hg:
                with patch("src.avatar_engine.ffmpeg_utils.burn_captions", side_effect=lambda v, t, o, **kw: v):
                    with patch("shutil.copy2"):
                        engine.render(
                            tmp_path / "audio.mp3",
                            tmp_path / "out.mp4",
                            intent="shock",
                            persona="casual",
                            subtitle_text="text",
                        )

        assert mock_hg.call_args.kwargs["avatar_id"] == "casual-avatar-xyz"

    def test_default_avatar_id_used_when_casual_setting_empty(self, tmp_path):
        """Falls back to _default_id when heygen_avatar_id_casual is empty."""
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        engine._default_id = "default-fallback-id"
        raw_path = tmp_path / "out_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.settings") as mock_settings:
            mock_settings.heygen_avatar_id_casual = ""
            with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path) as mock_hg:
                with patch("src.avatar_engine.ffmpeg_utils.burn_captions", side_effect=lambda v, t, o, **kw: v):
                    with patch("shutil.copy2"):
                        engine.render(
                            tmp_path / "audio.mp3",
                            tmp_path / "out.mp4",
                            intent="shock",
                            persona="casual",
                            subtitle_text="text",
                        )

        assert mock_hg.call_args.kwargs["avatar_id"] == "default-fallback-id"

    @pytest.mark.parametrize("persona,attr,expected_id", [
        ("curious",      "heygen_avatar_id_curious",      "curious-avatar-abc"),
        ("direct",       "heygen_avatar_id_direct",       "direct-avatar-def"),
        ("professional", "heygen_avatar_id_professional", "professional-avatar-ghi"),
        ("serious",      "heygen_avatar_id_serious",      "serious-avatar-jkl"),
    ])
    def test_all_personas_pass_correct_avatar_id(self, tmp_path, persona, attr, expected_id):
        """Each persona settings field must reach generate_heygen_clip as avatar_id."""
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        raw_path = tmp_path / "out_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.settings") as mock_settings:
            setattr(mock_settings, attr, expected_id)
            with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path) as mock_hg:
                with patch("src.avatar_engine.ffmpeg_utils.burn_captions", side_effect=lambda v, t, o, **kw: v):
                    with patch("shutil.copy2"):
                        engine.render(
                            tmp_path / "audio.mp3",
                            tmp_path / "out.mp4",
                            intent="shock",
                            persona=persona,
                            subtitle_text="text",
                        )

        assert mock_hg.call_args.kwargs["avatar_id"] == expected_id

    @pytest.mark.parametrize("persona,attr", [
        ("curious",      "heygen_avatar_id_curious"),
        ("direct",       "heygen_avatar_id_direct"),
        ("professional", "heygen_avatar_id_professional"),
        ("serious",      "heygen_avatar_id_serious"),
    ])
    def test_all_personas_fall_back_to_default(self, tmp_path, persona, attr):
        """Falls back to _default_id when persona-specific setting is empty."""
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        engine._default_id = "fallback-id"
        raw_path = tmp_path / "out_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.settings") as mock_settings:
            setattr(mock_settings, attr, "")
            with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path) as mock_hg:
                with patch("src.avatar_engine.ffmpeg_utils.burn_captions", side_effect=lambda v, t, o, **kw: v):
                    with patch("shutil.copy2"):
                        engine.render(
                            tmp_path / "audio.mp3",
                            tmp_path / "out.mp4",
                            intent="shock",
                            persona=persona,
                            subtitle_text="text",
                        )

        assert mock_hg.call_args.kwargs["avatar_id"] == "fallback-id"

    def test_voice_id_flows_to_generate_heygen_clip(self, tmp_path):
        """HEYGEN_VOICE_ID (self._voice_id) must reach generate_heygen_clip as voice_id."""
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        engine._voice_id = "voice-from-env-xyz"
        raw_path = tmp_path / "out_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path) as mock_hg:
            with patch("src.avatar_engine.ffmpeg_utils.burn_captions", side_effect=lambda v, t, o, **kw: v):
                with patch("shutil.copy2"):
                    engine.render(
                        tmp_path / "audio.mp3",
                        tmp_path / "out.mp4",
                        intent="shock",
                        subtitle_text="text",
                    )

        assert mock_hg.call_args.kwargs["voice_id"] == "voice-from-env-xyz"

    def test_calls_heygen_with_normal_style_for_explain(self, tmp_path):
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        raw_path = tmp_path / "out_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path) as mock_hg:
            with patch("src.avatar_engine.ffmpeg_utils.burn_captions", side_effect=lambda v, t, o, **kw: v):
                with patch("shutil.copy2"):
                    engine.render(
                        tmp_path / "audio.mp3",
                        tmp_path / "out.mp4",
                        intent="explain",
                        subtitle_text="text",
                    )

        assert mock_hg.call_args.kwargs["avatar_style"] == "normal"

    def test_shorts_mode_calls_zoom_clip(self, tmp_path):
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        raw_path = tmp_path / "out_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path):
            with patch("src.avatar_engine.ffmpeg_utils.zoom_clip", return_value=raw_path) as mock_zoom:
                with patch("src.avatar_engine.ffmpeg_utils.burn_captions", side_effect=lambda v, t, o, **kw: v):
                    with patch("shutil.copy2"):
                        engine.render(
                            tmp_path / "audio.mp3",
                            tmp_path / "out.mp4",
                            is_shorts=True,
                            subtitle_text="text",
                        )
        mock_zoom.assert_called_once()

    def test_non_shorts_mode_does_not_zoom(self, tmp_path):
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        raw_path = tmp_path / "out_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path):
            with patch("src.avatar_engine.ffmpeg_utils.zoom_clip") as mock_zoom:
                with patch("src.avatar_engine.ffmpeg_utils.burn_captions", side_effect=lambda v, t, o, **kw: v):
                    with patch("shutil.copy2"):
                        engine.render(
                            tmp_path / "audio.mp3",
                            tmp_path / "out.mp4",
                            is_shorts=False,
                            subtitle_text="text",
                        )
        mock_zoom.assert_not_called()

    def test_no_subtitle_skips_burn_captions(self, tmp_path):
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        raw_path = tmp_path / "out_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path):
            with patch("src.avatar_engine.ffmpeg_utils.burn_captions") as mock_captions:
                with patch("shutil.copy2"):
                    engine.render(
                        tmp_path / "audio.mp3",
                        tmp_path / "out.mp4",
                        subtitle_text="",
                    )
        mock_captions.assert_not_called()

    def test_render_for_scene_passes_intent_from_scene(self, tmp_path):
        from src.script_generator import Scene
        engine = self._engine_with_mock(tmp_path / "clip.mp4")
        engine._enabled = True
        scene = Scene(
            idx=3, heading="h", narration="The narration text.",
            visual_prompt="v", scene_intent="shock",
        )
        raw_path = tmp_path / "heygen_scene_03_av_raw.mp4"
        raw_path.write_bytes(b"fake")

        with patch("src.avatar_engine.generate_heygen_clip", return_value=raw_path) as mock_hg:
            with patch("src.avatar_engine.ffmpeg_utils.burn_captions", side_effect=lambda v, t, o, **kw: v):
                with patch("shutil.copy2"):
                    engine.render_for_scene(scene, tmp_path / "audio.mp3", tmp_path)

        assert mock_hg.call_args.kwargs["avatar_style"] == "closeup"
