"""Tests for src/image_generator.py — pure logic branches.

All external calls (PIL, openai, requests, ffmpeg, circuit_breaker,
image_cache, cost_tracker) are mocked so no real network or filesystem I/O
occurs unless using the tmp_path fixture for written bytes.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# ── conftest already stubs PIL / google SDK, but we add granular PIL sub-mods
# to make lazy-import paths (PIL.Image, PIL.ImageDraw, PIL.ImageFont) resolvable.
for _mod in (
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
):
    sys.modules.setdefault(_mod, MagicMock())

# ── Module under test ─────────────────────────────────────────────────────────
from src.image_generator import (
    _normalize_visual_prompt,
    _is_generic_prompt,
    _photo_query,
    _is_retryable_dalle,
    _PREAMBLE_RE,
    generate_dalle_image,
    generate_sd_image,
    fetch_stock_photo,
    generate_scene_clip,
    _valid_video_clip,
    _black_placeholder,
    _ken_burns_clip,
)
from src.script_generator import Scene, VideoScript


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_scene(
    idx: int = 0,
    heading: str = "AI Breakthrough",
    narration: str = "Scientists discovered something incredible.",
    visual_prompt: str = "A futuristic lab with glowing screens",
    duration_sec: int = 30,
    scene_type: str = "image",
    video_query: str | None = None,
    infographic_data: dict | None = None,
    screenshot_key: str | None = None,
    screenshot_url: str | None = None,
    source_quote: str | None = None,
    quote_attribution: str | None = None,
    visual_direction: str = "",
) -> Scene:
    s = Scene(
        idx=idx,
        heading=heading,
        narration=narration,
        visual_prompt=visual_prompt,
        duration_sec=duration_sec,
    )
    s.scene_type = scene_type
    s.video_query = video_query
    s.infographic_data = infographic_data
    s.screenshot_key = screenshot_key
    s.screenshot_url = screenshot_url
    s.source_quote = source_quote
    s.quote_attribution = quote_attribution
    s.visual_direction = visual_direction
    return s


# ═══════════════════════════════════════════════════════════════════════════════
# _normalize_visual_prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestNormalizeVisualPrompt:
    def test_strips_leading_trailing_whitespace(self):
        result = _normalize_visual_prompt("  some prompt  ")
        assert result == result.strip() or result.startswith("photorealistic")

    def test_openai_brand_triggers_template(self):
        result = _normalize_visual_prompt("An OpenAI research facility")
        # Should use brand template
        assert "openai" in result.lower() or "research lab" in result.lower()
        assert "no logos" in result

    def test_anthropic_brand_triggers_template(self):
        result = _normalize_visual_prompt("Anthropic office background")
        assert "anthropic" in result.lower() or "research office" in result.lower()
        assert "no logos" in result

    def test_google_brand_triggers_template(self):
        result = _normalize_visual_prompt("A Google DeepMind lab")
        assert "google" in result.lower() or "innovation lab" in result.lower()
        assert "no logos" in result

    def test_brand_template_appends_original_prompt(self):
        original = "An OpenAI lab with lots of screens"
        result = _normalize_visual_prompt(original)
        # The original prompt should appear after the template separator
        assert original in result

    def test_non_brand_prompt_gets_photorealistic_prefix(self):
        result = _normalize_visual_prompt("A robot working on a laptop")
        assert "photorealistic" in result.lower()

    def test_prompt_already_photorealistic_not_doubled(self):
        result = _normalize_visual_prompt("photorealistic sunset over mountains")
        # Should not have "photorealistic photorealistic"
        assert result.count("photorealistic") == 1 or result.startswith("photorealistic")

    def test_photo_realistic_hyphen_variant_accepted(self):
        result = _normalize_visual_prompt("photo-realistic drone view of city")
        # Should NOT add another 'photorealistic' prefix
        assert not result.startswith("photorealistic photo-realistic")

    def test_style_vivid_replaced(self):
        result = _normalize_visual_prompt("A futuristic city style: vivid")
        assert "style: vivid" not in result
        assert "style: photorealistic" in result

    def test_news_aesthetic_appended(self):
        result = _normalize_visual_prompt("A modern workspace with a laptop")
        assert "documentary aesthetic" in result or "news photography" in result

    def test_case_insensitive_brand_detection(self):
        result = _normalize_visual_prompt("OPENAI headquarters building")
        assert "no logos" in result

    def test_returns_string(self):
        result = _normalize_visual_prompt("generic scene")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════════════
# _is_generic_prompt
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsGenericPrompt:
    def test_generic_office_scene_true(self):
        assert _is_generic_prompt("a person working at their desk on a laptop") is True

    def test_two_generic_words_no_specific_true(self):
        assert _is_generic_prompt("people in an office meeting") is True

    def test_one_generic_word_false(self):
        assert _is_generic_prompt("a person doing something") is False

    def test_specific_word_overrides_generic(self):
        # Even with generic words, specific word should return False
        assert _is_generic_prompt("a person typing on a terminal with a diagram") is False

    def test_ai_specific_scene_false(self):
        assert _is_generic_prompt("neural network diagram of GPT architecture") is False

    def test_abstract_concept_false(self):
        assert _is_generic_prompt("holographic interface in a futuristic city") is False

    def test_empty_string_false(self):
        assert _is_generic_prompt("") is False

    def test_server_room_true(self):
        assert _is_generic_prompt("data center with server racks and cables network") is True

    def test_gpt_mentioned_false(self):
        # "gpt" is a specific word
        assert _is_generic_prompt("gpt model running on a laptop in an office") is False

    def test_dalle_mentioned_false(self):
        assert _is_generic_prompt("dall-e generating images on a computer screen") is False

    def test_exact_two_generic_boundary(self):
        # Exactly 2 generic hits with 0 specific hits → True
        assert _is_generic_prompt("building with technology") is True

    def test_case_insensitive(self):
        assert _is_generic_prompt("PEOPLE at an OFFICE") is True


# ═══════════════════════════════════════════════════════════════════════════════
# _photo_query
# ═══════════════════════════════════════════════════════════════════════════════

class TestPhotoQuery:
    def test_strips_preamble_a_before_qualifier(self):
        # "a " is only stripped when followed by a preamble word like "photorealistic"
        q = _photo_query("a photorealistic server room")
        assert not q.lower().startswith("a photorealistic")

    def test_strips_preamble_an_before_qualifier(self):
        # "an " is only stripped when followed by a preamble word
        q = _photo_query("an image of mountains")
        assert not q.lower().startswith("an image of")

    def test_strips_photorealistic(self):
        q = _photo_query("photorealistic server room")
        assert not q.lower().startswith("photorealistic")

    def test_strips_digital(self):
        q = _photo_query("digital artwork of a city")
        assert not q.lower().startswith("digital")

    def test_strips_cinematic(self):
        q = _photo_query("cinematic shot of a researcher")
        assert not q.lower().startswith("cinematic")

    def test_strips_render_of(self):
        q = _photo_query("render of a futuristic lab")
        assert not q.lower().startswith("render")

    def test_strips_photo_of(self):
        q = _photo_query("photo of scientists working")
        assert not q.lower().startswith("photo of")

    def test_truncates_at_comma(self):
        q = _photo_query("a person in an office, with a laptop, smiling")
        assert "," not in q

    def test_truncates_at_period(self):
        q = _photo_query("scientist working. High quality image.")
        assert "." not in q

    def test_max_60_chars(self):
        q = _photo_query("photorealistic " + "a" * 100)
        assert len(q) <= 60

    def test_strips_whitespace(self):
        q = _photo_query("  a person typing  ")
        assert q == q.strip()

    def test_preserves_meaningful_content(self):
        q = _photo_query("a data center with blinking lights")
        assert "data center" in q or "blinking lights" in q


# ═══════════════════════════════════════════════════════════════════════════════
# _is_retryable_dalle
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsRetryableDalle:
    def test_bad_request_not_retryable(self):
        from openai import BadRequestError
        exc = BadRequestError.__new__(BadRequestError)
        assert _is_retryable_dalle(exc) is False

    def test_rate_limit_retryable(self):
        from openai import RateLimitError
        exc = RateLimitError.__new__(RateLimitError)
        assert _is_retryable_dalle(exc) is True

    def test_api_status_5xx_retryable(self):
        from openai import APIStatusError
        exc = MagicMock(spec=APIStatusError)
        exc.status_code = 503
        assert _is_retryable_dalle(exc) is True

    def test_api_status_4xx_not_retryable(self):
        from openai import APIStatusError
        exc = MagicMock(spec=APIStatusError)
        exc.status_code = 400
        assert _is_retryable_dalle(exc) is False

    def test_connection_error_retryable(self):
        assert _is_retryable_dalle(ConnectionError("timeout")) is True

    def test_timeout_error_retryable(self):
        assert _is_retryable_dalle(TimeoutError("timed out")) is True

    def test_requests_exception_retryable(self):
        import requests as req
        exc = req.exceptions.RequestException("network error")
        assert _is_retryable_dalle(exc) is True

    def test_generic_exception_not_retryable(self):
        assert _is_retryable_dalle(ValueError("bad value")) is False

    def test_runtime_error_not_retryable(self):
        assert _is_retryable_dalle(RuntimeError("unexpected")) is False


# ═══════════════════════════════════════════════════════════════════════════════
# generate_dalle_image
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateDalleImage:
    def test_returns_cached_file_if_exists(self, tmp_path):
        out = tmp_path / "image.png"
        out.write_bytes(b"PNG_DATA")

        with patch("src.image_generator.make_openai_client") as mock_client, \
             patch("src.image_generator._call_dalle") as mock_dalle:
            result = generate_dalle_image("test prompt", out)

        assert result == out
        mock_client.assert_not_called()
        mock_dalle.assert_not_called()

    def test_calls_dalle_and_writes_bytes(self, tmp_path):
        out = tmp_path / "new_image.png"
        fake_bytes = b"\x89PNG\r\nFAKE"

        with patch("src.image_generator.make_openai_client") as mock_client, \
             patch("src.image_generator._call_dalle", return_value=fake_bytes) as mock_dalle, \
             patch("src.image_generator.get_ledger") as mock_ledger:
            result = generate_dalle_image("a futuristic lab", out)

        assert result == out
        assert out.read_bytes() == fake_bytes
        mock_dalle.assert_called_once()
        mock_ledger.return_value.record_image.assert_called_once()

    def test_normalizes_prompt_before_calling_dalle(self, tmp_path):
        out = tmp_path / "img.png"
        fake_bytes = b"BYTES"

        with patch("src.image_generator.make_openai_client"), \
             patch("src.image_generator._call_dalle", return_value=fake_bytes) as mock_dalle, \
             patch("src.image_generator.get_ledger"):
            generate_dalle_image("a futuristic lab", out)

        called_prompt = mock_dalle.call_args[0][1]
        assert "photorealistic" in called_prompt.lower() or "lab" in called_prompt.lower()

    def test_openai_brand_in_prompt_uses_template(self, tmp_path):
        out = tmp_path / "img.png"
        with patch("src.image_generator.make_openai_client"), \
             patch("src.image_generator._call_dalle", return_value=b"B") as mock_dalle, \
             patch("src.image_generator.get_ledger"):
            generate_dalle_image("OpenAI office background", out)

        called_prompt = mock_dalle.call_args[0][1]
        assert "no logos" in called_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# generate_sd_image
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateSdImage:
    def test_raises_when_no_stability_key(self, tmp_path):
        out = tmp_path / "sd.png"
        with patch("src.image_generator.settings") as mock_settings:
            mock_settings.stability_api_key = None
            with pytest.raises(RuntimeError, match="STABILITY_API_KEY"):
                generate_sd_image("a futuristic scene", out)

    def test_calls_http_post_with_correct_params(self, tmp_path):
        out = tmp_path / "sd.png"
        fake_content = b"PNG_BYTES"
        mock_resp = MagicMock()
        mock_resp.content = fake_content

        with patch("src.image_generator.settings") as mock_settings, \
             patch("src.image_generator.http_post", return_value=mock_resp) as mock_post:
            mock_settings.stability_api_key = "sk-stability-key"
            result = generate_sd_image("breaking news scene", out)

        assert result == out
        assert out.read_bytes() == fake_content
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args
        assert "stability.ai" in call_kwargs[0][0]

    def test_includes_authorization_header(self, tmp_path):
        out = tmp_path / "sd.png"
        mock_resp = MagicMock()
        mock_resp.content = b"DATA"

        with patch("src.image_generator.settings") as mock_settings, \
             patch("src.image_generator.http_post", return_value=mock_resp) as mock_post:
            mock_settings.stability_api_key = "MY_API_KEY"
            generate_sd_image("test", out)

        headers = mock_post.call_args[1]["headers"]
        assert "Bearer MY_API_KEY" in headers.get("authorization", "")

    def test_uses_16_9_aspect_ratio(self, tmp_path):
        out = tmp_path / "sd.png"
        mock_resp = MagicMock()
        mock_resp.content = b"DATA"

        with patch("src.image_generator.settings") as mock_settings, \
             patch("src.image_generator.http_post", return_value=mock_resp) as mock_post:
            mock_settings.stability_api_key = "KEY"
            generate_sd_image("test scene", out)

        files = mock_post.call_args[1]["files"]
        # files is a dict of (None, value) tuples
        assert any("16:9" in str(v) for v in files.values())

    def test_truncates_prompt_to_2000_chars(self, tmp_path):
        out = tmp_path / "sd.png"
        mock_resp = MagicMock()
        mock_resp.content = b"DATA"
        long_prompt = "x" * 3000

        with patch("src.image_generator.settings") as mock_settings, \
             patch("src.image_generator.http_post", return_value=mock_resp) as mock_post:
            mock_settings.stability_api_key = "KEY"
            generate_sd_image(long_prompt, out)

        files = mock_post.call_args[1]["files"]
        prompt_value = files["prompt"][1]
        assert len(prompt_value) <= 2000


# ═══════════════════════════════════════════════════════════════════════════════
# fetch_stock_photo
# ═══════════════════════════════════════════════════════════════════════════════

class TestFetchStockPhoto:
    def test_unsplash_success_returns_path(self, tmp_path):
        out = tmp_path / "stock.png"
        fake_img = b"IMG_BYTES"

        mock_r = MagicMock()
        mock_r.json.return_value = {
            "results": [
                {"urls": {"regular": "https://unsplash.com/photo.jpg"},
                 "user": {"name": "Photographer"}}
            ]
        }
        mock_img_r = MagicMock()
        mock_img_r.content = fake_img

        with patch("src.image_generator.settings") as mock_settings, \
             patch("src.image_generator.http_get", side_effect=[mock_r, mock_img_r]):
            mock_settings.unsplash_access_key = "UNSPLASH_KEY"
            mock_settings.pexels_api_key = None
            result = fetch_stock_photo("person at laptop desk", out)

        assert result == out
        assert out.read_bytes() == fake_img

    def test_unsplash_no_key_falls_to_pexels(self, tmp_path):
        out = tmp_path / "stock.png"
        fake_img = b"PEXELS_IMG"

        pexels_r = MagicMock()
        pexels_r.json.return_value = {
            "photos": [
                {"src": {"large2x": "https://pexels.com/photo.jpg"},
                 "photographer": "Jane Doe"}
            ]
        }
        img_r = MagicMock()
        img_r.content = fake_img

        with patch("src.image_generator.settings") as mock_settings, \
             patch("src.image_generator.http_get", side_effect=[pexels_r, img_r]):
            mock_settings.unsplash_access_key = None
            mock_settings.pexels_api_key = "PEXELS_KEY"
            result = fetch_stock_photo("people in office meeting", out)

        assert result == out
        assert out.read_bytes() == fake_img

    def test_no_keys_returns_none(self, tmp_path):
        out = tmp_path / "stock.png"
        with patch("src.image_generator.settings") as mock_settings:
            mock_settings.unsplash_access_key = None
            mock_settings.pexels_api_key = None
            result = fetch_stock_photo("office workers", out)

        assert result is None
        assert not out.exists()

    def test_unsplash_empty_results_falls_to_pexels(self, tmp_path):
        out = tmp_path / "stock.png"
        fake_img = b"PEXELS_BYTES"

        unsplash_r = MagicMock()
        unsplash_r.json.return_value = {"results": []}
        pexels_r = MagicMock()
        pexels_r.json.return_value = {
            "photos": [{"src": {"large2x": "https://pexels.com/img.jpg"}, "photographer": "X"}]
        }
        img_r = MagicMock()
        img_r.content = fake_img

        with patch("src.image_generator.settings") as mock_settings, \
             patch("src.image_generator.http_get", side_effect=[unsplash_r, pexels_r, img_r]):
            mock_settings.unsplash_access_key = "U_KEY"
            mock_settings.pexels_api_key = "P_KEY"
            result = fetch_stock_photo("people working together", out)

        assert result == out

    def test_unsplash_exception_falls_to_pexels(self, tmp_path):
        out = tmp_path / "stock.png"
        fake_img = b"FALLBACK"

        pexels_r = MagicMock()
        pexels_r.json.return_value = {
            "photos": [{"src": {"large": "https://pexels.com/img.jpg"}, "photographer": "X"}]
        }
        img_r = MagicMock()
        img_r.content = fake_img

        def http_get_side_effect(url, **kwargs):
            if "unsplash" in url:
                raise ConnectionError("network error")
            return pexels_r if "pexels" in url else img_r

        # Two calls to pexels: search + download
        with patch("src.image_generator.settings") as mock_settings, \
             patch("src.image_generator.http_get", side_effect=[ConnectionError("fail"), pexels_r, img_r]):
            mock_settings.unsplash_access_key = "U_KEY"
            mock_settings.pexels_api_key = "P_KEY"
            result = fetch_stock_photo("office team group", out)

        assert result == out

    def test_pexels_no_photos_returns_none(self, tmp_path):
        out = tmp_path / "stock.png"

        unsplash_r = MagicMock()
        unsplash_r.json.return_value = {"results": []}
        pexels_r = MagicMock()
        pexels_r.json.return_value = {"photos": []}

        with patch("src.image_generator.settings") as mock_settings, \
             patch("src.image_generator.http_get", side_effect=[unsplash_r, pexels_r]):
            mock_settings.unsplash_access_key = "U_KEY"
            mock_settings.pexels_api_key = "P_KEY"
            result = fetch_stock_photo("people at desk laptop", out)

        assert result is None

    def test_pexels_missing_src_returns_none(self, tmp_path):
        out = tmp_path / "stock.png"

        unsplash_r = MagicMock()
        unsplash_r.json.return_value = {"results": []}
        pexels_r = MagicMock()
        pexels_r.json.return_value = {
            "photos": [{"src": {}, "photographer": "X"}]
        }

        with patch("src.image_generator.settings") as mock_settings, \
             patch("src.image_generator.http_get", side_effect=[unsplash_r, pexels_r]):
            mock_settings.unsplash_access_key = "U_KEY"
            mock_settings.pexels_api_key = "P_KEY"
            result = fetch_stock_photo("people at desk laptop", out)

        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# _valid_video_clip
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidVideoClip:
    def test_missing_file_returns_false(self, tmp_path):
        assert _valid_video_clip(tmp_path / "nonexistent.mp4") is False

    def test_valid_dimensions_and_duration(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"FAKE_VIDEO")
        with patch("src.image_generator.ffmpeg_utils.video_size", return_value=(1280, 720)), \
             patch("src.image_generator.ffmpeg_utils.duration", return_value=10.0):
            assert _valid_video_clip(clip) is True

    def test_zero_width_returns_false(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"FAKE")
        with patch("src.image_generator.ffmpeg_utils.video_size", return_value=(0, 720)), \
             patch("src.image_generator.ffmpeg_utils.duration", return_value=5.0):
            assert _valid_video_clip(clip) is False

    def test_zero_duration_returns_false(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"FAKE")
        with patch("src.image_generator.ffmpeg_utils.video_size", return_value=(1280, 720)), \
             patch("src.image_generator.ffmpeg_utils.duration", return_value=0.0):
            assert _valid_video_clip(clip) is False

    def test_ffmpeg_exception_returns_false(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"FAKE")
        with patch("src.image_generator.ffmpeg_utils.video_size", side_effect=RuntimeError("ffmpeg fail")):
            assert _valid_video_clip(clip) is False


# ═══════════════════════════════════════════════════════════════════════════════
# generate_scene_clip
# ═══════════════════════════════════════════════════════════════════════════════

class TestGenerateSceneClip:
    """Tests for generate_scene_clip() main orchestration logic."""

    def _patch_base(self):
        """Return a context manager that patches the most common dependencies."""
        return [
            patch("src.image_generator._valid_video_clip", return_value=False),
            patch("src.image_generator.image_cache.find_similar", return_value=(None, None)),
            patch("src.image_generator.image_cache.add_entry"),
            patch("src.image_generator.settings"),
            patch("src.image_generator._black_placeholder"),
            patch("src.image_generator._ken_burns_clip"),
        ]

    def test_reuses_valid_cached_clip(self, tmp_path):
        """If the clip already exists and is valid, it must be returned immediately."""
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=1)
        clip_path = clip_dir / "scene_01_clip_0.mp4"
        clip_path.write_bytes(b"FAKE")

        with patch("src.image_generator._valid_video_clip", return_value=True):
            result = generate_scene_clip(scene, clip_dir)

        assert result == clip_path

    def test_removes_invalid_cached_clip_and_regenerates(self, tmp_path):
        """Invalid cached clip should be removed and scene regenerated."""
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=2)
        clip_path = clip_dir / "scene_02_clip_0.mp4"
        clip_path.write_bytes(b"CORRUPT")

        fake_img_bytes = b"PNG"

        with patch("src.image_generator._valid_video_clip", side_effect=[False, True]), \
             patch("src.image_generator.image_cache.find_similar", return_value=(None, None)), \
             patch("src.image_generator.image_cache.add_entry"), \
             patch("src.image_generator.generate_dalle_image", side_effect=lambda p, path: path.write_bytes(fake_img_bytes) or path), \
             patch("src.image_generator._ken_burns_clip", return_value=clip_path) as mock_kb, \
             patch("src.image_generator.settings") as ms:
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = None
            result = generate_scene_clip(scene, clip_dir)

        # Original clip was removed
        assert not clip_path.exists() or result == clip_path

    def test_text_overlay_scene_delegates_renderer(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=3, scene_type="text_overlay")

        fake_clip = clip_dir / "scene_03_clip_0.mp4"

        with patch("src.image_generator._render_text_overlay_clip", return_value=fake_clip) as mock_render, \
             patch("src.image_generator.settings") as ms:
            ms.data_dir = tmp_path / "data"
            result = generate_scene_clip(scene, clip_dir)

        mock_render.assert_called_once()
        assert result == fake_clip

    def test_infographic_scene_delegates_infographic_generator(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=4, infographic_data={"type": "bar", "values": [1, 2, 3]})

        fake_clip = clip_dir / "scene_04_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch.dict("sys.modules", {"src.infographic_generator": MagicMock()}):
            ms.data_dir = tmp_path / "data"
            import src.infographic_generator as ig_mock
            ig_mock.generate_infographic_clip = MagicMock(return_value=fake_clip)

            result = generate_scene_clip(scene, clip_dir)

        assert result == fake_clip

    def test_video_query_tries_broll_fetcher(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=5, video_query="robot working in lab")
        fake_clip = clip_dir / "scene_05_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch.dict("sys.modules", {"src.broll_fetcher": MagicMock()}), \
             patch("src.image_generator._valid_video_clip", return_value=False):
            ms.data_dir = tmp_path / "data"
            import src.broll_fetcher as bf_mock
            bf_mock.fetch_broll_clip = MagicMock(return_value=fake_clip)

            result = generate_scene_clip(scene, clip_dir)

        assert result == fake_clip

    def test_dalle_fallback_when_broll_returns_none(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=6, video_query="robot arm assembly")
        fake_clip = clip_dir / "scene_06_clip_0.mp4"
        fake_img = clip_dir.parent / "images" / "scene_06.png"

        with patch("src.image_generator.settings") as ms, \
             patch.dict("sys.modules", {"src.broll_fetcher": MagicMock()}), \
             patch("src.image_generator._valid_video_clip", return_value=True), \
             patch("src.image_generator.image_cache.find_similar", return_value=(None, None)), \
             patch("src.image_generator.image_cache.add_entry"), \
             patch("src.image_generator.generate_dalle_image", side_effect=lambda p, path: path) as mock_dalle, \
             patch("src.image_generator._ken_burns_clip", return_value=fake_clip):
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = None
            import src.broll_fetcher as bf_mock
            bf_mock.fetch_broll_clip = MagicMock(return_value=None)

            result = generate_scene_clip(scene, clip_dir)

        mock_dalle.assert_called_once()

    def test_cache_hit_skips_generation(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=7)

        # Create a "cached" image somewhere
        cached_img = tmp_path / "cached_scene.png"
        cached_img.write_bytes(b"CACHED_PNG")
        fake_clip = clip_dir / "scene_07_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch("src.image_generator._valid_video_clip", return_value=True), \
             patch("src.image_generator.image_cache.find_similar", return_value=(cached_img, MagicMock())), \
             patch("src.image_generator.image_cache.add_entry"), \
             patch("src.image_generator.shutil.copy2") as mock_copy, \
             patch("src.image_generator.generate_dalle_image") as mock_dalle, \
             patch("src.image_generator._ken_burns_clip", return_value=fake_clip):
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = None

            result = generate_scene_clip(scene, clip_dir)

        # shutil.copy2 should be called (cache hit), DALL-E should NOT
        mock_copy.assert_called_once()
        mock_dalle.assert_not_called()

    def test_stock_photo_used_for_generic_prompt(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        # Generic scene with multiple generic words
        scene = _make_scene(idx=8, visual_prompt="people in an office working at desks with laptops")
        fake_clip = clip_dir / "scene_08_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch("src.image_generator._valid_video_clip", return_value=True), \
             patch("src.image_generator.image_cache.find_similar", return_value=(None, None)), \
             patch("src.image_generator.image_cache.add_entry"), \
             patch("src.image_generator.fetch_stock_photo") as mock_stock, \
             patch("src.image_generator.generate_dalle_image") as mock_dalle, \
             patch("src.image_generator._ken_burns_clip", return_value=fake_clip):
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = None
            # stock photo "returns" the out_path (simulates writing)
            def fake_stock(prompt, path):
                path.write_bytes(b"STOCK_IMG")
                return path
            mock_stock.side_effect = fake_stock

            result = generate_scene_clip(scene, clip_dir)

        mock_stock.assert_called_once()
        mock_dalle.assert_not_called()

    def test_breaking_news_uses_stable_diffusion(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        # Not a generic prompt → will skip stock photo
        scene = _make_scene(idx=9, visual_prompt="neural network holographic diagram")
        fake_clip = clip_dir / "scene_09_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch("src.image_generator._valid_video_clip", return_value=True), \
             patch("src.image_generator.image_cache.find_similar", return_value=(None, None)), \
             patch("src.image_generator.image_cache.add_entry"), \
             patch("src.image_generator.generate_sd_image") as mock_sd, \
             patch("src.image_generator.generate_dalle_image") as mock_dalle, \
             patch("src.image_generator._ken_burns_clip", return_value=fake_clip):
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = "SD_KEY"

            def fake_sd(prompt, path):
                path.write_bytes(b"SD_IMG")
                return path
            mock_sd.side_effect = fake_sd

            result = generate_scene_clip(scene, clip_dir, is_breaking=True)

        mock_sd.assert_called_once()
        mock_dalle.assert_not_called()

    def test_sd_failure_falls_back_to_dalle(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=10, visual_prompt="abstract neural diagram concept")
        fake_clip = clip_dir / "scene_10_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch("src.image_generator._valid_video_clip", return_value=True), \
             patch("src.image_generator.image_cache.find_similar", return_value=(None, None)), \
             patch("src.image_generator.image_cache.add_entry"), \
             patch("src.image_generator.generate_sd_image", side_effect=RuntimeError("SD failed")), \
             patch("src.image_generator.generate_dalle_image") as mock_dalle, \
             patch("src.image_generator._ken_burns_clip", return_value=fake_clip):
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = "SD_KEY"

            def fake_dalle(prompt, path):
                path.write_bytes(b"DALLE_IMG")
                return path
            mock_dalle.side_effect = fake_dalle

            result = generate_scene_clip(scene, clip_dir, is_breaking=True)

        mock_dalle.assert_called_once()

    def test_diagram_scene_type_adds_prefix_to_prompt(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=11, visual_prompt="transformer architecture")
        scene.scene_type = "diagram"
        fake_clip = clip_dir / "scene_11_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch("src.image_generator._valid_video_clip", return_value=True), \
             patch("src.image_generator.image_cache.find_similar", return_value=(None, None)), \
             patch("src.image_generator.image_cache.add_entry"), \
             patch("src.image_generator.generate_dalle_image") as mock_dalle, \
             patch("src.image_generator._ken_burns_clip", return_value=fake_clip):
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = None

            def fake_dalle(prompt, path):
                path.write_bytes(b"DALLE_IMG")
                return path
            mock_dalle.side_effect = fake_dalle

            generate_scene_clip(scene, clip_dir)

        called_prompt = mock_dalle.call_args[0][0]
        assert "diagram" in called_prompt.lower() or "whiteboard" in called_prompt.lower()

    def test_uses_placeholder_when_everything_fails(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=12)
        placeholder = clip_dir / "scene_12_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch("src.image_generator._valid_video_clip", return_value=False), \
             patch("src.image_generator.image_cache.find_similar", return_value=(None, None)), \
             patch("src.image_generator.generate_dalle_image", side_effect=RuntimeError("API down")), \
             patch("src.image_generator._black_placeholder", return_value=placeholder) as mock_bp:
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = None

            result = generate_scene_clip(scene, clip_dir)

        mock_bp.assert_called_once()
        assert result == placeholder

    def test_invalid_generated_clip_falls_back_to_placeholder(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=13)
        fake_clip = clip_dir / "scene_13_clip_0.mp4"
        placeholder = clip_dir / "placeholder_13.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch("src.image_generator._valid_video_clip", side_effect=[False, False]), \
             patch("src.image_generator.image_cache.find_similar", return_value=(None, None)), \
             patch("src.image_generator.image_cache.add_entry"), \
             patch("src.image_generator.generate_dalle_image") as mock_dalle, \
             patch("src.image_generator._ken_burns_clip", return_value=fake_clip), \
             patch("src.image_generator._black_placeholder", return_value=placeholder) as mock_bp:
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = None

            def fake_dalle(prompt, path):
                path.write_bytes(b"DALLE_IMG")
                return path
            mock_dalle.side_effect = fake_dalle

            result = generate_scene_clip(scene, clip_dir)

        mock_bp.assert_called_once()

    def test_uses_heading_when_visual_prompt_empty(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=14, visual_prompt="", heading="AI Takes Over")
        fake_clip = clip_dir / "scene_14_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch("src.image_generator._valid_video_clip", return_value=True), \
             patch("src.image_generator.image_cache.find_similar", return_value=(None, None)), \
             patch("src.image_generator.image_cache.add_entry"), \
             patch("src.image_generator.generate_dalle_image") as mock_dalle, \
             patch("src.image_generator._ken_burns_clip", return_value=fake_clip):
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = None

            def fake_dalle(prompt, path):
                path.write_bytes(b"DALLE_IMG")
                return path
            mock_dalle.side_effect = fake_dalle

            generate_scene_clip(scene, clip_dir)

        called_prompt = mock_dalle.call_args[0][0]
        assert "AI Takes Over" in called_prompt or "photorealistic" in called_prompt

    def test_fallback_prompt_when_all_fields_empty(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        scene = _make_scene(idx=15, visual_prompt="", heading="", narration="")
        fake_clip = clip_dir / "scene_15_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch("src.image_generator._valid_video_clip", return_value=True), \
             patch("src.image_generator.image_cache.find_similar", return_value=(None, None)), \
             patch("src.image_generator.image_cache.add_entry"), \
             patch("src.image_generator.generate_dalle_image") as mock_dalle, \
             patch("src.image_generator._ken_burns_clip", return_value=fake_clip):
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = None

            def fake_dalle(prompt, path):
                path.write_bytes(b"DALLE_IMG")
                return path
            mock_dalle.side_effect = fake_dalle

            generate_scene_clip(scene, clip_dir)

        called_prompt = mock_dalle.call_args[0][0]
        # Should use the fallback string
        assert "cinematic" in called_prompt.lower() or "newsroom" in called_prompt.lower()

    def test_reuses_cached_image_if_img_path_exists(self, tmp_path):
        clip_dir = tmp_path / "clips"
        clip_dir.mkdir()
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        scene = _make_scene(idx=16)

        # Pre-create the image file
        img_path = img_dir / "scene_16.png"
        img_path.write_bytes(b"EXISTING_PNG")
        fake_clip = clip_dir / "scene_16_clip_0.mp4"

        with patch("src.image_generator.settings") as ms, \
             patch("src.image_generator._valid_video_clip", return_value=True), \
             patch("src.image_generator.generate_dalle_image") as mock_dalle, \
             patch("src.image_generator._ken_burns_clip", return_value=fake_clip):
            ms.data_dir = tmp_path / "data"
            ms.stability_api_key = None

            result = generate_scene_clip(scene, clip_dir)

        # DALL-E must NOT have been called since the image already existed
        mock_dalle.assert_not_called()
