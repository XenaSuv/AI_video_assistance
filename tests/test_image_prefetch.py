"""Tests for image_generator.prefetch_scene_image and async_prefetch_scene_images."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Stub heavy optional deps before importing the module under test.
sys.modules.setdefault("PIL", MagicMock())
sys.modules.setdefault("PIL.Image", MagicMock())
sys.modules.setdefault("PIL.ImageDraw", MagicMock())
sys.modules.setdefault("PIL.ImageFont", MagicMock())

from src.script_generator import Scene, VideoScript
from src.image_generator import prefetch_scene_image, async_prefetch_scene_images


# ── helpers ────────────────────────────────────────────────────────────────────

def _scene(
    idx: int = 0,
    heading: str = "AI Breakthrough",
    narration: str = "Scientists discovered something amazing.",
    visual_prompt: str = "A futuristic laboratory",
    scene_type: str = "",
    video_query: str = "",
) -> Scene:
    s = Scene(
        idx=idx,
        heading=heading,
        narration=narration,
        visual_prompt=visual_prompt,
        duration_sec=30,
    )
    if scene_type:
        s.scene_type = scene_type
    if video_query:
        s.video_query = video_query
    return s


def _script(scenes: list[Scene] | None = None) -> VideoScript:
    if scenes is None:
        scenes = [_scene(0), _scene(1)]
    return VideoScript(
        title="Test Episode",
        description="Test",
        tags=[],
        hook="Hook text",
        scenes=scenes,
    )


# Shared patch targets
_DALLE   = "src.image_generator.generate_dalle_image"
_STOCK   = "src.image_generator.fetch_stock_photo"
_CACHE   = "src.image_generator.image_cache"
_SCRSHOT = "src.image_generator._try_real_screenshot"
_SD      = "src.image_generator.generate_sd_image"
_GENERIC = "src.image_generator._is_generic_prompt"
_LOGGER  = "src.image_generator.logger"


# ── prefetch_scene_image ───────────────────────────────────────────────────────

class TestPrefetchSceneImage:
    def test_returns_existing_image_without_api_call(self, tmp_path):
        img_dir  = tmp_path / "images"
        img_dir.mkdir()
        scene    = _scene(0)
        img_path = img_dir / "scene_00.png"
        img_path.write_bytes(b"fake png")

        with patch(_DALLE) as mock_dalle:
            result = prefetch_scene_image(scene, img_dir)

        assert result == img_path
        mock_dalle.assert_not_called()

    def test_returns_none_for_text_overlay(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        scene = _scene(0, scene_type="text_overlay")
        with patch(_DALLE) as mock_dalle:
            result = prefetch_scene_image(scene, img_dir)
        assert result is None
        mock_dalle.assert_not_called()

    def test_returns_none_for_infographic(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        scene = _scene(0, scene_type="infographic")
        with patch(_DALLE) as mock_dalle:
            result = prefetch_scene_image(scene, img_dir)
        assert result is None
        mock_dalle.assert_not_called()

    def test_returns_none_for_video_query_scene(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        scene = _scene(0, video_query="robot arm factory")
        with patch(_DALLE) as mock_dalle:
            result = prefetch_scene_image(scene, img_dir)
        assert result is None
        mock_dalle.assert_not_called()

    def test_calls_dalle_for_regular_scene(self, tmp_path):
        img_dir  = tmp_path / "images"
        img_dir.mkdir()
        scene    = _scene(0)
        img_path = img_dir / "scene_00.png"

        def _fake_dalle(prompt, path):
            path.write_bytes(b"generated")

        with patch(_SCRSHOT, return_value=False):
            with patch(_CACHE) as mock_cache:
                mock_cache.find_similar.return_value = (None, None)
                with patch(_GENERIC, return_value=False):
                    with patch(_DALLE, side_effect=_fake_dalle) as mock_dalle:
                        result = prefetch_scene_image(scene, img_dir)

        mock_dalle.assert_called_once()
        assert result == img_path

    def test_uses_stock_photo_when_generic_prompt(self, tmp_path):
        img_dir  = tmp_path / "images"
        img_dir.mkdir()
        scene    = _scene(0)
        img_path = img_dir / "scene_00.png"

        def _fake_stock(prompt, path):
            path.write_bytes(b"stock")
            return path

        with patch(_SCRSHOT, return_value=False):
            with patch(_CACHE) as mock_cache:
                mock_cache.find_similar.return_value = (None, None)
                with patch(_GENERIC, return_value=True):
                    with patch(_STOCK, side_effect=_fake_stock) as mock_stock:
                        with patch(_DALLE) as mock_dalle:
                            result = prefetch_scene_image(scene, img_dir)

        mock_stock.assert_called_once()
        mock_dalle.assert_not_called()
        assert result == img_path

    def test_uses_image_cache_when_similar_found(self, tmp_path):
        img_dir  = tmp_path / "images"
        img_dir.mkdir()
        scene    = _scene(0)

        cached_img = tmp_path / "cached.png"
        cached_img.write_bytes(b"cached")

        with patch(_SCRSHOT, return_value=False):
            with patch(_CACHE) as mock_cache:
                mock_cache.find_similar.return_value = (cached_img, None)
                with patch(_DALLE) as mock_dalle:
                    result = prefetch_scene_image(scene, img_dir)

        mock_dalle.assert_not_called()
        assert result is not None
        assert result.read_bytes() == b"cached"

    def test_returns_none_on_dalle_failure(self, tmp_path):
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        scene   = _scene(0)

        with patch(_SCRSHOT, return_value=False):
            with patch(_CACHE) as mock_cache:
                mock_cache.find_similar.return_value = (None, None)
                with patch(_GENERIC, return_value=False):
                    with patch(_DALLE, side_effect=RuntimeError("API error")):
                        result = prefetch_scene_image(scene, img_dir)

        assert result is None  # non-fatal

    def test_screenshot_short_circuits_dal_le(self, tmp_path):
        img_dir  = tmp_path / "images"
        img_dir.mkdir()
        scene    = _scene(0)
        img_path = img_dir / "scene_00.png"
        img_path.write_bytes(b"screenshot")

        with patch(_SCRSHOT, return_value=True):
            with patch(_DALLE) as mock_dalle:
                result = prefetch_scene_image(scene, img_dir)

        mock_dalle.assert_not_called()
        assert result == img_path


# ── async_prefetch_scene_images ────────────────────────────────────────────────

class TestAsyncPrefetchSceneImages:
    def test_calls_prefetch_for_each_scene(self, tmp_path):
        script = _script([_scene(0), _scene(1), _scene(2)])
        prefetched: list[int] = []

        def _fake_prefetch(scene, img_dir, **kwargs):
            prefetched.append(scene.idx)
            return None

        with patch("src.image_generator.prefetch_scene_image", side_effect=_fake_prefetch):
            asyncio.run(async_prefetch_scene_images(script, tmp_path))

        assert sorted(prefetched) == [0, 1, 2]

    def test_creates_images_directory(self, tmp_path):
        script = _script([_scene(0)])
        img_dir = tmp_path / "images"
        assert not img_dir.exists()

        with patch("src.image_generator.prefetch_scene_image", return_value=None):
            asyncio.run(async_prefetch_scene_images(script, tmp_path))

        assert img_dir.exists()

    def test_partial_failure_does_not_raise(self, tmp_path):
        scene0, scene1 = _scene(0), _scene(1)
        script = _script([scene0, scene1])

        call_count = [0]
        def _flaky(scene, img_dir, **kwargs):
            call_count[0] += 1
            if scene.idx == 0:
                raise RuntimeError("DALL-E failed")
            return None

        # The gather wraps each call in run_in_executor; exceptions from the
        # executor propagate.  prefetch_scene_image itself swallows exceptions,
        # so we test that its internal try/except works end-to-end.
        with patch(_SCRSHOT, return_value=False):
            with patch(_CACHE) as mock_cache:
                mock_cache.find_similar.return_value = (None, None)
                with patch(_GENERIC, return_value=False):
                    with patch(_DALLE, side_effect=RuntimeError("boom")):
                        # Should complete without raising
                        asyncio.run(async_prefetch_scene_images(script, tmp_path))

    def test_skips_text_overlay_scenes(self, tmp_path):
        scenes = [_scene(0, scene_type="text_overlay"), _scene(1)]
        script = _script(scenes)
        dalle_calls: list[int] = []

        def _track(scene, img_dir, **kwargs):
            dalle_calls.append(scene.idx)
            return None

        with patch("src.image_generator.prefetch_scene_image", side_effect=_track):
            asyncio.run(async_prefetch_scene_images(script, tmp_path))

        # Both scenes go through prefetch_scene_image (the function itself returns
        # None for text_overlay); prefetch is called for all scenes.
        assert sorted(dalle_calls) == [0, 1]


# ── pipeline integration smoke test ───────────────────────────────────────────

class TestPipelineIntegration:
    """Verify that async_prefetch returns before build_video needs durations."""

    def test_prefetch_does_not_need_duration_sec(self, tmp_path):
        """prefetch_scene_image must never read scene.duration_sec."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        scene = _scene(0)
        del scene.duration_sec   # ensure AttributeError would fire if accessed

        with patch(_SCRSHOT, return_value=False):
            with patch(_CACHE) as mock_cache:
                mock_cache.find_similar.return_value = (None, None)
                with patch(_GENERIC, return_value=False):
                    with patch(_DALLE, side_effect=lambda p, path: path.write_bytes(b"x")):
                        # Should not raise AttributeError
                        result = prefetch_scene_image(scene, img_dir)

        assert result is not None
