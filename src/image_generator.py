"""Generate scene b-roll by producing an image and animating it with Ken Burns.

Image source selection (cheapest-first):
  1. image_cache      — cosine-similar prompt found in previous runs  (~$0.000006)
  2. stock_photo      — Unsplash / Pexels photo for generic real-world scenes  (free)
  3. stable_diffusion — Stability AI stable-image-core for breaking-news clips  (~$0.003)
  4. dall_e_3         — DALL-E 3 HD for all other scenes                       (~$0.080)

Cost at ~8 scenes/episode (daily), 5 scenes (breaking):
  Daily   — with cache + stock photo: expect 40-60 % reduction vs DALL-E for all
  Breaking — Stability AI replaces DALL-E entirely: ~$0.015 vs ~$0.40/episode
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import requests
from loguru import logger
from openai import BadRequestError, OpenAI, RateLimitError, APIStatusError
from PIL import Image as PILImage
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.cost_tracker import get_ledger
from src.script_generator import Scene
import src.ffmpeg_utils as ffmpeg_utils
import src.image_cache as image_cache

OUT_W, OUT_H = 1280, 720


# ── DALL-E 3 ─────────────────────────────────────────────────────────────────

def _is_retryable_dalle(exc: BaseException) -> bool:
    if isinstance(exc, BadRequestError):
        return False
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return isinstance(exc, (requests.exceptions.RequestException, ConnectionError, TimeoutError))


def _log_dalle_retry(retry_state) -> None:
    logger.warning(f"DALL-E retry {retry_state.attempt_number}: {retry_state.outcome.exception()}")


@retry(
    retry=retry_if_exception(_is_retryable_dalle),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(4),
    before_sleep=_log_dalle_retry,
    reraise=True,
)
def _call_dalle(client: OpenAI, prompt: str) -> bytes:
    response = client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size="1792x1024",
        quality="hd",
        n=1,
    )
    url = response.data[0].url
    dl  = requests.get(url, timeout=60)
    dl.raise_for_status()
    return dl.content


_BRAND_BACKGROUND_TEMPLATES: dict[str, str] = {
    "openai": (
        "photorealistic editorial AI newsroom background for an OpenAI story, "
        "modern research lab with subtle blue lighting and an abstract data display wall, "
        "no logos, no people, serious tone"
    ),
    "anthropic": (
        "photorealistic editorial AI workspace background for an Anthropic story, "
        "calm cream-and-blue research office with a schematic data wall, "
        "no logos, no people, serious tone"
    ),
    "google": (
        "photorealistic editorial technology newsroom background for a Google story, "
        "glass-walled innovation lab with subtle color accents and data visualizations, "
        "no logos, no people, serious tone"
    ),
}


def _normalize_visual_prompt(prompt: str) -> str:
    """Normalize visual prompt style and apply brand-backed backgrounds when appropriate."""
    prompt = prompt.strip()
    lower = prompt.lower()
    for company, template in _BRAND_BACKGROUND_TEMPLATES.items():
        if company in lower:
            return f"{template} — {prompt}"

    prompt = re.sub(r"style:\s*vivid", "style: photorealistic", prompt, flags=re.IGNORECASE)
    if "photorealistic" not in prompt.lower() and "photo-realistic" not in prompt.lower():
        prompt = f"photorealistic {prompt}"
    return prompt


def generate_dalle_image(prompt: str, out_path: Path) -> Path:
    """Call DALL-E 3 HD and save the image. Cached if already exists."""
    prompt = _normalize_visual_prompt(prompt)
    if out_path.exists():
        logger.info(f"Reusing cached image: {out_path.name}")
        return out_path
    client = make_openai_client()
    logger.info(f"DALL-E 3 HD: {prompt[:80]}…")
    data = _call_dalle(client, prompt)
    out_path.write_bytes(data)
    get_ledger().record_image(f"image-{out_path.stem}", "dall-e-3")
    logger.info(f"  → {out_path.name} ({len(data)//1024} KB)")
    return out_path


# ── Stable Diffusion (Stability AI) ──────────────────────────────────────────

def generate_sd_image(prompt: str, out_path: Path) -> Path:
    """Generate via Stability AI stable-image-core (16:9, ~$0.003/image).

    Requires STABILITY_API_KEY.  Raises RuntimeError if the key is absent or
    the API returns an error so the caller can fall back to DALL-E.
    """
    key = settings.stability_api_key
    if not key:
        raise RuntimeError("STABILITY_API_KEY not configured")

    logger.info(f"Stable Diffusion (breaking): {prompt[:80]}…")
    resp = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/core",
        headers={"authorization": f"Bearer {key}", "accept": "image/*"},
        files={
            "prompt":          (None, prompt[:2000]),
            "output_format":   (None, "png"),
            "aspect_ratio":    (None, "16:9"),
            "negative_prompt": (None, "text, watermark, logo, blurry, low quality, distorted"),
        },
        timeout=90,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"Stability AI {resp.status_code}: {resp.text[:200]}")

    out_path.write_bytes(resp.content)
    logger.info(f"  → {out_path.name} (SD, {len(resp.content)//1024} KB)")
    return out_path


# ── Stock photos (Unsplash → Pexels photo) ───────────────────────────────────

_PEXELS_PHOTO_URL  = "https://api.pexels.com/v1/search"
_UNSPLASH_PHOTO_URL = "https://api.unsplash.com/search/photos"

# Regex to strip opening phrases from DALL-E prompts before using as stock query
_PREAMBLE_RE = re.compile(
    r"^(?:a |an )?(?:photorealistic |digital |cinematic |illustration of |"
    r"render(?:ing)? of |photo of |image of |picture of |scene of )",
    re.IGNORECASE,
)

# Prompt words that suggest generic real-world scenes (stock-searchable)
_GENERIC_WORDS = {
    "person", "people", "man", "woman", "team", "group", "crowd",
    "office", "workspace", "desk", "laptop", "computer", "keyboard",
    "meeting", "conference", "presentation", "whiteboard", "screen",
    "server", "data center", "datacenter", "network", "cables",
    "smartphone", "phone", "tablet", "device",
    "hand", "hands", "finger", "typing",
    "building", "cityscape", "technology", "background",
    "researcher", "scientist", "engineer", "developer", "coder",
}

# Prompt words that mark AI-specific / abstract scenes not in stock libraries
_SPECIFIC_WORDS = {
    "dall-e", "gpt", "claude", "gemini", "llama", "anthropic", "openai",
    "diagram", "chart", "graph", "infographic", "neural network",
    "robot arm", "humanoid", "avatar", "holographic", "hologram",
    "brain", "circuit board", "code snippet", "terminal", "command line",
    "pixel art", "abstract art", "concept art", "futuristic",
}


def _is_generic_prompt(prompt: str) -> bool:
    """True when the prompt describes a real-world scene searchable in stock libraries."""
    lower = prompt.lower()
    generic_hits  = sum(1 for w in _GENERIC_WORDS  if w in lower)
    specific_hits = sum(1 for w in _SPECIFIC_WORDS if w in lower)
    return generic_hits >= 2 and specific_hits == 0


def _photo_query(visual_prompt: str) -> str:
    """Extract a short stock-search query from a DALL-E visual_prompt."""
    q = _PREAMBLE_RE.sub("", visual_prompt)
    q = re.split(r"[.,;]", q)[0]   # take first clause
    return q[:60].strip()


def _unsplash_photo(query: str, out_path: Path) -> Path | None:
    key = settings.unsplash_access_key
    if not key:
        return None
    try:
        r = requests.get(
            _UNSPLASH_PHOTO_URL,
            headers={"Authorization": f"Client-ID {key}"},
            params={"query": query, "orientation": "landscape", "per_page": 5},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None
        photo = results[0]
        img_url = photo["urls"].get("regular") or photo["urls"]["full"]
        img_r = requests.get(img_url, timeout=60)
        img_r.raise_for_status()
        out_path.write_bytes(img_r.content)
        user = photo.get("user", {}).get("name", "Unknown")
        logger.info(f"  Stock photo (Unsplash): '{query}' by {user} → {out_path.name}")
        return out_path
    except Exception as exc:
        logger.warning(f"Unsplash photo failed for '{query}': {exc}")
        return None


def _pexels_photo(query: str, out_path: Path) -> Path | None:
    key = settings.pexels_api_key
    if not key:
        return None
    try:
        r = requests.get(
            _PEXELS_PHOTO_URL,
            headers={"Authorization": key},
            params={"query": query, "orientation": "landscape", "size": "medium", "per_page": 5},
            timeout=15,
        )
        r.raise_for_status()
        photos = r.json().get("photos", [])
        if not photos:
            return None
        src = photos[0].get("src", {})
        img_url = src.get("large2x") or src.get("large") or src.get("original")
        if not img_url:
            return None
        img_r = requests.get(img_url, timeout=60)
        img_r.raise_for_status()
        out_path.write_bytes(img_r.content)
        photographer = photos[0].get("photographer", "Unknown")
        logger.info(f"  Stock photo (Pexels): '{query}' by {photographer} → {out_path.name}")
        return out_path
    except Exception as exc:
        logger.warning(f"Pexels photo failed for '{query}': {exc}")
        return None


def fetch_stock_photo(visual_prompt: str, out_path: Path) -> Path | None:
    """Search Unsplash (primary) then Pexels (fallback) for a landscape photo.

    Returns *out_path* on success, None if no stock photo is found.
    """
    query = _photo_query(visual_prompt)
    return _unsplash_photo(query, out_path) or _pexels_photo(query, out_path)


# ── Ken Burns effect ──────────────────────────────────────────────────────────

def _ken_burns_clip(img_path: Path, duration: float, clip_path: Path) -> Path:
    """Animate a static image with one of four Ken Burns movements via ffmpeg.

    Input dimensions are read from the image itself so the effect works
    for any image source (DALL-E 1792×1024, SD 1344×768, stock photos, etc.).
    """
    stem_last = img_path.stem.split("_")[-1]
    variant   = int(stem_last) % 4 if stem_last.isdigit() else (hash(img_path.stem) % 4)

    with PILImage.open(img_path) as pil_img:
        in_w, in_h = pil_img.size

    return ffmpeg_utils.ken_burns(
        img_path,
        clip_path,
        duration_sec=duration,
        variant=variant,
        in_w=in_w,
        in_h=in_h,
        out_w=OUT_W,
        out_h=OUT_H,
    )


# ── Fallback ──────────────────────────────────────────────────────────────────

def _black_placeholder(path: Path, duration: int) -> Path:
    return ffmpeg_utils.black_clip(path, width=OUT_W, height=OUT_H, duration_sec=float(duration))


def _valid_video_clip(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        w, h = ffmpeg_utils.video_size(path)
        dur = ffmpeg_utils.duration(path)
        return w > 0 and h > 0 and dur > 0
    except Exception:
        return False


# ── Screenshot helper ─────────────────────────────────────────────────────────

def _try_real_screenshot(scene: Scene, tool: str | None, img_path: Path) -> bool:
    if not tool or not scene.screenshot_key:
        return False
    try:
        from src.screenshot_capturer import capture, has_key
        if not has_key(tool, scene.screenshot_key):
            return False
        capture(tool, scene.screenshot_key, img_path)
        return img_path.exists()
    except Exception as e:
        logger.warning(
            f"Scene {scene.idx} screenshot failed ({tool}/{scene.screenshot_key}): {e} "
            "— falling back to image generation"
        )
        return False


# ── Public API ────────────────────────────────────────────────────────────────

def generate_scene_clip(
    scene: Scene,
    clip_dir: Path,
    *,
    tool: str | None = None,
    run_dir: Path | None = None,
    is_breaking: bool = False,
) -> Path:
    """
    Produce a single .mp4 clip for *scene* at exactly scene.duration_sec length.

    Priority order:
      1. infographic_data — animated chart/stat rendered entirely in Python
      2. video_query      — stock B-roll video from Pexels / Pixabay
      3. screenshot_key   — real UI screenshot with Ken Burns (weekly tutorials)
      4. image_cache      — cosine-similar prompt from a previous run   (~$0.000006)
      5. stock_photo      — Unsplash / Pexels photo for generic scenes   (free)
      6. stable_diffusion — Stability AI for breaking-news clips          (~$0.003)
      7. dall_e_3         — DALL-E 3 HD for all other scenes              (~$0.080)
      8. placeholder      — solid dark frame if everything fails

    *is_breaking* activates Stable Diffusion as the default generator (step 6).
    """
    img_dir  = clip_dir.parent / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    img_path  = img_dir  / f"scene_{scene.idx:02d}.png"
    clip_path = clip_dir / f"scene_{scene.idx:02d}_clip_0.mp4"

    if clip_path.exists():
        if _valid_video_clip(clip_path):
            logger.info(f"Reusing cached clip: {clip_path.name}")
            return clip_path
        logger.warning(
            f"Scene {scene.idx}: cached clip is invalid, removing and regenerating"
        )
        clip_path.unlink(missing_ok=True)

    data_dir = settings.data_dir

    try:
        # 1. Infographic — fully animated
        if scene.infographic_data:
            from src.infographic_generator import generate_infographic_clip
            return generate_infographic_clip(
                scene.infographic_data,
                clip_path,
                duration_sec=float(max(scene.duration_sec, 4)),
            )

        # 2. Stock video B-roll
        if scene.video_query:
            from src.broll_fetcher import fetch_broll_clip
            result = fetch_broll_clip(scene, clip_path, run_dir=run_dir)
            if result:
                return result
            logger.info(f"Scene {scene.idx}: B-roll unavailable — trying image sources")

        # 3 & 4+. Image path: screenshot → cache → stock photo → SD / DALL-E
        if not img_path.exists():
            embedding = None  # carry from cache lookup to add_entry

            if not _try_real_screenshot(scene, tool, img_path):
                # 4. Cross-run similarity cache
                cached, embedding = image_cache.find_similar(
                    scene.visual_prompt, data_dir
                )
                if cached:
                    shutil.copy2(str(cached), str(img_path))
                else:
                    # 5. Stock photo for generic real-world scenes
                    generated = False
                    if _is_generic_prompt(scene.visual_prompt):
                        result = fetch_stock_photo(scene.visual_prompt, img_path)
                        if result:
                            generated = True
                            logger.info(f"Scene {scene.idx}: stock photo used (saved ~$0.08)")

                    # 6. Stable Diffusion for breaking-news clips
                    if not generated and is_breaking and settings.stability_api_key:
                        try:
                            generate_sd_image(scene.visual_prompt, img_path)
                            generated = True
                        except Exception as exc:
                            logger.warning(
                                f"Scene {scene.idx}: SD failed ({exc}) — falling back to DALL-E"
                            )

                    # 7. DALL-E 3 HD — universal fallback
                    if not generated:
                        generate_dalle_image(scene.visual_prompt, img_path)

                    # Persist in cross-run cache (skip cache-hit copies)
                    if img_path.exists():
                        image_cache.add_entry(
                            scene.visual_prompt, img_path, data_dir,
                            embedding=embedding,
                        )
        else:
            logger.info(f"Reusing cached image: {img_path.name}")

        result = _ken_burns_clip(img_path, float(scene.duration_sec), clip_path)
        if not _valid_video_clip(result):
            logger.warning(
                f"Scene {scene.idx}: generated clip invalid, falling back to placeholder"
            )
            return _black_placeholder(clip_path, scene.duration_sec)
        logger.info(f"Scene {scene.idx} clip written: {clip_path.name}")
        return result

    except Exception as e:
        logger.error(f"Scene {scene.idx} clip failed, using placeholder: {e}")
        return _black_placeholder(clip_path, scene.duration_sec)
