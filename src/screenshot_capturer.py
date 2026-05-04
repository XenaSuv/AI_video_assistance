"""Capture screenshots of public AI tool pages for use as scene visuals.

For weekly tutorials, real screenshots of the actual product pages
(claude.com, chat.openai.com, gemini.google.com, their docs and consoles)
look far more authentic than DALL-E mockups of UI.

This module keeps a curated library of public, login-free URLs grouped
by tool key.  Playwright captures each URL at 1792x1024 — the same size
as the DALL-E 3 images — so the existing Ken Burns pan pipeline works
unchanged.

Robustness:
- The screenshot library and capture logic are tool-agnostic and lazy:
  Playwright is imported only when capture() is actually called, so the
  rest of the pipeline runs fine without the dependency installed.
- Any failure (Playwright missing, network error, timeout) raises an
  exception that the caller (image_generator) catches to fall back to
  DALL-E generation.

Usage from the pipeline:
    from src.screenshot_capturer import capture, has_key
    if has_key(tool, scene.screenshot_key):
        capture(tool, scene.screenshot_key, img_path)
"""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CAPTURE_W, CAPTURE_H = 1792, 1024
_NAV_TIMEOUT_MS  = 30_000
_SETTLE_DELAY_MS = 2_500


# Curated library of public, login-free URLs grouped by tool.
# Keys must match what GPT picks from in the tutorial system prompt.
# Adding a new key here automatically makes it selectable in scripts.
_LIBRARY: dict[str, dict[str, dict]] = {
    "claude": {
        "homepage":          {"url": "https://www.claude.com"},
        "claude_login":      {"url": "https://claude.ai/login"},
        "anthropic_home":    {"url": "https://www.anthropic.com"},
        "pricing":           {"url": "https://www.anthropic.com/pricing"},
        "models_overview":   {"url": "https://docs.claude.com/en/docs/about-claude/models/overview"},
        "api_intro":         {"url": "https://docs.claude.com/en/docs/intro"},
        "api_quickstart":    {"url": "https://docs.claude.com/en/docs/get-started"},
        "claude_code":       {"url": "https://docs.claude.com/en/docs/claude-code/overview"},
        "tool_use":          {"url": "https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview"},
        "prompt_caching":    {"url": "https://docs.claude.com/en/docs/build-with-claude/prompt-caching"},
        "vision":            {"url": "https://docs.claude.com/en/docs/build-with-claude/vision"},
        "extended_thinking": {"url": "https://docs.claude.com/en/docs/build-with-claude/extended-thinking"},
        "mcp":               {"url": "https://docs.claude.com/en/docs/agents-and-tools/mcp"},
        "console":           {"url": "https://console.anthropic.com"},
    },
    "chatgpt": {
        "homepage":           {"url": "https://chat.openai.com"},
        "openai_home":        {"url": "https://openai.com"},
        "api_pricing":        {"url": "https://openai.com/api/pricing"},
        "platform_docs":      {"url": "https://platform.openai.com/docs/overview"},
        "models_overview":    {"url": "https://platform.openai.com/docs/models"},
        "function_calling":   {"url": "https://platform.openai.com/docs/guides/function-calling"},
        "structured_outputs": {"url": "https://platform.openai.com/docs/guides/structured-outputs"},
        "vision":             {"url": "https://platform.openai.com/docs/guides/vision"},
        "assistants_api":     {"url": "https://platform.openai.com/docs/assistants/overview"},
        "fine_tuning":        {"url": "https://platform.openai.com/docs/guides/fine-tuning"},
        "images_guide":       {"url": "https://platform.openai.com/docs/guides/images"},
        "gpt_store":          {"url": "https://chat.openai.com/gpts"},
    },
    "gemini": {
        "homepage":         {"url": "https://gemini.google.com"},
        "ai_studio":        {"url": "https://aistudio.google.com"},
        "api_docs":         {"url": "https://ai.google.dev/gemini-api/docs"},
        "api_quickstart":   {"url": "https://ai.google.dev/gemini-api/docs/quickstart"},
        "models_overview":  {"url": "https://ai.google.dev/gemini-api/docs/models"},
        "function_calling": {"url": "https://ai.google.dev/gemini-api/docs/function-calling"},
        "long_context":     {"url": "https://ai.google.dev/gemini-api/docs/long-context"},
        "vision":           {"url": "https://ai.google.dev/gemini-api/docs/vision"},
        "audio":             {"url": "https://ai.google.dev/gemini-api/docs/audio"},
        "embeddings":       {"url": "https://ai.google.dev/gemini-api/docs/embeddings"},
        "pricing":          {"url": "https://ai.google.dev/pricing"},
        "vertex_ai":        {"url": "https://cloud.google.com/vertex-ai"},
        "notebooklm":       {"url": "https://notebooklm.google"},
    },
}


# ─────────────────── Public lookup helpers ───────────────────

def available_keys(tool: str) -> list[str]:
    """All screenshot keys configured for *tool* (empty list if unknown tool)."""
    return list(_LIBRARY.get(tool, {}).keys())


def has_key(tool: str, key: str | None) -> bool:
    """True if a screenshot URL exists for the given tool/key combination."""
    if not key:
        return False
    return key in _LIBRARY.get(tool, {})


def url_for(tool: str, key: str) -> str:
    return _LIBRARY[tool][key]["url"]


# ─────────────────── Capture ───────────────────

# Cookie-banner buttons we attempt to click after page load. Best-effort —
# silent failures are fine; the screenshot is still usable with a banner.
_DISMISS_SELECTORS = (
    "button:has-text('Accept all')",
    "button:has-text('Accept')",
    "button:has-text('I agree')",
    "button:has-text('Got it')",
    "button:has-text('Allow all')",
    "[aria-label='Accept all']",
)


def capture(tool: str, key: str, out_path: Path) -> Path:
    """Capture *tool*/*key* to *out_path* (1792x1024 JPEG). Cached if exists.

    Raises ValueError on unknown key, RuntimeError if Playwright fails.
    """
    if not has_key(tool, key):
        raise ValueError(f"Unknown screenshot key: {tool}/{key}")

    if out_path.exists():
        logger.info(f"Reusing cached screenshot: {out_path.name}")
        return out_path

    entry  = _LIBRARY[tool][key]
    url    = entry["url"]
    scroll = int(entry.get("scroll", 0))

    logger.info(f"Screenshot: [{tool}/{key}] {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import so the rest of the pipeline runs without Playwright installed.
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError("Playwright not installed; install with `pip install playwright`") from e

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(
            viewport={"width": CAPTURE_W, "height": CAPTURE_H},
            device_scale_factor=1,
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()
        try:
            page.goto(url, timeout=_NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            page.wait_for_timeout(_SETTLE_DELAY_MS)

            for selector in _DISMISS_SELECTORS:
                try:
                    btn = page.query_selector(selector)
                    if btn:
                        btn.click(timeout=1_000)
                        page.wait_for_timeout(300)
                        break
                except Exception:
                    pass

            if scroll > 0:
                page.evaluate(f"window.scrollTo(0, {scroll})")
                page.wait_for_timeout(500)

            page.screenshot(path=str(out_path), full_page=False, type="jpeg", quality=92)
        finally:
            context.close()
            browser.close()

    logger.info(f"  -> {out_path.name} ({out_path.stat().st_size // 1024} KB)")
    return out_path


def capture_url(url: str, out_path: Path, scroll: int = 0) -> Path:
    """Capture any public URL to *out_path* (1792x1024 JPEG). Cached if exists.

    Used by topic segment pipeline for arbitrary AI tool pages.
    Raises RuntimeError if Playwright fails or times out.
    """
    if out_path.exists():
        logger.info(f"Reusing cached screenshot: {out_path.name}")
        return out_path

    logger.info(f"Screenshot (dynamic): {url}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError("Playwright not installed; run `pip install playwright && playwright install chromium`") from e

    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = browser.new_context(
            viewport={"width": CAPTURE_W, "height": CAPTURE_H},
            device_scale_factor=1,
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
            ),
            locale="en-US",
        )
        page = context.new_page()
        try:
            page.goto(url, timeout=_NAV_TIMEOUT_MS, wait_until="domcontentloaded")
            page.wait_for_timeout(_SETTLE_DELAY_MS)

            for selector in _DISMISS_SELECTORS:
                try:
                    btn = page.query_selector(selector)
                    if btn:
                        btn.click(timeout=1_000)
                        page.wait_for_timeout(300)
                        break
                except Exception:
                    pass

            if scroll > 0:
                page.evaluate(f"window.scrollTo(0, {scroll})")
                page.wait_for_timeout(500)

            page.screenshot(path=str(out_path), full_page=False, type="jpeg", quality=92)
        finally:
            context.close()
            browser.close()

    logger.info(f"  -> {out_path.name} ({out_path.stat().st_size // 1024} KB)")
    return out_path


# ─────────────────── CLI for quick local testing ───────────────────

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Capture a screenshot from the curated library")
    ap.add_argument("tool", choices=list(_LIBRARY.keys()))
    ap.add_argument("key",  help="Screenshot key (use --list to see options)")
    ap.add_argument("--out", default=None, help="Output JPEG path (default: ./{tool}_{key}.jpg)")
    ap.add_argument("--list", action="store_true", help="List available keys for the tool and exit")
    args = ap.parse_args()

    if args.list:
        for k in available_keys(args.tool):
            print(f"  {k:<22}  {url_for(args.tool, k)}")
        sys.exit(0)

    out = Path(args.out or f"./{args.tool}_{args.key}.jpg")
    capture(args.tool, args.key, out)
    print(f"Saved -> {out}")
