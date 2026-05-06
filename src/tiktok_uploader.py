"""Post short videos to TikTok via the Content Posting API v2.

First-run auth (one-time, run locally):
    python src/tiktok_uploader.py --auth

This opens a browser, completes the OAuth2 + PKCE flow, and saves
tokens to config/tiktok_token.json.  Then encode for GitHub Actions:
    base64 -w0 config/tiktok_token.json   →  TIKTOK_TOKEN_JSON_B64 secret

Subsequent CI runs decode the secret into config/tiktok_token.json,
then auto-refresh the access token (24h TTL) using the stored
refresh_token (365-day TTL).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import secrets
import sys
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.retry_utils import http_post

_AUTH_URL   = "https://www.tiktok.com/v2/auth/authorize/"
_TOKEN_URL  = "https://open.tiktokapis.com/v2/oauth/token/"
_UPLOAD_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
_CHUNK_SIZE = 10 * 1024 * 1024   # 10 MB per chunk
_SCOPES     = "video.publish,video.upload,user.info.basic,video.list"
_REDIRECT   = "http://localhost:8080/callback"


# ─────────────────── Token management ───────────────────

def _load_tokens(token_file: Path) -> dict:
    return json.loads(token_file.read_text())


def _save_tokens(token_file: Path, tokens: dict) -> None:
    token_file.write_text(json.dumps(tokens, indent=2))


def _do_refresh(client_key: str, client_secret: str, refresh_token: str) -> dict:
    resp = http_post(
        _TOKEN_URL,
        data={
            "client_key":    client_key,
            "client_secret": client_secret,
            "grant_type":    "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    return resp.json()


def _get_access_token(client_key: str, client_secret: str, token_file: Path) -> str:
    """Return a valid access token, refreshing automatically when expired."""
    tokens = _load_tokens(token_file)

    # Refresh 5 minutes before actual expiry to avoid mid-upload failures
    if time.time() > tokens.get("expires_at", 0) - 300:
        logger.info("TikTok: access token expired — refreshing")
        fresh = _do_refresh(client_key, client_secret, tokens["refresh_token"])
        tokens["access_token"]  = fresh["access_token"]
        tokens["refresh_token"] = fresh.get("refresh_token", tokens["refresh_token"])
        tokens["expires_at"]    = time.time() + fresh.get("expires_in", 86400)
        _save_tokens(token_file, tokens)
        logger.info("TikTok: token refreshed and saved")

    return tokens["access_token"]


# ─────────────────── First-run OAuth2 ───────────────────

def run_auth_flow(client_key: str, client_secret: str, token_file: Path) -> None:
    """Interactive OAuth2 + PKCE flow. Opens browser, starts local redirect server."""
    code_verifier  = secrets.token_urlsafe(64)
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode()).digest()
    ).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)

    auth_url = _AUTH_URL + "?" + urlencode({
        "client_key":            client_key,
        "scope":                 _SCOPES,
        "response_type":         "code",
        "redirect_uri":          _REDIRECT,
        "state":                 state,
        "code_challenge":        code_challenge,
        "code_challenge_method": "S256",
    })

    code_holder: dict = {}

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            qs = parse_qs(urlparse(self.path).query)
            if "code" in qs:
                code_holder["code"] = qs["code"][0]
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"<h2>TikTok auth complete - you can close this tab.</h2>")

        def log_message(self, *_):
            pass

    server = HTTPServer(("localhost", 8080), _Handler)
    Thread(target=server.handle_request, daemon=True).start()

    print(f"\nOpening browser for TikTok OAuth2...\n{auth_url}\n")
    webbrowser.open(auth_url)

    while "code" not in code_holder:
        time.sleep(0.2)

    resp = http_post(
        _TOKEN_URL,
        data={
            "client_key":    client_key,
            "client_secret": client_secret,
            "code":          code_holder["code"],
            "grant_type":    "authorization_code",
            "redirect_uri":  _REDIRECT,
            "code_verifier": code_verifier,
        },
        timeout=30,
    )
    tokens = resp.json()
    tokens["expires_at"] = time.time() + tokens.get("expires_in", 86400)

    token_file.parent.mkdir(parents=True, exist_ok=True)
    _save_tokens(token_file, tokens)
    logger.info(f"TikTok tokens saved → {token_file}")


# ─────────────────── Upload ───────────────────

def _init_upload(
    access_token: str,
    caption: str,
    video_size: int,
    privacy: str,
) -> tuple[str, str]:
    """Start a TikTok direct-post upload. Returns (publish_id, upload_url)."""
    total_chunks = math.ceil(video_size / _CHUNK_SIZE)
    resp = http_post(
        _UPLOAD_URL,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type":  "application/json; charset=UTF-8",
        },
        json={
            "post_info": {
                "title":                    caption[:2200],
                "privacy_level":            privacy,
                "disable_duet":             False,
                "disable_comment":          False,
                "disable_stitch":           False,
                "video_cover_timestamp_ms": 1000,
            },
            "source_info": {
                "source":            "FILE_UPLOAD",
                "video_size":        video_size,
                "chunk_size":        _CHUNK_SIZE,
                "total_chunk_count": total_chunks,
            },
        },
        timeout=30,
    )
    data = resp.json().get("data", {})
    return data["publish_id"], data["upload_url"]


def _upload_chunks(upload_url: str, video_path: Path) -> None:
    video_size = video_path.stat().st_size
    with open(video_path, "rb") as fh:
        for part in range(math.ceil(video_size / _CHUNK_SIZE)):
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            start = part * _CHUNK_SIZE
            end   = start + len(chunk) - 1
            resp  = requests.put(
                upload_url,
                headers={
                    "Content-Range":  f"bytes {start}-{end}/{video_size}",
                    "Content-Length": str(len(chunk)),
                    "Content-Type":   "video/mp4",
                },
                data=chunk,
                timeout=120,
            )
            resp.raise_for_status()
            logger.info(f"  TikTok chunk {part}: {end + 1}/{video_size} bytes")


def post_short(
    video_path: Path,
    caption: str,
    *,
    client_key: str | None = None,
    client_secret: str | None = None,
    token_file: Path | None = None,
    privacy: str | None = None,
) -> str:
    """Upload *video_path* to TikTok as a Short. Returns the publish_id."""
    ck = client_key    or settings.tiktok_client_key
    cs = client_secret or settings.tiktok_client_secret
    tf = token_file    or settings.tiktok_token_file
    pv = privacy       or settings.tiktok_privacy

    access_token = _get_access_token(ck, cs, tf)
    video_size   = video_path.stat().st_size

    logger.info(f"TikTok: uploading {video_path.name} ({video_size / 1e6:.1f} MB)")
    publish_id, upload_url = _init_upload(access_token, caption, video_size, pv)
    logger.info(f"TikTok publish_id: {publish_id}")
    _upload_chunks(upload_url, video_path)
    logger.info(f"TikTok upload complete: {publish_id}")
    return publish_id


# ─────────────────── CLI ───────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="TikTok uploader for AI News Shorts")
    ap.add_argument("--auth", action="store_true", help="Run OAuth2 flow and save token")
    args = ap.parse_args()

    if args.auth:
        if not settings.tiktok_client_key or not settings.tiktok_client_secret:
            print("Set TIKTOK_CLIENT_KEY and TIKTOK_CLIENT_SECRET env vars first.")
            sys.exit(1)
        run_auth_flow(
            settings.tiktok_client_key,
            settings.tiktok_client_secret,
            settings.tiktok_token_file,
        )
        print(f"\nToken saved to: {settings.tiktok_token_file}")
        print("Encode for GitHub Actions secret (TIKTOK_TOKEN_JSON_B64):")
        print(f"  base64 -w0 {settings.tiktok_token_file}")


if __name__ == "__main__":
    main()
