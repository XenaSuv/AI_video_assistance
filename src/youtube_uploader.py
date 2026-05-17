"""Upload long-form video + Shorts to YouTube via Data API v3.

First run: python youtube_uploader.py --auth
This does the OAuth2 desktop flow once and caches the refresh token.
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import time
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from google.auth.exceptions import RefreshError
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src import ffmpeg_utils
from src.circuit_breaker import youtube_breaker
from src.exceptions import QuotaExhaustedError
from src.script_generator import VideoScript
from src.youtube_quota_guard import YouTubeQuotaGuard


SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    # Required for thumbnails.set and captions.insert
    "https://www.googleapis.com/auth/youtube.force-ssl",
    # Required for YouTube Analytics API (retention curves, view metrics)
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

_CAPTION_TRACK_NAMES: dict[str, str] = {
    "en": "English",
    "ru": "Русский",
    "de": "Deutsch",
    "fr": "Français",
    "es": "Español",
}


def _migrate_pickle(pickle_path: Path, json_path: Path) -> None:
    """One-time migration from legacy pickle token to JSON. Deletes the pickle file."""
    logger.warning(
        "Migrating legacy pickle token %s → %s",
        pickle_path.name, json_path.name,
    )
    raw = pickle_path.read_bytes()
    try:
        import pickle as _pickle  # scoped import — only used during migration
        creds = _pickle.load(__import__("io").BytesIO(raw))
        token_json = creds.to_json()
    except Exception:
        # File was already written as JSON (e.g. saved with wrong extension).
        token_json = raw.decode()
        json.loads(token_json)  # validate — raises if truly corrupt
    json_path.write_text(token_json)
    pickle_path.unlink()
    logger.info("Token migration complete. Update YOUTUBE_TOKEN_FILE to %s in your env.", json_path.name)


def _get_creds(
    client_secrets: Path | None = None,
    token_file: Path | None = None,
    force_browser: bool = False,
) -> Credentials:
    client_secrets = client_secrets or settings.youtube_client_secrets
    token_file     = token_file     or settings.youtube_token_file

    # Auto-migrate: env var still points to .pickle → switch to .json sibling.
    if token_file.suffix == ".pickle":
        json_path = token_file.with_suffix(".json")
        if token_file.exists():
            _migrate_pickle(token_file, json_path)
        token_file = json_path
    elif not token_file.exists():
        # Expected .json but not found — check for legacy .pickle sibling.
        pickle_sibling = token_file.with_suffix(".pickle")
        if pickle_sibling.exists():
            _migrate_pickle(pickle_sibling, token_file)

    creds: Credentials | None = None
    if token_file.exists():
        try:
            creds = Credentials.from_authorized_user_info(
                json.loads(token_file.read_text()), SCOPES
            )
        except Exception as exc:
            logger.warning("Failed to load token from %s: %s. Will re-authenticate.", token_file, exc)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_file.write_text(creds.to_json())
        except RefreshError as exc:
            headless = (
                os.getenv("CI", "").lower() in ("1", "true")
                or os.getenv("GITHUB_ACTIONS", "").lower() == "true"
                or not os.getenv("DISPLAY")
            ) and not force_browser
            if headless:
                raise RuntimeError(
                    "YouTube refresh token expired and cannot re-authorize in headless CI environment. "
                    "Please update the token manually by running 'python src/youtube_uploader.py --auth --force' locally, "
                    "then update the YOUTUBE_TOKEN_JSON_B64 GitHub secret."
                ) from exc
            logger.warning(
                "Existing YouTube token refresh failed: {}. "
                "Removing stale token and re-running OAuth flow.",
                exc,
            )
            creds = None
            try:
                token_file.unlink()
            except OSError:
                pass

    if not creds or not creds.valid:
        if not client_secrets.exists():
            raise FileNotFoundError(
                f"Missing {client_secrets}. "
                "Download OAuth client_secrets.json from Google Cloud Console."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)

        in_ci = os.getenv("CI", "").lower() in ("1", "true") or os.getenv("GITHUB_ACTIONS") == "true"
        in_codespaces = os.getenv("CODESPACES") == "true"

        if in_ci and not force_browser:
            # True CI — no interactive user present
            raise RuntimeError(
                "YouTube OAuth requires interactive login, which is not possible in CI. "
                "Run 'python src/youtube_uploader.py --auth --force' locally or in Codespaces, "
                "then upload the token JSON as the YOUTUBE_TOKEN_JSON_B64 GitHub secret."
            )
        elif in_codespaces or not os.getenv("DISPLAY"):
            # Codespaces or any headless environment with a real user:
            # use a fixed port so VS Code / Codespaces can forward it predictably.
            _OAUTH_PORT = 8080
            codespace_name = os.getenv("CODESPACE_NAME", "")
            if codespace_name:
                forwarded = f"https://{codespace_name}-{_OAUTH_PORT}.app.github.dev"
                logger.info(
                    "GitHub Codespaces detected.\n"
                    "  1. The OAuth server will start on port %d.\n"
                    "  2. Open the PORTS tab in VS Code and make port %d 'Public' if prompted.\n"
                    "  3. Complete the auth in your browser — Google will redirect to\n"
                    "     http://localhost:%d which Codespaces tunnels automatically.\n"
                    "  Forwarded URL (for reference): %s",
                    _OAUTH_PORT, _OAUTH_PORT, _OAUTH_PORT, forwarded,
                )
            else:
                logger.info(
                    "Headless environment detected. OAuth server starting on port %d. "
                    "Open the auth URL in your browser and complete the login.",
                    _OAUTH_PORT,
                )
            creds = flow.run_local_server(port=_OAUTH_PORT, open_browser=False)
        else:
            try:
                creds = flow.run_local_server(port=0)
            except Exception as exc:
                logger.warning(
                    "Unable to launch local browser for OAuth: %s. "
                    "Falling back to no-browser mode.",
                    exc,
                )
                creds = flow.run_local_server(port=0, open_browser=False)
        token_file.write_text(creds.to_json())
    return creds


def _youtube_client(
    client_secrets: Path | None = None,
    token_file: Path | None = None,
):
    return build(
        "youtube", "v3",
        credentials=_get_creds(client_secrets, token_file),
        cache_discovery=False,
    )


# --------------------- Upload helpers ---------------------

_RETRYABLE_STATUS = (500, 502, 503, 504)
_MAX_RETRIES = 5


def _upload_with_retry(request: "Any", video_path: Path, max_retries: int = _MAX_RETRIES) -> dict:
    """Execute a resumable upload with exponential backoff and cross-run state persistence.

    State (upload URI + byte offset) is written to *video_path*.upload_state.json
    after every successful chunk so the upload can be resumed if the process is
    restarted mid-way.  The state file is deleted on success or fatal error.
    """
    state_path = video_path.with_name(video_path.name + ".upload_state.json")

    # Restore a previous partial upload if one exists.
    if state_path.exists():
        try:
            saved = json.loads(state_path.read_text())
            request.resumable_uri      = saved["resumable_uri"]
            request.resumable_progress = int(saved.get("resumable_progress", 0))
            logger.info(
                f"Resuming upload from {request.resumable_progress / 1e6:.1f} MB "
                f"(saved URI found)"
            )
        except Exception as exc:
            logger.warning(f"Could not restore upload state, starting fresh: {exc}")
            state_path.unlink(missing_ok=True)

    http_retries = 0
    net_retries  = 0
    response     = None

    while response is None:
        try:
            status, response = request.next_chunk()
            http_retries = 0
            net_retries  = 0

            # Persist URI + progress so we can resume after a crash.
            if getattr(request, "resumable_uri", None):
                state_path.write_text(json.dumps({
                    "resumable_uri":      request.resumable_uri,
                    "resumable_progress": getattr(request, "resumable_progress", 0),
                }))

            if status:
                pct = int(status.progress() * 100)
                mb  = getattr(request, "resumable_progress", 0) / 1e6
                logger.info(f"  Upload {pct}% ({mb:.0f} MB)")

        except HttpError as exc:
            code = exc.resp.status
            if code in _RETRYABLE_STATUS:
                http_retries += 1
                if http_retries > max_retries:
                    logger.error(
                        f"Upload failed after {max_retries} retries (HTTP {code})"
                    )
                    state_path.unlink(missing_ok=True)
                    raise
                wait = 2 ** http_retries
                logger.warning(
                    f"HTTP {code} from YouTube, retry {http_retries}/{max_retries} in {wait}s"
                )
                time.sleep(wait)

            elif code == 429:
                http_retries += 1
                if http_retries > max_retries:
                    state_path.unlink(missing_ok=True)
                    raise
                retry_after = exc.resp.get("retry-after")
                wait = int(retry_after) if retry_after else min(2 ** http_retries, 64)
                logger.warning(f"Rate-limited by YouTube (429), waiting {wait}s")
                time.sleep(wait)

            else:
                # 4xx (auth, quota exceeded, etc.) — not retryable.
                state_path.unlink(missing_ok=True)
                raise

        except (TimeoutError, ConnectionError, OSError) as exc:
            net_retries += 1
            if net_retries > max_retries:
                logger.error(f"Upload failed after {max_retries} network errors")
                raise
            wait = 2 ** net_retries
            logger.warning(
                f"Network error ({type(exc).__name__}), retry {net_retries}/{max_retries} in {wait}s"
            )
            time.sleep(wait)

    state_path.unlink(missing_ok=True)
    return response


# --------------------- Upload ---------------------

@youtube_breaker
def upload_video(
    video_path: Path,
    title: str,
    description: str,
    tags: list[str],
    category_id: str | None = None,
    privacy: str | None = None,
    is_short: bool = False,
    client_secrets: Path | None = None,
    token_file: Path | None = None,
    quota_guard: YouTubeQuotaGuard | None = None,
) -> str:
    """Upload a video. Returns the YouTube video id."""
    guard = quota_guard or YouTubeQuotaGuard(settings.data_dir)
    op    = "videos.insert"

    if not guard.can_afford(op):
        used = guard.used_today()
        logger.error(
            f"YouTube daily quota exhausted ({used}/{guard.daily_limit} units). "
            "Skipping upload."
        )
        raise QuotaExhaustedError(used, guard.daily_limit)

    if not is_short and not ffmpeg_utils.has_audio_stream(video_path):
        raise ValueError(f"Refusing to upload video without audio stream: {video_path}")

    youtube = _youtube_client(client_secrets, token_file)

    # Shorts are identified by hashtag + vertical aspect; YT figures it out.
    final_title = title
    final_desc = description
    if is_short:
        final_title = f"{title[:90]} #Shorts"
        final_desc = f"#Shorts #AI #TechNews\n\n{description}"

    body = {
        "snippet": {
            "title": final_title[:100],
            "description": final_desc[:5000],
            "tags": tags[:500],  # cumulative tag length <500 chars
            "categoryId": category_id or settings.youtube_category_id,
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": {
            "privacyStatus": privacy or settings.youtube_privacy,
            "selfDeclaredMadeForKids": False,
            "embeddable": True,
        },
    }

    media = MediaFileUpload(str(video_path), chunksize=8 * 1024 * 1024, resumable=True)
    request = youtube.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media,
    )

    logger.info(f"Uploading {video_path.name} ({video_path.stat().st_size / 1e6:.1f} MB)")
    try:
        response = _upload_with_retry(request, video_path)
    except Exception as exc:
        if YouTubeQuotaGuard.is_quota_error(exc):
            guard.mark_exhausted()
            raise QuotaExhaustedError(guard.used_today(), guard.daily_limit) from exc
        raise

    guard.charge(op)
    if guard.near_limit():
        logger.warning(
            f"YouTube quota nearly exhausted: {guard.used_today()}/{guard.daily_limit} units used."
        )

    video_id = response["id"]
    logger.info(f"Uploaded: https://youtu.be/{video_id}")
    return video_id


def upload_short(
    video_path: Path,
    title: str,
    description: str | None = None,
    tags: list[str] | None = None,
    client_secrets: Path | None = None,
    token_file: Path | None = None,
) -> str:
    """Upload a Short to YouTube using the standard upload helper."""
    description = description or "#Shorts #AI #TechNews"
    tags = tags or ["shorts", "AI", "technews"]
    return upload_video(
        video_path,
        title=title,
        description=description,
        tags=tags,
        is_short=True,
        client_secrets=client_secrets,
        token_file=token_file,
    )


def set_thumbnail(
    video_id: str,
    thumbnail_path: Path,
    client_secrets: Path | None = None,
    token_file: Path | None = None,
    quota_guard: YouTubeQuotaGuard | None = None,
) -> None:
    """Upload a custom thumbnail for an already-uploaded video."""
    guard = quota_guard or YouTubeQuotaGuard(settings.data_dir)
    if not guard.can_afford("thumbnails.set"):
        logger.warning("Skipping thumbnail upload: YouTube quota exhausted.")
        return

    youtube = _youtube_client(client_secrets, token_file)
    media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
    try:
        youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
        guard.charge("thumbnails.set")
        logger.info(f"Thumbnail set for {video_id}")
    except Exception as e:
        if YouTubeQuotaGuard.is_quota_error(e):
            guard.mark_exhausted()
        # 403 means the channel isn't verified for custom thumbnails (>100 subs)
        logger.warning(f"Could not set thumbnail (skipping): {e}")


def upload_captions(
    video_id: str,
    srt_path: Path,
    language: str = "en",
    client_secrets: Path | None = None,
    token_file: Path | None = None,
    quota_guard: YouTubeQuotaGuard | None = None,
) -> None:
    """Upload an SRT caption track for an already-uploaded video.

    Fails silently with a warning — captions are a nice-to-have and should
    never block the main publish flow.
    """
    guard = quota_guard or YouTubeQuotaGuard(settings.data_dir)
    if not guard.can_afford("captions.insert"):
        logger.warning("Skipping caption upload: YouTube quota exhausted.")
        return

    youtube = _youtube_client(client_secrets, token_file)
    name = _CAPTION_TRACK_NAMES.get(language, language.upper())
    media = MediaFileUpload(str(srt_path), mimetype="text/plain", resumable=False)
    try:
        youtube.captions().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "language": language,
                    "name": name,
                    "isDraft": False,
                }
            },
            media_body=media,
        ).execute()
        guard.charge("captions.insert")
        logger.info(f"Captions [{language}] uploaded for {video_id}")
    except Exception as e:
        if YouTubeQuotaGuard.is_quota_error(e):
            guard.mark_exhausted()
        logger.warning(f"Caption upload failed (non-fatal): {e}")


def _fill_timestamps(description: str, script: VideoScript) -> str:
    """Replace (TIMESTAMPS_AUTOFILL) with real chapter markers from scene durations."""
    if "(TIMESTAMPS_AUTOFILL)" not in description:
        return description

    lines = ["Chapters:"]
    t = 0
    for s in script.scenes:
        mm, ss = divmod(t, 60)
        lines.append(f"{mm:02d}:{ss:02d} {s.heading}")
        t += s.duration_sec
    return description.replace("(TIMESTAMPS_AUTOFILL)", "\n".join(lines))


def publish_episode(
    script: VideoScript,
    long_video: Path,
    short_video: Path | None = None,
    thumbnail: Path | None = None,
    subtitle_path: Path | None = None,
    subtitle_language: str = "en",
    client_secrets: Path | None = None,
    token_file: Path | None = None,
) -> dict:
    """Upload both long-form and Short. Returns {long_id, short_id}.

    Pass *client_secrets* / *token_file* to publish to a channel other than
    the default one configured in settings (e.g. a Russian-language channel).
    Pass *subtitle_path* to upload an SRT caption track alongside the video.
    """
    desc = _fill_timestamps(script.description, script)
    creds_kwargs = {"client_secrets": client_secrets, "token_file": token_file}

    result = {}
    result["long_id"] = upload_video(
        long_video,
        title=script.title,
        description=desc,
        tags=script.tags,
        is_short=False,
        **creds_kwargs,
    )

    if thumbnail and thumbnail.exists():
        set_thumbnail(result["long_id"], thumbnail, **creds_kwargs)

    if subtitle_path and subtitle_path.exists():
        upload_captions(
            result["long_id"], subtitle_path,
            language=subtitle_language,
            **creds_kwargs,
        )

    if short_video and short_video.exists():
        result["short_id"] = upload_video(
            short_video,
            title=script.title,
            description=script.hook,
            tags=script.tags[:10] + ["shorts", "ainews"],
            is_short=True,
            **creds_kwargs,
        )
    return result


# --------------------- CLI ---------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auth",    action="store_true", help="Run OAuth flow only")
    parser.add_argument("--force",   action="store_true", help="Force browser-based OAuth even in headless environment")
    parser.add_argument("--profile", default="default",   choices=["default", "ru"],
                        help="Which channel credentials to use")
    args = parser.parse_args()

    if args.auth:
        if args.profile == "ru":
            _get_creds(settings.ru_youtube_client_secrets, settings.ru_youtube_token_file, force_browser=args.force)
            logger.info(f"RU OAuth token saved to {settings.ru_youtube_token_file}")
        else:
            _get_creds(force_browser=args.force)
            logger.info(f"OAuth token saved to {settings.youtube_token_file}")


if __name__ == "__main__":
    main()
