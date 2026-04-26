"""Upload long-form video + Shorts to YouTube via Data API v3.

First run: python youtube_uploader.py --auth
This does the OAuth2 desktop flow once and caches the refresh token.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from src.script_generator import VideoScript


SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    # Required for thumbnails.set
    "https://www.googleapis.com/auth/youtube.force-ssl",
]


def _get_creds(
    client_secrets: Path | None = None,
    token_file: Path | None = None,
):
    client_secrets = client_secrets or settings.youtube_client_secrets
    token_file     = token_file     or settings.youtube_token_file

    creds = None
    if token_file.exists():
        with open(token_file, "rb") as f:
            creds = pickle.load(f)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not client_secrets.exists():
            raise FileNotFoundError(
                f"Missing {client_secrets}. "
                "Download OAuth client_secrets.json from Google Cloud Console."
            )
        flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_file, "wb") as f:
            pickle.dump(creds, f)
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


# --------------------- Upload ---------------------

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
) -> str:
    """Upload a video. Returns the YouTube video id."""
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
    response = None
    while response is None:
        try:
            status, response = request.next_chunk()
            if status:
                logger.info(f"  {int(status.progress() * 100)}%")
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504):
                logger.warning(f"Retryable error {e.resp.status}, retrying...")
                continue
            raise

    video_id = response["id"]
    logger.info(f"Uploaded: https://youtu.be/{video_id}")
    return video_id


def set_thumbnail(
    video_id: str,
    thumbnail_path: Path,
    client_secrets: Path | None = None,
    token_file: Path | None = None,
) -> None:
    """Upload a custom thumbnail for an already-uploaded video."""
    youtube = _youtube_client(client_secrets, token_file)
    media = MediaFileUpload(str(thumbnail_path), mimetype="image/jpeg")
    try:
        youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
        logger.info(f"Thumbnail set for {video_id}")
    except HttpError as e:
        # 403 means the channel isn't verified for custom thumbnails (>100 subs)
        logger.warning(f"Could not set thumbnail (skipping): {e}")


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
    client_secrets: Path | None = None,
    token_file: Path | None = None,
) -> dict:
    """Upload both long-form and Short. Returns {long_id, short_id}.

    Pass *client_secrets* / *token_file* to publish to a channel other than
    the default one configured in settings (e.g. a Russian-language channel).
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
    parser.add_argument("--profile", default="default",   choices=["default", "ru"],
                        help="Which channel credentials to use")
    args = parser.parse_args()

    if args.auth:
        if args.profile == "ru":
            _get_creds(settings.ru_youtube_client_secrets, settings.ru_youtube_token_file)
            logger.info(f"RU OAuth token saved to {settings.ru_youtube_token_file}")
        else:
            _get_creds()
            logger.info(f"OAuth token saved to {settings.youtube_token_file}")


if __name__ == "__main__":
    main()
