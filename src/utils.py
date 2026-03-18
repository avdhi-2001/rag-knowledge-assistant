from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, urlparse


YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "youtu.be", "m.youtube.com"}


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    return value.lower() or "source"


def chunk_list(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def extract_youtube_video_id(raw: str) -> str | None:
    raw = raw.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", raw):
        return raw

    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if host not in YOUTUBE_HOSTS:
        return None

    if host == "youtu.be":
        video_id = parsed.path.strip("/")
        return video_id if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None

    if parsed.path == "/watch":
        query = parse_qs(parsed.query)
        video_id = query.get("v", [None])[0]
        return video_id if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None

    path_parts = [part for part in parsed.path.split("/") if part]
    for idx, part in enumerate(path_parts):
        if part in {"embed", "shorts", "live"} and idx + 1 < len(path_parts):
            video_id = path_parts[idx + 1]
            return video_id if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None
    return None
