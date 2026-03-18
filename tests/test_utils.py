from __future__ import annotations

import pytest
from src.utils import extract_youtube_video_id


# --- valid cases ---

def test_plain_video_id():
    # just the 11-char ID on its own
    assert extract_youtube_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_standard_watch_url():
    assert extract_youtube_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_short_url():
    assert extract_youtube_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_shorts_url():
    assert extract_youtube_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_embed_url():
    assert extract_youtube_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_mobile_url():
    assert extract_youtube_video_id("https://m.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

def test_url_with_extra_params():
    # timestamp and playlist params shouldn't break anything
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s&list=PLxxx"
    assert extract_youtube_video_id(url) == "dQw4w9WgXcQ"

def test_strips_whitespace():
    assert extract_youtube_video_id("  dQw4w9WgXcQ  ") == "dQw4w9WgXcQ"


# --- invalid cases ---

def test_returns_none_for_random_string():
    assert extract_youtube_video_id("not-a-url") is None

def test_returns_none_for_non_youtube_url():
    assert extract_youtube_video_id("https://vimeo.com/123456789") is None

def test_returns_none_for_empty_string():
    assert extract_youtube_video_id("") is None

def test_returns_none_for_id_too_short():
    # 10 chars — one short of the required 11
    assert extract_youtube_video_id("dQw4w9WgXc") is None

def test_returns_none_for_id_too_long():
    assert extract_youtube_video_id("dQw4w9WgXcQQ") is None
