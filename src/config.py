from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = DATA_DIR / "faiss_index"

load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    chunk_size: int = 900
    chunk_overlap: int = 180


settings = Settings()
