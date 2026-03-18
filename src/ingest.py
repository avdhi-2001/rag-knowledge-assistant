from __future__ import annotations

from pathlib import Path
from typing import Iterable

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from youtube_transcript_api import TranscriptsDisabled, YouTubeTranscriptApi

from .utils import extract_youtube_video_id


class IngestionError(Exception):
    """Raised when a source cannot be ingested."""


class MultiSourceIngestor:
    def load_youtube(self, url_or_id: str) -> list[Document]:
        video_id = extract_youtube_video_id(url_or_id)
        if not video_id:
            raise IngestionError(f"Could not parse a valid YouTube video ID from: {url_or_id}")

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        except TranscriptsDisabled as exc:
            raise IngestionError(f"No English captions available for video: {url_or_id}") from exc
        except Exception as exc:  # pragma: no cover - network/API-dependent
            raise IngestionError(f"Unable to fetch transcript for video: {url_or_id}") from exc

        docs: list[Document] = []
        for idx, chunk in enumerate(transcript_list):
            start = float(chunk.get("start", 0.0))
            duration = float(chunk.get("duration", 0.0))
            docs.append(
                Document(
                    page_content=chunk.get("text", "").strip(),
                    metadata={
                        "source_type": "youtube",
                        "source": f"https://www.youtube.com/watch?v={video_id}",
                        "video_id": video_id,
                        "segment_index": idx,
                        "start_seconds": start,
                        "end_seconds": round(start + duration, 2),
                        "title": f"YouTube Video {video_id}",
                    },
                )
            )
        return docs

    def load_web(self, url: str) -> list[Document]:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
        except Exception as exc:  # pragma: no cover - network/API-dependent
            raise IngestionError(f"Unable to load website: {url}") from exc

        for doc in docs:
            doc.metadata.update({
                "source_type": "web",
                "source": url,
                "title": doc.metadata.get("title") or url,
            })
        return docs

    def load_file(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs = PyPDFLoader(str(path)).load()
        elif suffix in {".txt", ".md", ".py", ".csv"}:
            docs = TextLoader(str(path), encoding="utf-8").load()
        else:
            raise IngestionError(f"Unsupported file type: {suffix}")

        for page_idx, doc in enumerate(docs):
            doc.metadata.update(
                {
                    "source_type": "file",
                    "source": str(path.resolve()),
                    "title": path.name,
                    "page_or_chunk": page_idx + 1,
                }
            )
        return docs

    def load_many(
        self,
        youtube_urls: Iterable[str] | None = None,
        web_urls: Iterable[str] | None = None,
        file_paths: Iterable[str | Path] | None = None,
    ) -> tuple[list[Document], list[str]]:
        all_docs: list[Document] = []
        errors: list[str] = []

        for url in youtube_urls or []:
            if not url.strip():
                continue
            try:
                all_docs.extend(self.load_youtube(url.strip()))
            except IngestionError as exc:
                errors.append(str(exc))

        for url in web_urls or []:
            if not url.strip():
                continue
            try:
                all_docs.extend(self.load_web(url.strip()))
            except IngestionError as exc:
                errors.append(str(exc))

        for path in file_paths or []:
            try:
                all_docs.extend(self.load_file(path))
            except IngestionError as exc:
                errors.append(str(exc))

        return all_docs, errors
