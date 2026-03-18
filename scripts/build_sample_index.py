from __future__ import annotations

import argparse

from src.config import INDEX_DIR
from src.ingest import MultiSourceIngestor
from src.splitter import split_documents
from src.vectorstore import build_vectorstore, save_vectorstore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a FAISS index for the RAG Knowledge Assistant.")
    parser.add_argument("--youtube", nargs="*", default=[], help="YouTube URLs or IDs")
    parser.add_argument("--web", nargs="*", default=[], help="Website URLs")
    parser.add_argument("--files", nargs="*", default=[], help="Local file paths")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingestor = MultiSourceIngestor()
    docs, errors = ingestor.load_many(
        youtube_urls=args.youtube,
        web_urls=args.web,
        file_paths=args.files,
    )
    if errors:
        for error in errors:
            print(f"Warning: {error}")
    if not docs:
        raise SystemExit("No documents loaded. Nothing to index.")

    chunks = split_documents(docs)
    vectorstore = build_vectorstore(chunks)
    save_vectorstore(vectorstore, INDEX_DIR)
    print(f"Saved index with {len(chunks)} chunks to {INDEX_DIR}")


if __name__ == "__main__":
    main()
