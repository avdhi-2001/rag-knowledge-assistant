from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from .config import INDEX_DIR, settings
from .utils import ensure_directory


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model)


def build_vectorstore(documents: list[Document]) -> FAISS:
    if not documents:
        raise ValueError("No documents were provided for indexing.")
    return FAISS.from_documents(documents, get_embeddings())


def save_vectorstore(vectorstore: FAISS, index_dir: Path = INDEX_DIR) -> None:
    ensure_directory(index_dir)
    vectorstore.save_local(str(index_dir))


def load_vectorstore(index_dir: Path = INDEX_DIR) -> FAISS | None:
    if not index_dir.exists():
        return None
    return FAISS.load_local(
        str(index_dir),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )
