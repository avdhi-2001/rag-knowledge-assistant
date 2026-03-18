from __future__ import annotations

import time
from textwrap import dedent
from typing import Any

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import settings
from .models import StructuredAnswer

# keep the last N turns in context — beyond this the token cost grows
# faster than the quality benefit, so older turns get dropped
MEMORY_WINDOW = 10

PROMPT = ChatPromptTemplate.from_template(
    dedent(
        """
        You are a helpful research assistant. Stick to the context snippets below —
        if they don't cover the question, just say so rather than guessing.
        Reference snippets as [S1], [S2] etc. when they back up a point.

        {chat_history}
        Context:
        {context}

        Question: {question}
        """
    ).strip()
)


@st.cache_resource(show_spinner=False)
def get_llm() -> ChatOpenAI:
    # cached so we don't spin up a new client on every single query
    return ChatOpenAI(model=settings.chat_model, temperature=0.1)


def format_chat_history(history: list[dict[str, str]]) -> str:
    if not history:
        return ""
    lines = []
    for turn in history:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "Previous conversation:\n" + "\n".join(lines) + "\n\n"


class RagAssistant:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def retrieve(
        self,
        question: str,
        *,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ) -> tuple[list[Document], float]:
        # MMR trades a little relevance for diversity — stops the context window
        # filling up with near-identical chunks from the same paragraph
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": top_k,
                "fetch_k": fetch_k,
                "lambda_mult": lambda_mult,
            },
        )
        t0 = time.perf_counter()
        docs = retriever.invoke(question)
        retrieval_ms = round((time.perf_counter() - t0) * 1000)
        return docs, retrieval_ms

    def _format_context(self, docs: list[Document]) -> tuple[str, list[dict[str, Any]]]:
        lines: list[str] = []
        citations: list[dict[str, Any]] = []

        for idx, doc in enumerate(docs, start=1):
            sid = f"S{idx}"
            meta = doc.metadata or {}
            source_type = meta.get("source_type", "unknown")
            title = meta.get("title") or meta.get("source") or f"Source {idx}"
            source = meta.get("source", "")

            location = None
            if meta.get("page") is not None:
                location = f"page {meta['page'] + 1}"
            elif meta.get("page_or_chunk") is not None:
                location = f"section {meta['page_or_chunk']}"
            elif meta.get("start_seconds") is not None:
                s = meta.get("start_seconds", 0)
                e = meta.get("end_seconds", s)
                location = f"{s:.1f}s–{e:.1f}s"

            lines.append(
                f"[{sid}] {title}\n"
                f"Type: {source_type} | Source: {source}\n"
                f"{doc.page_content}"
            )
            citations.append(
                {
                    "id": sid,
                    "title": title,
                    "source": source,
                    "source_type": source_type,
                    "location": location,
                    "preview": doc.page_content[:220].strip(),
                }
            )
        return "\n\n".join(lines), citations

    def answer(
        self,
        question: str,
        *,
        top_k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        chat_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        docs, retrieval_ms = self.retrieve(
            question,
            top_k=top_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
        )
        context, citations = self._format_context(docs)

        # slice to the window — keeps token count flat no matter how long the chat gets
        window = (chat_history or [])[-MEMORY_WINDOW:]
        history_text = format_chat_history(window)

        llm = get_llm().with_structured_output(StructuredAnswer)
        t0 = time.perf_counter()
        result = llm.invoke(
            PROMPT.format_messages(
                chat_history=history_text,
                context=context,
                question=question,
            )
        )
        llm_ms = round((time.perf_counter() - t0) * 1000)

        out = result.model_dump()
        out["citations"] = citations
        out["metrics"] = {
            "retrieval_ms": retrieval_ms,
            "llm_ms": llm_ms,
            "total_ms": retrieval_ms + llm_ms,
            "chunks_retrieved": len(docs),
            "unique_sources": len({c["source"] for c in citations}),
        }
        return out
