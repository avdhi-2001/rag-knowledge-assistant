from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from src.chain import RagAssistant
from src.config import INDEX_DIR, settings
from src.ingest import MultiSourceIngestor
from src.splitter import split_documents
from src.vectorstore import build_vectorstore, load_vectorstore, save_vectorstore

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
def apply_theme(dark: bool) -> None:
    if dark:
        bg        = "#0F1117"
        surface   = "#1A1D27"
        surface2  = "#22263A"
        border    = "#2E3250"
        text      = "#E8E8F0"
        muted     = "#8B8FA8"
        accent    = "#6C63FF"
        accent2   = "#48E5C2"
        badge_bg  = "#22263A"
    else:
        bg        = "#F5F6FA"
        surface   = "#FFFFFF"
        surface2  = "#EEEEF6"
        border    = "#DDDDE8"
        text      = "#1A1A2E"
        muted     = "#6B6B80"
        accent    = "#6C63FF"
        accent2   = "#0F9B8E"
        badge_bg  = "#EEEEF6"

    st.markdown(f"""
    <style>
    /* page background */
    .stApp {{ background-color: {bg}; }}
    section[data-testid="stSidebar"] {{ background-color: {surface} !important; border-right: 1px solid {border}; }}

    /* header bar */
    .rag-header {{
        background: linear-gradient(135deg, {accent} 0%, {accent2} 100%);
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 28px;
        display: flex;
        align-items: center;
        gap: 16px;
    }}
    .rag-header h1 {{ color: #fff !important; font-size: 2rem; margin: 0; }}
    .rag-header p  {{ color: rgba(255,255,255,0.82); margin: 4px 0 0; font-size: 0.95rem; }}

    /* answer card */
    .answer-card {{
        background: {surface};
        border: 1px solid {border};
        border-left: 4px solid {accent};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        color: {text};
    }}

    /* key point pill */
    .kp-pill {{
        background: {surface2};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 8px 14px;
        margin: 5px 0;
        font-size: 0.9rem;
        color: {text};
    }}

    /* follow-up chip */
    .fq-chip {{
        display: inline-block;
        background: {badge_bg};
        border: 1px solid {border};
        border-radius: 20px;
        padding: 5px 14px;
        margin: 4px 4px 4px 0;
        font-size: 0.85rem;
        color: {accent};
        cursor: default;
    }}

    /* citation card */
    .cite-card {{
        background: {surface2};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }}
    .cite-id {{
        background: {accent};
        color: #fff;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 8px;
    }}
    .cite-title {{ font-weight: 600; color: {text}; font-size: 0.9rem; }}
    .cite-source {{ color: {muted}; font-size: 0.78rem; margin-top: 2px; }}
    .cite-preview {{ color: {muted}; font-size: 0.83rem; margin-top: 8px; line-height: 1.5; }}

    /* confidence badge */
    .conf-high   {{ background:#1a3a2a; color:#4ade80; border:1px solid #2d6a4a; border-radius:8px; padding:6px 14px; font-weight:600; display:inline-block; }}
    .conf-medium {{ background:#3a2e1a; color:#fbbf24; border:1px solid #6a4e2d; border-radius:8px; padding:6px 14px; font-weight:600; display:inline-block; }}
    .conf-low    {{ background:#3a1a1a; color:#f87171; border:1px solid #6a2d2d; border-radius:8px; padding:6px 14px; font-weight:600; display:inline-block; }}

    /* metric cards */
    .metric-row {{ display:flex; gap:10px; margin-bottom:16px; }}
    .metric-box {{
        flex:1; background:{surface2}; border:1px solid {border};
        border-radius:10px; padding:12px; text-align:center;
    }}
    .metric-val {{ font-size:1.3rem; font-weight:700; color:{accent}; }}
    .metric-lbl {{ font-size:0.75rem; color:{muted}; margin-top:2px; }}

    /* sidebar section label */
    .sidebar-label {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: .08em;
        text-transform: uppercase;
        color: {muted};
        margin: 18px 0 8px;
    }}

    /* hide default streamlit branding */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)


# ── theme toggle ─────────────────────────────────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True

dark = st.session_state["dark_mode"]
apply_theme(dark)

# ── helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_saved_assistant() -> RagAssistant | None:
    vs = load_vectorstore(INDEX_DIR)
    return RagAssistant(vs) if vs else None


def save_uploaded_files(uploaded_files) -> list[str]:
    upload_dir = Path(tempfile.gettempdir()) / "rag_assistant_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in uploaded_files:
        target = upload_dir / f.name
        target.write_bytes(f.getbuffer())
        paths.append(str(target))
    return paths


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # theme toggle at the very top
    col_logo, col_toggle = st.columns([3, 1])
    with col_logo:
        st.markdown("### 🔎 RAG Assistant")
    with col_toggle:
        mode_label = "☀️" if dark else "🌙"
        if st.button(mode_label, help="Toggle dark / light mode"):
            st.session_state["dark_mode"] = not dark
            st.rerun()

    st.markdown('<div class="sidebar-label">Knowledge base</div>', unsafe_allow_html=True)
    youtube_text = st.text_area(
        "YouTube URLs or IDs",
        placeholder="One per line",
        help="Full YouTube URLs or plain 11-character video IDs both work.",
        label_visibility="collapsed",
    )
    st.caption("YouTube URLs or IDs — one per line")

    web_text = st.text_area(
        "Website URLs",
        placeholder="One per line",
        help="Paste any blog, article, or docs page URL.",
        label_visibility="collapsed",
    )
    st.caption("Website URLs — one per line")

    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "md", "csv", "py"],
        accept_multiple_files=True,
    )

    if st.button("Build index", type="primary", use_container_width=True):
        ingestor = MultiSourceIngestor()
        file_paths = save_uploaded_files(uploaded_files) if uploaded_files else []
        with st.spinner("Ingesting sources..."):
            docs, errors = ingestor.load_many(
                youtube_urls=youtube_text.splitlines(),
                web_urls=web_text.splitlines(),
                file_paths=file_paths,
            )
        if not docs:
            st.error("Nothing loaded — add at least one source.")
        else:
            with st.spinner("Building FAISS index..."):
                chunked_docs = split_documents(docs)
                vectorstore = build_vectorstore(chunked_docs)
                save_vectorstore(vectorstore, INDEX_DIR)
                st.session_state["assistant"] = RagAssistant(vectorstore)
                st.session_state["chat_history"] = []

            source_types: dict[str, int] = {}
            for d in docs:
                t = d.metadata.get("source_type", "unknown")
                source_types[t] = source_types.get(t, 0) + 1
            breakdown = ", ".join(f"{v} {k}" for k, v in source_types.items())
            st.success(
                f"Indexed {len(chunked_docs)} chunks from {len(docs)} documents"
                + (f" ({breakdown})" if breakdown else "") + "."
            )
        for err in errors:
            st.warning(err)

    st.markdown('<div class="sidebar-label">Retrieval</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-k chunks", min_value=2, max_value=10, value=5)
    fetch_k = st.slider("Fetch-k pool", min_value=5, max_value=40, value=20)
    lambda_mult = st.slider(
        "MMR diversity",
        min_value=0.0, max_value=1.0, value=0.5, step=0.1,
        help="0 = maximise diversity, 1 = maximise relevance",
    )

    st.markdown('<div class="sidebar-label">Memory</div>', unsafe_allow_html=True)
    st.caption("Sliding window — last 10 turns only, so token cost stays flat.")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state["chat_history"] = []
        st.success("Cleared.")

    if not settings.openai_api_key:
        st.error("No API key found — add OPENAI_API_KEY to your .env file.")


# ── main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rag-header">
    <div>
        <h1>RAG Knowledge Assistant</h1>
        <p>Ask questions across YouTube videos, PDFs, and websites — with cited, grounded answers.</p>
    </div>
</div>
""", unsafe_allow_html=True)

assistant = st.session_state.get("assistant") or get_saved_assistant()

# past conversation
chat_history: list[dict] = st.session_state.get("chat_history", [])
if chat_history:
    st.markdown("#### Conversation")
    for turn in chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
    st.divider()

# query input
question = st.text_input(
    "Ask a question",
    placeholder="What are the main ideas across these sources?",
    label_visibility="collapsed",
)

ask_col, hint_col = st.columns([1, 5])
with ask_col:
    ask_clicked = st.button("Ask", type="primary", disabled=not bool(question and assistant), use_container_width=True)
with hint_col:
    if not assistant:
        st.caption("Build an index first using the sidebar.")

if ask_clicked:
    with st.spinner("Searching sources and generating answer..."):
        result = assistant.answer(
            question,
            top_k=top_k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            chat_history=chat_history,
        )

    chat_history.append({"question": question, "answer": result["answer"]})
    st.session_state["chat_history"] = chat_history

    m = result["metrics"]
    stats = st.session_state.setdefault(
        "stats", {"queries": 0, "total_retrieval_ms": 0, "total_llm_ms": 0}
    )
    stats["queries"] += 1
    stats["total_retrieval_ms"] += m["retrieval_ms"]
    stats["total_llm_ms"] += m["llm_ms"]

    col1, col2 = st.columns([3, 2])

    with col1:
        # answer
        st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)

        # key points
        if result["key_points"]:
            st.markdown("**Key points**")
            for pt in result["key_points"]:
                st.markdown(f'<div class="kp-pill">• {pt}</div>', unsafe_allow_html=True)

        # follow-ups
        if result["follow_up_questions"]:
            st.markdown("<br>**You might also ask**", unsafe_allow_html=True)
            chips = "".join(f'<span class="fq-chip">{q}</span>' for q in result["follow_up_questions"])
            st.markdown(chips, unsafe_allow_html=True)

    with col2:
        # confidence
        conf = result["confidence"]
        st.markdown("**Confidence**")
        st.markdown(f'<div class="conf-{conf}">{conf.upper()}</div>', unsafe_allow_html=True)

        # metrics
        st.markdown("<br>**Query metrics**", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-box"><div class="metric-val">{m['retrieval_ms']}ms</div><div class="metric-lbl">Retrieval</div></div>
            <div class="metric-box"><div class="metric-val">{m['llm_ms']}ms</div><div class="metric-lbl">LLM</div></div>
            <div class="metric-box"><div class="metric-val">{m['unique_sources']}</div><div class="metric-lbl">Sources</div></div>
        </div>
        """, unsafe_allow_html=True)

        # citations
        st.markdown("**Citations**")
        for c in result["citations"]:
            loc = f" <span style='font-size:0.78rem;opacity:0.6'>— {c['location']}</span>" if c.get("location") else ""
            st.markdown(f"""
            <div class="cite-card">
                <span class="cite-id">{c['id']}</span>
                <span class="cite-title">{c['title']}</span>{loc}
                <div class="cite-source">{c['source']}</div>
                <div class="cite-preview">{c['preview']}</div>
            </div>
            """, unsafe_allow_html=True)

# session stats footer
if "stats" in st.session_state and st.session_state["stats"]["queries"] > 0:
    s = st.session_state["stats"]
    n = s["queries"]
    st.divider()
    st.caption(f"Session — {n} queries · avg retrieval {s['total_retrieval_ms']//n}ms · avg LLM {s['total_llm_ms']//n}ms")
