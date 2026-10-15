"""AI Study Assistant (ChromaDB + Groq) - Streamlit web app.   Run with:  streamlit run app.py"""
import os

import streamlit as st

import config
from embedder import get_embedder
from groq_client import GroqClient, GroqError
from rag import StudyAssistant, ensure_index, format_sources

st.set_page_config(page_title="AI Study Assistant", page_icon="📚")
st.title("📚 AI Study Assistant")
st.caption("Ask questions from your PDF chapters. Retrieval-Augmented Generation: ChromaDB + Groq.")


@st.cache_resource(show_spinner="Loading the embedding model (first time only)...")
def load_embedder():
    return get_embedder("minilm")


embedder = load_embedder()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API key", type="password", value=os.getenv("GROQ_API_KEY", ""),
                            help="Get a free key at console.groq.com/keys. It is only kept in this session.")
    model = st.text_input("Groq model", value=config.get_model(), help="For example llama-3.3-70b-versatile or llama-3.1-8b-instant")
    top_k = st.slider("Chunks sent to the model", 1, 8, config.TOP_K)

    st.header("Chapters")
    uploaded = st.file_uploader("Add chapter PDFs", type="pdf", accept_multiple_files=True)
    new_files = []
    saved = st.session_state.setdefault("saved_uploads", set())
    for f in uploaded or []:
        if f.name not in saved:                     # save each uploaded file only once
            config.CHAPTERS_DIR.mkdir(exist_ok=True)
            (config.CHAPTERS_DIR / f.name).write_bytes(f.getbuffer())
            saved.add(f.name); new_files.append(f.name)
    if new_files:
        st.success(f"Saved {len(new_files)} new file(s).")
    rebuild = st.button("Rebuild embeddings")
    st.write(f"PDFs in folder: {len(list(config.CHAPTERS_DIR.glob('*.pdf'))) if config.CHAPTERS_DIR.exists() else 0}")

# ---------------- ChromaDB index (loaded from disk; rebuilt only when needed) ----------------
if "store" not in st.session_state or rebuild or new_files:
    try:
        with st.spinner("Preparing embeddings..."):
            st.session_state.store = ensure_index(embedder, force=bool(rebuild), log=lambda m: None)
    except (FileNotFoundError, ValueError) as e:
        st.error(str(e)); st.stop()
store = st.session_state.store
st.sidebar.write(f"ChromaDB: {store.count()} chunks")

assistant = StudyAssistant(embedder, store, GroqClient(api_key=api_key, model=model), top_k=top_k)
st.session_state.setdefault("messages", [])

# ---------------- study tools ----------------
with st.expander("Study tools: summary and quiz"):
    topic = st.text_input("Topic (for example: neural networks)")
    c1, c2 = st.columns(2)
    for col, label, fn in ((c1, "Summarize", assistant.summarize), (c2, "Quiz me", assistant.quiz)):
        if col.button(label) and topic:
            try:
                with st.spinner("Asking Groq..."):
                    r = fn(topic)
                st.session_state.messages.append({"role": "user", "content": f"{label}: {topic}"})
                st.session_state.messages.append({"role": "assistant", "content": r["answer"], "sources": r["sources"]})
            except GroqError as e:
                st.error(str(e))

# ---------------- chat ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for line in format_sources(m["sources"]):
                    st.write(line)

question = st.chat_input("Ask a question about your chapters...")
if question:
    history = [(a["content"], b["content"]) for a, b in zip(st.session_state.messages[::2], st.session_state.messages[1::2])]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the chapters and asking Groq..."):
                result = assistant.ask(question, history)
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for line in format_sources(result["sources"]):
                        st.write(line)
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        except GroqError as e:
            st.error(str(e))
