"""Step 5: the RAG pipeline = Retrieve relevant chunks (ChromaDB) -> Augment the prompt -> Generate the answer (Groq)."""
import config
from chroma_store import ChromaStore, pdf_signature
from pdf_loader import load_and_chunk

NOT_FOUND = ("I couldn't find this in your chapters. Try rephrasing the question, or check that "
             "the topic is covered in the PDFs you added.")

ANSWER_SYSTEM = """You are an AI Study Assistant that helps a student learn from their textbook chapters.
Rules:
1. Answer ONLY with information from the numbered CONTEXT passages. Never use outside knowledge and never invent facts.
2. After every statement that comes from a passage, add its number in square brackets, for example [1] or [2][3].
3. If the context does not contain the answer, reply exactly: "I couldn't find this in your chapters."
4. Explain in simple, clear language for a student. Be concise. Use short bullet points for lists or steps.
5. Reply in the same language as the student's question (for example English, or Urdu/Hindi written in English letters)."""

QUIZ_SYSTEM = """You are an AI Study Assistant. Using ONLY the numbered CONTEXT passages, write a short practice quiz.
Create {n} multiple-choice questions. For each question give 4 options (A-D), then 'Answer:' with the correct letter,
and a one-line explanation with a source number like [1]. Do not use outside knowledge."""

SUMMARY_SYSTEM = """You are an AI Study Assistant. Using ONLY the numbered CONTEXT passages, write a study summary of the topic:
a 2-3 sentence overview, then 5-8 bullet points of key ideas, then a list of 'Key terms' with one-line definitions.
Add source numbers like [1] after the facts. Do not use outside knowledge."""


def ensure_index(embedder, force=False, chapters_dir=None, store_dir=None, log=print):
    """Use the embeddings saved in ChromaDB if they are up to date; otherwise (re)build them from the PDFs."""
    chapters_dir = chapters_dir or config.CHAPTERS_DIR
    store_dir = store_dir or config.CHROMA_DIR
    signature = pdf_signature(chapters_dir, embedder.name) if chapters_dir.exists() else ""

    store = ChromaStore(store_dir)
    if not force and store.is_current(signature):
        log(f"Loaded saved ChromaDB: {store.count()} chunks (created {store.meta['created']})")
        return store
    if not force and store.meta:
        log("The PDFs or settings changed since the embeddings were saved -> rebuilding...")

    log("Reading and chunking the PDFs...")
    chunks = load_and_chunk(chapters_dir, log=log)
    log(f"Creating embeddings for {len(chunks)} chunks and saving them in ChromaDB (first run can take a minute)...")
    store.rebuild(chunks, embedder, signature)
    log(f"Saved to '{store_dir}'")
    return store


class StudyAssistant:
    def __init__(self, embedder, store, llm, top_k=config.TOP_K, min_score=config.MIN_SCORE):
        self.embedder, self.store, self.llm = embedder, store, llm
        self.top_k, self.min_score = top_k, min_score

    # ---- Retrieve ----
    def retrieve(self, query, k=None):
        vector = self.embedder.encode([query])[0]
        return self.store.search(vector, k or self.top_k)

    @staticmethod
    def format_context(hits):
        return "\n\n".join(f"[{i}] ({h['chapter']}, page {h['page']})\n{h['text']}" for i, h in enumerate(hits, 1))

    # ---- Augment + Generate ----
    def ask(self, question, history=None, k=None):
        """Answer a question. `history` = earlier [(question, answer), ...] so follow-up questions work."""
        hits = [h for h in self.retrieve(question, k) if h["score"] >= self.min_score]
        if not hits:                                        # nothing relevant: do not waste an API call
            return {"answer": NOT_FOUND, "sources": [], "grounded": False}

        messages = [{"role": "system", "content": ANSWER_SYSTEM}]
        for q, a in (history or [])[-3:]:                   # keep the last 3 turns for follow-up questions
            messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        messages.append({"role": "user",
                         "content": f"CONTEXT:\n{self.format_context(hits)}\n\nQUESTION: {question}"})
        return {"answer": self.llm.chat(messages), "sources": hits, "grounded": True}

    def summarize(self, topic):
        hits = [h for h in self.retrieve(topic, 8) if h["score"] >= self.min_score]
        if not hits:
            return {"answer": NOT_FOUND, "sources": [], "grounded": False}
        messages = [{"role": "system", "content": SUMMARY_SYSTEM},
                    {"role": "user", "content": f"CONTEXT:\n{self.format_context(hits)}\n\nTOPIC: {topic}"}]
        return {"answer": self.llm.chat(messages), "sources": hits, "grounded": True}

    def quiz(self, topic, n=5):
        hits = [h for h in self.retrieve(topic, 6) if h["score"] >= self.min_score]
        if not hits:
            return {"answer": NOT_FOUND, "sources": [], "grounded": False}
        messages = [{"role": "system", "content": QUIZ_SYSTEM.format(n=n)},
                    {"role": "user", "content": f"CONTEXT:\n{self.format_context(hits)}\n\nTOPIC: {topic}"}]
        return {"answer": self.llm.chat(messages), "sources": hits, "grounded": True}


def format_sources(hits):
    """Human-readable source list for the terminal / web page."""
    return [f"[{i}] {h['chapter']} - page {h['page']}  (similarity {h['score']:.2f})" for i, h in enumerate(hits, 1)]
