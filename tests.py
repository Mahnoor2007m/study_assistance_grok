"""Automatic tests.  Run:  python tests.py

They need NO internet, NO API key and NOT the embedding model: a small word-count embedder and a fake Groq are used.
If the real `chromadb` package is installed, the tests run against the REAL ChromaDB. If it is not installed,
a tiny stand-in with the same functions is used, so the rest of the project can still be tested."""
import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------------------------
# Stand-in for chromadb (used ONLY when the real package is not installed)
# ---------------------------------------------------------------------------------------------
try:
    import chromadb                                   # real ChromaDB available
    USING_REAL_CHROMA = True
except ImportError:
    USING_REAL_CHROMA = False

    class _Collection:
        def __init__(self, client, name, metadata):
            self.client, self.name, self.metadata, self.rows = client, name, metadata, {}

        def upsert(self, ids, documents=None, embeddings=None, metadatas=None):
            for i, id_ in enumerate(ids):
                self.rows[id_] = (documents[i], list(embeddings[i]), metadatas[i])
            self.client._save()

        def count(self):
            return len(self.rows)

        def get(self, limit=None, include=None):
            ids = list(self.rows)[:limit]
            return {"ids": ids, "documents": [self.rows[i][0] for i in ids],
                    "embeddings": [self.rows[i][1] for i in ids], "metadatas": [self.rows[i][2] for i in ids]}

        def query(self, query_embeddings, n_results=10, include=None):
            q = np.asarray(query_embeddings[0], dtype="float32")
            ids = list(self.rows)
            dist = [1 - float(np.dot(q, np.asarray(self.rows[i][1], dtype="float32"))) for i in ids]
            order = np.argsort(dist)[:n_results]
            pick = [ids[i] for i in order]
            return {"ids": [pick], "distances": [[dist[i] for i in order]],
                    "documents": [[self.rows[i][0] for i in pick]], "metadatas": [[self.rows[i][2] for i in pick]],
                    "embeddings": [[np.asarray(self.rows[i][1]) for i in pick]]}

    class _Client:
        def __init__(self, path, settings=None):
            self.file = Path(path) / "fake_chroma.pkl"
            self.collections = {}
            if self.file.exists():
                for name, (metadata, rows) in pickle.loads(self.file.read_bytes()).items():
                    c = _Collection(self, name, metadata); c.rows = rows; self.collections[name] = c

        def _save(self):
            self.file.write_bytes(pickle.dumps({n: (c.metadata, c.rows) for n, c in self.collections.items()}))

        def get_or_create_collection(self, name, metadata=None, **kwargs):
            if name not in self.collections:
                self.collections[name] = _Collection(self, name, metadata)
                self._save()
            return self.collections[name]

        def delete_collection(self, name):
            del self.collections[name]
            self._save()

    fake = types.ModuleType("chromadb")
    fake.PersistentClient = _Client
    fake_config = types.ModuleType("chromadb.config")
    fake_config.Settings = lambda **kw: None
    fake.config = fake_config
    sys.modules["chromadb"], sys.modules["chromadb.config"] = fake, fake_config

# ---------------------------------------------------------------------------------------------
import generate_sample_chapters
from chroma_store import ChromaStore
from embedder import HashingEmbedder
from groq_client import GroqClient, GroqError
from pdf_loader import chunk_text, load_and_chunk, chapter_name
from rag import StudyAssistant, ensure_index, NOT_FOUND


class FakeResponse:
    def __init__(self, status=200, content="fake answer [1]"):
        self.status_code, self._content, self.text = status, content, "error text"

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeLLM:
    """Records what would be sent to Groq."""
    def __init__(self):
        self.calls = []

    def chat(self, messages):
        self.calls.append(messages)
        return "Fake answer from Groq [1]"


class StudyAssistantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(cls.tmp.name)
        cls.chapters, cls.store_dir = root / "chapters", root / "chroma_db"
        generate_sample_chapters.make(cls.chapters)
        cls.embedder = HashingEmbedder()
        cls.store = ensure_index(cls.embedder, chapters_dir=cls.chapters, store_dir=cls.store_dir, log=lambda m: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---------- loading and chunking ----------
    def test_chunk_text_respects_size_and_keeps_all_words(self):
        text = " ".join(f"This is sentence number {i} about learning." for i in range(60))
        chunks = chunk_text(text, size=300, overlap=60)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 300 + 60 for c in chunks))
        self.assertIn("sentence number 59", chunks[-1])
        self.assertTrue(set(chunks[0].split()[-3:]) & set(chunks[1].split()[:12]))

    def test_loader_reads_five_chapters(self):
        chunks = load_and_chunk(self.chapters, log=lambda m: None)
        self.assertEqual(len({c["file"] for c in chunks}), 5)
        self.assertTrue(all(c["page"] >= 1 and c["text"] for c in chunks))
        self.assertEqual(chapter_name("Chapter_2_Machine_Learning_Basics.pdf"), "Chapter 2 Machine Learning Basics")

    def test_empty_folder_gives_clear_error(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            with self.assertRaises(FileNotFoundError):
                load_and_chunk(d)

    # ---------- ChromaDB: embeddings are saved locally ----------
    def test_embeddings_are_stored_in_chromadb_with_metadata(self):
        n = len(load_and_chunk(self.chapters, log=lambda m: None))
        self.assertEqual(self.store.count(), n)
        info = self.store.stats()
        self.assertEqual(len(info["per_chapter"]), 5)
        self.assertEqual(sum(info["per_chapter"].values()), n)
        sample = self.store.sample(1)[0]
        self.assertEqual(sample["dimension"], 4096)                      # size of the test embedder
        self.assertTrue({"chapter", "file", "page", "chunk_id"} <= set(sample["metadata"]))
        meta = json.loads((self.store_dir / "index_meta.json").read_text())
        self.assertEqual(meta["n_chunks"], n)

    def test_saved_embeddings_survive_a_restart_and_are_reused(self):
        reopened = ChromaStore(self.store_dir)                           # a "new run" of the program
        self.assertEqual(reopened.count(), self.store.count())
        logs = []
        ensure_index(self.embedder, chapters_dir=self.chapters, store_dir=self.store_dir, log=logs.append)
        self.assertTrue(any("Loaded saved ChromaDB" in m for m in logs))  # no re-embedding was needed

    def test_embeddings_are_rebuilt_when_pdfs_change(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            ch, st = d / "c", d / "s"
            generate_sample_chapters.make(ch)
            store1 = ensure_index(self.embedder, chapters_dir=ch, store_dir=st, log=lambda m: None)
            before = store1.count()
            (ch / "Chapter_6_extra.pdf").write_bytes((ch / "Chapter_1_What_is_AI.pdf").read_bytes())
            logs = []
            store2 = ensure_index(self.embedder, chapters_dir=ch, store_dir=st, log=logs.append)
            self.assertTrue(any("changed" in m for m in logs))
            self.assertGreater(store2.count(), before)
            self.assertEqual(len(store2.stats()["per_chapter"]), 6)      # old data was replaced, not duplicated

    # ---------- retrieval ----------
    def test_retrieval_finds_the_right_chapter_with_scores(self):
        assistant = StudyAssistant(self.embedder, self.store, FakeLLM(), min_score=0.05)
        cases = {
            "Who coined the term artificial intelligence?": "Chapter_1",
            "What is overfitting and how can it be reduced?": "Chapter_2",
            "What is backpropagation and gradient descent?": "Chapter_3",
            "What are the steps of the RAG pipeline?": "Chapter_4",
            "What is algorithmic bias and what causes it?": "Chapter_5",
        }
        for question, expected in cases.items():
            hits = assistant.retrieve(question)
            self.assertIn(expected, hits[0]["file"], f"{question!r} -> {hits[0]['file']}")
            self.assertGreater(hits[0]["score"], 0.05)
            self.assertEqual(len(hits), 4)
            self.assertEqual([h["score"] for h in hits], sorted((h["score"] for h in hits), reverse=True))

    # ---------- RAG answer ----------
    def test_prompt_contains_context_question_and_rules(self):
        llm = FakeLLM()
        assistant = StudyAssistant(self.embedder, self.store, llm, min_score=0.05)
        result = assistant.ask("What is overfitting?", history=[("hello", "hi")])
        self.assertEqual(result["answer"], "Fake answer from Groq [1]")
        self.assertTrue(result["grounded"] and 1 <= len(result["sources"]) <= 4)
        messages = llm.calls[0]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("ONLY", messages[0]["content"])
        self.assertEqual(messages[1:3], [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}])
        user = messages[-1]["content"]
        self.assertIn("QUESTION: What is overfitting?", user)
        self.assertIn("[1] (Chapter 2", user)
        self.assertIn("memorises the training data", user)               # the retrieved text really is in the prompt

    def test_unrelated_question_does_not_call_the_llm(self):
        llm = FakeLLM()
        result = StudyAssistant(self.embedder, self.store, llm, min_score=0.05).ask("xyzzy plugh quux banana")
        self.assertEqual(result["answer"], NOT_FOUND)
        self.assertFalse(result["grounded"])
        self.assertEqual(llm.calls, [])

    def test_summary_and_quiz_use_their_own_instructions(self):
        llm = FakeLLM()
        assistant = StudyAssistant(self.embedder, self.store, llm, min_score=0.05)
        assistant.summarize("neural networks backpropagation")
        assistant.quiz("neural networks backpropagation", n=3)
        self.assertIn("study summary", llm.calls[0][0]["content"])
        self.assertIn("3 multiple-choice", llm.calls[1][0]["content"])

    # ---------- Groq client ----------
    def test_groq_client_request_format_and_errors(self):
        sent = {}

        def ok_post(url, headers, json, timeout):
            sent.update(url=url, headers=headers, json=json)
            return FakeResponse(200, "  hello  ")

        client = GroqClient(api_key="gsk_test", model="llama-3.3-70b-versatile", post=ok_post)
        self.assertEqual(client.chat([{"role": "user", "content": "hi"}]), "hello")
        self.assertEqual(sent["url"], "https://api.groq.com/openai/v1/chat/completions")
        self.assertEqual(sent["headers"]["Authorization"], "Bearer gsk_test")
        self.assertEqual(sent["json"]["model"], "llama-3.3-70b-versatile")

        for status, word in ((401, "API key"), (404, "GROQ_MODEL"), (413, "too large"), (429, "Rate limit"), (500, "500")):
            c = GroqClient(api_key="k", post=lambda *a, status=status, **k: FakeResponse(status))
            with self.assertRaises(GroqError) as cm:
                c.chat([])
            self.assertIn(word, str(cm.exception))
        with self.assertRaises(GroqError):
            GroqClient(api_key="", post=ok_post).chat([])                # missing key

        def broken_post(*a, **k):
            raise ConnectionError("no internet")
        with self.assertRaises(GroqError):
            GroqClient(api_key="k", post=broken_post).chat([])


if __name__ == "__main__":
    print("ChromaDB used in the tests:", "REAL chromadb package" if USING_REAL_CHROMA else "stand-in (chromadb is not installed here)")
    unittest.main(verbosity=2)
