"""Step 3: store the embeddings in ChromaDB (saved on disk) and search them (semantic search).

ChromaDB keeps, for every chunk:  the embedding vector + the chunk text + metadata (chapter, file, page).
Folder chroma_db/ contains the database files and index_meta.json (a fingerprint of the PDFs, used to
detect when the PDFs changed and the embeddings must be rebuilt).
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import config

BATCH = 500


def pdf_signature(folder, embedder_name, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """A fingerprint of the PDFs + settings. If it changes, the saved embeddings are out of date."""
    parts = [embedder_name, str(size), str(overlap)]
    for pdf in sorted(Path(folder).glob("*.pdf")):
        stat = pdf.stat()
        parts.append(f"{pdf.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


class ChromaStore:
    def __init__(self, path=None, collection_name=config.COLLECTION_NAME):
        import chromadb
        self.path = Path(path or config.CHROMA_DIR)
        self.path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        try:
            from chromadb.config import Settings
            self.client = chromadb.PersistentClient(path=str(self.path), settings=Settings(anonymized_telemetry=False))
        except Exception:
            self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self._open_collection()

    def _open_collection(self):
        """Open (or create) the collection. Cosine distance is requested; the score is also computed in search()
        from the stored vectors, so results are correct even if a ChromaDB version ignores this setting."""
        try:
            return self.client.get_or_create_collection(name=self.collection_name, metadata={"hnsw:space": "cosine"})
        except Exception:
            return self.client.get_or_create_collection(name=self.collection_name)

    # ---------- information about the saved index ----------
    @property
    def meta(self):
        meta_file = self.path / "index_meta.json"
        return json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}

    def count(self):
        return self.collection.count()

    def is_current(self, signature):
        """True if embeddings are saved and the PDFs have not changed since."""
        meta = self.meta
        return bool(meta) and meta.get("signature") == signature and 0 < self.count() == meta.get("n_chunks")

    # ---------- build ----------
    def rebuild(self, chunks, embedder, signature=""):
        """Create the embeddings of all chunks and store them in a fresh ChromaDB collection."""
        vectors = embedder.encode([c["text"] for c in chunks])
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass                                                     # nothing to delete the first time
        self.collection = self._open_collection()
        for start in range(0, len(chunks), BATCH):
            part = chunks[start:start + BATCH]
            self.collection.upsert(
                ids=[str(c["id"]) for c in part],
                documents=[c["text"] for c in part],
                embeddings=vectors[start:start + BATCH].tolist(),
                metadatas=[{"chapter": c["chapter"], "file": c["file"], "page": int(c["page"]), "chunk_id": int(c["id"])}
                           for c in part])
        meta = {"embedder": embedder.name, "n_chunks": len(chunks), "dimension": int(vectors.shape[1]),
                "collection": self.collection_name, "signature": signature, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        (self.path / "index_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    # ---------- search ----------
    def search(self, query_vector, k=4):
        """Return the k chunks most similar to the question. Vectors have length 1, so the dot product
        of the question vector and a chunk vector is the cosine similarity (1.0 = same meaning, 0 = unrelated)."""
        k = min(k, self.count())
        if k == 0:
            return []
        q = np.asarray(query_vector, dtype="float32")
        res = self.collection.query(query_embeddings=[q.tolist()], n_results=k,
                                    include=["documents", "metadatas", "embeddings"])
        hits = []
        for text, meta, emb in zip(res["documents"][0], res["metadatas"][0], res["embeddings"][0]):
            score = float(np.dot(q, np.asarray(emb, dtype="float32")))
            hits.append({"id": meta.get("chunk_id"), "text": text, "chapter": meta["chapter"],
                         "file": meta["file"], "page": meta["page"], "score": score})
        hits.sort(key=lambda h: -h["score"])
        return hits

    # ---------- look inside the database (used by inspect_db.py) ----------
    def stats(self):
        data = self.collection.get(include=["metadatas"])
        per_chapter = {}
        for m in data["metadatas"]:
            per_chapter[m["chapter"]] = per_chapter.get(m["chapter"], 0) + 1
        return {"chunks": self.count(), "per_chapter": dict(sorted(per_chapter.items())), "meta": self.meta}

    def sample(self, n=2):
        data = self.collection.get(limit=n, include=["documents", "metadatas", "embeddings"])
        return [{"text": t, "metadata": m, "dimension": len(e), "embedding_preview": [round(float(x), 4) for x in list(e)[:8]]}
                for t, m, e in zip(data["documents"], data["metadatas"], data["embeddings"])]
