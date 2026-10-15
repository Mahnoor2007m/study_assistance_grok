# AI Study Assistant (RAG with ChromaDB + Groq)

Ask questions from your **5 PDF chapters**. The app creates **embeddings**, stores them in **ChromaDB** (saved on your
computer), retrieves the most relevant passages and uses an LLM on **Groq** to write an accurate answer with sources.

## How it works (RAG pipeline)
```
 PDF chapters ──► pdf_loader.py ──► chunks (about 900 characters, 150 overlap, with chapter + page)
                                         │
                                         ▼
                       embedder.py  (sentence-transformers all-MiniLM-L6-v2, runs locally, 384 numbers per chunk)
                                         │
                                         ▼
                  chroma_store.py ──► ChromaDB collection saved in chroma_db/  (vector + text + chapter/page)
                                         │
 Question ──► embedding ──► similarity search in ChromaDB ──► top-4 chunks  (rag.py: retrieve)
                                         │
                                         ▼
          prompt = rules + numbered context + question ──► Groq API (groq_client.py) ──► answer with [1] [2] sources
```
| Requirement | Where |
|---|---|
| Generate embeddings | `embedder.py` (Groq has no embedding model, so a local sentence-transformer is used) |
| Save them in ChromaDB | `chroma_store.py` (`chroma_db/` folder; reused on the next run, rebuilt only when the PDFs change) |
| Retrieve relevant information | `ChromaStore.search()` and `StudyAssistant.retrieve()` |
| Use Groq for accurate answers | `groq_client.py` + the prompt in `rag.py` (answer only from context, cite sources, say "not found" otherwise) |

Extras: follow-up questions (chat memory), `/summary` and `/quiz` study tools, Streamlit web app, `inspect_db.py`
(shows what is inside ChromaDB), automatic tests.

## Setup
1. Install Python 3.10 or newer.
2. In the project folder:
   ```bash
   python -m venv venv
   venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Create a free API key at https://console.groq.com/keys and make a file named `.env` (copy `.env.example`):
   ```
   GROQ_API_KEY=gsk_your_key_here
   ```
4. **Add your 5 chapters:** delete the 5 sample PDFs in `chapters/` and copy your own PDF files there.
   (The samples are about "Introduction to AI" and are only for trying the project.)

## Run
| What | Command |
|---|---|
| Chat in the terminal | `python study_assistant.py` |
| One question | `python study_assistant.py --ask "What is overfitting?"` |
| Web app | `streamlit run app.py` |
| Look inside ChromaDB (chunks, sample embedding) | `python inspect_db.py` |
| Force new embeddings | `python study_assistant.py --rebuild` |
| Run the tests (no internet / key needed) | `python tests.py` |

The first run downloads the embedding model (about 90 MB) and creates the embeddings; later runs load them from ChromaDB.
In the chat you can also type `/summary <topic>`, `/quiz <topic>` and `/sources`.

## Settings (config.py / .env)
- `GROQ_MODEL` - default `llama-3.3-70b-versatile`. For faster answers use `llama-3.1-8b-instant`. If you get a
  "model not found" or "decommissioned" error, pick a current model from https://console.groq.com/docs/models.
- `MIN_SCORE` - chunks less similar than this are ignored (default 0.10).
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K` - chunking and retrieval settings.

## Troubleshooting
- **"No API key found"**: check that `.env` exists in the project folder and contains `GROQ_API_KEY=...`.
- **401 / 404 / 413 / 429 errors**: wrong key, wrong model name, request too large, or rate limit (the free plan has limits; wait a minute). The message says which one.
- **"has no selectable text"**: the PDF is a scan (pictures). Run OCR on it first, then add it again.
- **Wrong or odd answers**: increase `TOP_K` to 5-6 or reduce `CHUNK_SIZE`, then run with `--rebuild`.
- **Delete the database**: close the program and delete the `chroma_db/` folder; it is re-created on the next run.

## Possible viva questions
- **What is RAG and why use it?** It retrieves relevant text first and gives it to the LLM, so answers come from your chapters, hallucination is reduced, and no retraining is needed.
- **What is an embedding?** A vector of numbers that represents the meaning of a text. Similar meanings are close together.
- **What is ChromaDB / a vector database?** A database made to store embeddings and quickly find the ones closest to a query vector. Here it also stores each chunk's text and its chapter/page, and it saves everything on disk.
- **Why chunk the documents?** An embedding of a whole chapter mixes many topics; small chunks give precise matches and fit in the prompt.
- **Why overlap?** So a sentence or idea at the edge of a chunk is not lost.
- **How is similarity computed?** Cosine similarity: vectors are normalised, so the dot product gives it (1.0 = same meaning).
- **Why Groq?** It runs open LLMs (Llama) on special hardware, so answers are very fast, and it has a free plan. Groq only generates text; the embeddings are made locally.
- **How do you reduce hallucination?** The prompt allows only the supplied context, asks for citations, and requires "not found" when the context lacks the answer.
