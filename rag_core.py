"""
rag_core.py  -  Speed-optimized backend for fast CPU inference.
"""

import hashlib
import os
import re
from pathlib import Path

import chromadb
import ollama
import pymupdf as fitz

# ---------- Hardware & Model Configurations ----------
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:0.5b"          # You can also use "qwen2.5:0.5b" for maximum speed
CHUNK_SIZE = 400                   # Smaller chunks for faster processing
CHUNK_OVERLAP = 80
TOP_K = 2                          # Keep at 2 to minimize context load speed
DB_PATH = "./chroma_db"
NOT_FOUND_TOKEN = "NOT_FOUND"
KEEP_ALIVE = -1               # Keeps model in RAM permanently (no loading lag)

# Auto-detect CPU cores to maximize speed
CPU_CORES = os.cpu_count() or 4


class RagError(Exception):
    """Friendly error message for the UI."""


# ---------- 1. Load PDF ----------
def load_pdf(path: Path) -> list[tuple[int, str]]:
    """Return a list of (page_number, text)."""
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise RagError(f"Could not open the PDF ({e}).")

    if doc.needs_pass:
        raise RagError("This PDF is password protected.")

    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append((i, text))

    if not pages:
        raise RagError(
            "No text found. This PDF may be scanned images (it would need OCR first)."
        )
    return pages


# ---------- 2. Chunk Text ----------
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks cleanly at spaces."""
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start + size // 2:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------- 3. Embeddings ----------
def embed(texts: list[str]) -> list[list[float]]:
    try:
        response = ollama.embed(
            model=EMBED_MODEL, 
            input=texts, 
            keep_alive=KEEP_ALIVE
        )
        return response["embeddings"]
    except Exception as e:
        raise RagError(
            f"Could not talk to Ollama ({e}). "
            f"Is Ollama running? Did you run: ollama pull {EMBED_MODEL}?"
        )


# ---------- 4. ChromaDB Storage ----------
def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def build_index(path: Path, display_name: str):
    """Create or reuse a Chroma collection for the uploaded PDF."""
    path = Path(path)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=f"pdf_{file_hash(path)}",
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        return collection, True

    pages = load_pdf(path)

    ids, docs, metas = [], [], []
    for page_num, text in pages:
        for i, chunk in enumerate(chunk_text(text)):
            ids.append(f"p{page_num}_c{i}")
            docs.append(chunk)
            metas.append({"source": display_name, "page": page_num})

    batch = 32
    for i in range(0, len(docs), batch):
        collection.add(
            ids=ids[i:i + batch],
            documents=docs[i:i + batch],
            metadatas=metas[i:i + batch],
            embeddings=embed(docs[i:i + batch]),
        )
    return collection, False


# ---------- 5. Retrieval & Optimized Streaming ----------
def retrieve(collection, question: str, top_k: int = TOP_K):
    """Retrieve top matching chunks."""
    q_emb = embed([question])[0]
    result = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    if result and result["documents"] and result["documents"][0]:
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            hits.append({"text": doc, "meta": meta, "distance": dist})
    return hits


async def answer_stream(question: str, hits: list[dict], on_token) -> str:
    """Stream LLM response with hardware-level speed optimizations."""
    context = "\n\n".join(
        f"[{i}] (page {h['meta']['page']}) {h['text']}" for i, h in enumerate(hits, start=1)
    )
    system = (
        "Answer the user's question using ONLY the context provided. "
        "Cite the passage like [1] at the end. Keep it to 1 or 2 sentences max. "
        f"If not in context, output exactly: {NOT_FOUND_TOKEN}"
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        client = ollama.AsyncClient()
        stream = await client.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": 0.0,
                "num_thread": CPU_CORES,  # Force usage of all CPU cores
                "num_ctx": 2048,           # Reduce context size for 3x speed boost
                "num_predict": 128,        # Cap max generation tokens (short direct answers)
            },
            stream=True,
            keep_alive=KEEP_ALIVE,
        )

        full = ""
        showing = False

        async for part in stream:
            piece = part["message"]["content"]
            full += piece

            if not showing:
                start = full.lstrip()
                if start.startswith(NOT_FOUND_TOKEN):
                    break
                if NOT_FOUND_TOKEN.startswith(start):
                    continue
                await on_token(start)
                showing = True
            else:
                await on_token(piece)

        return full.strip()
    except Exception as e:
        raise RagError(
            f"Could not talk to Ollama ({e}). Did you run: ollama pull {CHAT_MODEL}?"
        )


def pick_source(reply: str, hits: list[dict]) -> tuple[int, dict]:
    """Identify which citation the LLM used."""
    cited = [int(n) for n in re.findall(r"\[(\d+)\]", reply)]
    for n in cited:
        if 1 <= n <= len(hits):
            return n, hits[n - 1]
    return 1, hits[0]