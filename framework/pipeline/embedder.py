"""
Stage 3: Embedding + Vector Store
Embeds chunks and stores them with a lightweight numpy-based vector store.
Each book gets its own namespace (JSON file on disk).

ChromaDB segfaults on Python 3.14, so we use a simple cosine-similarity
store backed by numpy + JSON. Production will use Qdrant.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from framework.config import (
    OPENAI_API_KEY,
    EMBEDDING_MODEL,
    DATA_DIR,
)
from framework.models.chunk import Chunk


def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _make_namespace(author_slug: str, book_slug: str) -> str:
    return f"{author_slug}_{book_slug}"


def _store_path(namespace: str) -> Path:
    store_dir = DATA_DIR / "vector_store"
    store_dir.mkdir(parents=True, exist_ok=True)
    return store_dir / f"{namespace}.json"


def _slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = slug.replace("'", "").replace('"', "")
    slug = "".join(c if c.isalnum() or c == " " else " " for c in slug)
    slug = "_".join(slug.split())
    return slug[:60]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between vector a and matrix b."""
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return b_norm @ a_norm


def embed_chunks(
    chunks: list[Chunk],
    author_name: str,
    book_title: str,
) -> dict:
    """
    Embed all chunks and store to disk as JSON + numpy arrays.
    Returns metadata about the operation.
    """
    author_slug = _slugify(author_name)
    book_slug = _slugify(book_title)
    namespace = _make_namespace(author_slug, book_slug)

    openai_client = _get_openai_client()

    all_embeddings = []
    all_documents = []
    all_metadatas = []
    all_ids = []

    # Batch embedding (100 at a time to avoid rate limits)
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.text for c in batch]

        response = openai_client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL,
        )

        for chunk, emb_data in zip(batch, response.data):
            chunk_id = hashlib.md5(
                f"{namespace}_{chunk.chunk_index}".encode()
            ).hexdigest()

            all_ids.append(chunk_id)
            all_embeddings.append(emb_data.embedding)
            all_documents.append(chunk.text)
            all_metadatas.append({
                "chapter_number": chunk.chapter_number,
                "chapter_title": chunk.chapter_title,
                "position_in_chapter": chunk.position_in_chapter,
                "chunk_index": chunk.chunk_index,
                "is_summary": chunk.is_summary,
            })

    # Save to disk
    store_data = {
        "namespace": namespace,
        "author": author_name,
        "book": book_title,
        "ids": all_ids,
        "documents": all_documents,
        "metadatas": all_metadatas,
        "embeddings": all_embeddings,
    }
    path = _store_path(namespace)
    path.write_text(json.dumps(store_data), encoding="utf-8")

    return {
        "namespace": namespace,
        "author_slug": author_slug,
        "book_slug": book_slug,
        "chunks_embedded": len(all_ids),
    }


def _load_store(namespace: str) -> dict:
    """Load a vector store from disk."""
    path = _store_path(namespace)
    if not path.exists():
        raise FileNotFoundError(f"No vector store found for namespace: {namespace}")
    return json.loads(path.read_text(encoding="utf-8"))


def retrieve(
    author_slug: str,
    book_slug: str,
    query: str,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a query using cosine similarity.
    """
    namespace = _make_namespace(author_slug, book_slug)
    store = _load_store(namespace)
    openai_client = _get_openai_client()

    # Embed the query
    response = openai_client.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL,
    )
    query_vec = np.array(response.data[0].embedding, dtype=np.float32)

    # Compute similarities
    embeddings_matrix = np.array(store["embeddings"], dtype=np.float32)
    similarities = _cosine_similarity(query_vec, embeddings_matrix)

    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]

    retrieved = []
    for idx in top_indices:
        retrieved.append({
            "text": store["documents"][idx],
            "chapter_number": store["metadatas"][idx]["chapter_number"],
            "chapter_title": store["metadatas"][idx]["chapter_title"],
            "score": float(similarities[idx]),
            "position": store["metadatas"][idx]["position_in_chapter"],
        })

    return retrieved
