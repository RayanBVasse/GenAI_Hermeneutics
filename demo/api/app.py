"""
Demo API for GenAI Hermeneutics — Canon Pack vs Vanilla RAG comparison.

Endpoints:
    GET  /books             List available books
    GET  /examples/{slug}   Pre-computed A/B examples for a book
    POST /ask               Live question (rate-limited)
    GET  /health            Health check
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Path setup — add repo root so framework/ and study/ are importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="GenAI Hermeneutics Demo",
    description="Canon Pack vs Vanilla RAG — interactive research demo",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # GitHub Pages + local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pre-computed examples (loaded once at startup)
# ---------------------------------------------------------------------------
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

_books_index: list[dict] = []
_examples_cache: dict[str, dict] = {}


def _load_examples():
    global _books_index
    index_path = EXAMPLES_DIR / "_index.json"
    if index_path.exists():
        _books_index = json.loads(index_path.read_text(encoding="utf-8"))
    for f in EXAMPLES_DIR.glob("*.json"):
        if f.name.startswith("_"):
            continue
        slug = f.stem
        _examples_cache[slug] = json.loads(f.read_text(encoding="utf-8"))


_load_examples()

# ---------------------------------------------------------------------------
# Rate limiter — 10 requests per IP per day (in-memory)
# ---------------------------------------------------------------------------
_rate_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_DAY", "10"))
RATE_WINDOW = 86400  # 24 hours


def _check_rate_limit(ip: str) -> bool:
    now = time.time()
    timestamps = _rate_store[ip]
    # Prune old entries
    _rate_store[ip] = [t for t in timestamps if now - t < RATE_WINDOW]
    if len(_rate_store[ip]) >= RATE_LIMIT:
        return False
    _rate_store[ip].append(now)
    return True


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    book_slug: str
    question: str = Field(..., min_length=10, max_length=500)


class AskResponse(BaseModel):
    question: str
    vanilla_rag_response: str
    canon_pack_response: str
    retrieved_chunks: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/books")
def list_books():
    """Return the list of available books with metadata."""
    return {"books": _books_index}


@app.get("/examples/{slug}")
def get_examples(slug: str):
    """Return pre-computed A/B comparison examples for a book."""
    if slug not in _examples_cache:
        raise HTTPException(status_code=404, detail=f"Book '{slug}' not found")
    return _examples_cache[slug]


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest, request: Request):
    """
    Ask a live question — both companions answer the same question.
    Rate-limited to protect API costs.
    """
    # Rate limit
    client_ip = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Daily rate limit reached. Try again tomorrow or browse pre-computed examples.",
        )

    # Check the book exists
    book_meta = None
    for b in _books_index:
        if b["slug"] == req.book_slug:
            book_meta = b
            break
    if not book_meta:
        raise HTTPException(status_code=404, detail=f"Book '{req.book_slug}' not found")

    author_slug = book_meta["author"].lower().replace(" ", "_").replace(".", "")
    # Use the export script's author_slug mapping
    AUTHOR_SLUGS = {
        "meditations": "marcus_aurelius",
        "the_art_of_war": "sun_tzu",
        "content": "cory_doctorow",
        "being_no_one": "thomas_metzinger",
        "the_fourth_culture_identity_without_borders": "rayan_vasse",
    }
    author_slug = AUTHOR_SLUGS.get(req.book_slug, author_slug)

    # Check that vector store exists
    vs_path = REPO_ROOT / "data" / "vector_store" / f"{author_slug}_{req.book_slug}.json"
    if not vs_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Vector store not available for this book. Browse pre-computed examples instead.",
        )

    try:
        from study.companions.vanilla_rag import VanillaRAGCompanion
        from study.companions.canon_pack import CanonPackCompanion

        # Vanilla RAG
        vanilla = VanillaRAGCompanion(
            author_slug=author_slug,
            book_slug=req.book_slug,
            book_title=book_meta["title"],
            author_name=book_meta["author"],
        )
        vanilla_resp = vanilla.ask(req.question)

        # Canon Pack
        canon = CanonPackCompanion(
            author_slug=author_slug,
            book_slug=req.book_slug,
        )
        canon_resp = canon.ask(req.question)

        return AskResponse(
            question=req.question,
            vanilla_rag_response=vanilla_resp.answer,
            canon_pack_response=canon_resp.answer,
            retrieved_chunks=vanilla_resp.retrieved_chunks,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "books_loaded": len(_books_index),
        "live_mode": _check_live_mode(),
    }


def _check_live_mode() -> bool:
    """Check if at least one vector store is available for live questions."""
    vs_dir = REPO_ROOT / "data" / "vector_store"
    if not vs_dir.exists():
        return False
    return any(vs_dir.glob("*.json"))
