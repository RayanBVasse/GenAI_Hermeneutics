"""Tests for vector store structure and retrieval math."""
import json
from pathlib import Path
import numpy as np


class TestVectorStore:
    def test_stored_vectors_exist(self):
        store_dir = Path("data/vector_store")
        files = list(store_dir.glob("*.json"))
        assert len(files) > 0, "No vector store files found"

    def test_stored_vectors_structure(self):
        store_dir = Path("data/vector_store")
        files = list(store_dir.glob("*.json"))
        with open(files[0], encoding="utf-8") as f:
            data = json.load(f)
        for key in ["namespace", "ids", "documents", "metadatas", "embeddings"]:
            assert key in data, f"Missing key: {key}"
        assert len(data["ids"]) == len(data["embeddings"])
        assert len(data["ids"]) == len(data["documents"])
        assert len(data["ids"]) == len(data["metadatas"])

    def test_embedding_dimensions(self):
        store_dir = Path("data/vector_store")
        files = list(store_dir.glob("*.json"))
        with open(files[0], encoding="utf-8") as f:
            data = json.load(f)
        for emb in data["embeddings"][:5]:
            assert len(emb) == 1536

    def test_metadata_has_chapter_info(self):
        store_dir = Path("data/vector_store")
        files = list(store_dir.glob("*.json"))
        with open(files[0], encoding="utf-8") as f:
            data = json.load(f)
        for meta in data["metadatas"][:5]:
            assert "chapter_number" in meta
            assert "chapter_title" in meta
            assert "position_in_chapter" in meta

    def test_cosine_similarity_self(self):
        """Cosine similarity of a vector with itself should be ~1.0."""
        a = np.random.randn(1536).astype(np.float32)
        a = a / np.linalg.norm(a)
        sim = float(np.dot(a, a))
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal(self):
        """Orthogonal vectors should have ~0 cosine similarity."""
        a = np.zeros(1536, dtype=np.float32)
        b = np.zeros(1536, dtype=np.float32)
        a[0] = 1.0
        b[1] = 1.0
        sim = float(np.dot(a, b))
        assert abs(sim) < 1e-6
