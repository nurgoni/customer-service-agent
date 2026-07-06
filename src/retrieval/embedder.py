from __future__ import annotations
from sentence_transformers import SentenceTransformer
import numpy as np


_MODEL_NAME = "all-MiniLM-L6-v2"
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    """
    Return the singleton embedding model, loading it on first call.
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_MODEL_NAME)
    return _embedder

def embed(text: str) -> np.ndarray:
    """
    Embed a single string. Returns a normalized float32 vector.
    """
    model = get_embedder()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype=np.float32)

def embed_batch(texts: list[str]) -> np.ndarray:
    """
    Batch embed for seeding the database efficiently.
    """
    model = get_embedder()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=True
    )
    return np.asarray(vecs, dtype=np.float32)
