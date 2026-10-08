"""Embedding lewat endpoint /v1/embeddings yang kompatibel OpenAI (Ollama: bge-m3)."""

from __future__ import annotations

import numpy as np
from openai import OpenAI

from src.config import EMBEDDING_BASE_URL, EMBEDDING_MODEL, OPENAI_API_KEY

_client: OpenAI | None = None


def get_embedder() -> OpenAI:
    """Singleton client untuk embedding."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY or "ollama", base_url=EMBEDDING_BASE_URL)
    return _client


def _normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def embed(text: str) -> np.ndarray:
    """Embed satu string. Return vektor float32 ter-normalisasi."""
    return embed_batch([text])[0]


def embed_batch(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Embed banyak string sekaligus (dipakai saat seed)."""
    client = get_embedder()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts[i : i + batch_size])
        out.extend(d.embedding for d in sorted(response.data, key=lambda d: d.index))
    return _normalize(np.asarray(out, dtype=np.float32))
