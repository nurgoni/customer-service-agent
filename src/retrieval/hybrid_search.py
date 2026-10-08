import re
import sqlite3

import numpy as np
from rank_bm25 import BM25Okapi

from src.db.database import has_vector_index, lookup_product_by_id, vector_search
from src.models.schemas import Product, ProductSearchResult

_bm25: BM25Okapi | None = None
_bm25_ids: list[str] = []

# Diset saat startup: True hanya bila sqlite-vec aktif, index vektor terisi, dan model embedding bisa dipanggil.
_vector_enabled = False


def set_vector_enabled(enabled: bool) -> None:
    global _vector_enabled
    _vector_enabled = enabled


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(conn: sqlite3.Connection) -> None:
    """Bangun index BM25 dari semua produk. Panggil saat startup (dan setelah seed ulang)."""
    global _bm25, _bm25_ids

    rows = conn.execute("SELECT id, name, product_type, description FROM products").fetchall()
    if not rows:
        print("[ bm25 ] tidak ada produk, index kosong", flush=True)
        _bm25, _bm25_ids = None, []
        return

    corpus, ids = [], []
    for r in rows:
        # nama diulang agar bobot kecocokan nama lebih tinggi
        corpus.append(_tokenize(f"{r['name']} {r['name']} {r['product_type']} {r['description']}"))
        ids.append(r["id"])

    _bm25, _bm25_ids = BM25Okapi(corpus), ids
    print(f"[ bm25 ] {len(corpus)} produk ter-index", flush=True)


def _bm25_search(query: str, top_k: int = 20) -> list[tuple[str, float]]:
    if _bm25 is None:
        return []
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = _bm25.get_scores(tokens)
    top = np.argsort(scores)[::-1][:top_k]
    return [(_bm25_ids[i], float(scores[i])) for i in top if scores[i] > 0]


def _rrf_merge(*ranked_lists: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (pid, _) in enumerate(ranked):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(conn: sqlite3.Connection, query: str, top_k: int = 5) -> list[ProductSearchResult]:
    """BM25 + vector (RRF). Otomatis fallback ke BM25 saja bila vector search tidak tersedia."""
    bm25_results = _bm25_search(query, top_k=20)

    vec_results: list[tuple[str, float]] = []
    if _vector_enabled and has_vector_index(conn):
        try:
            from src.retrieval.embedder import embed

            raw = vector_search(conn, embed(query), k=20)
            vec_results = [(pid, 1.0 / (1.0 + dist)) for pid, dist in raw]
        except Exception as e:
            print(f"[ search ] vector search dilewati: {e}", flush=True)
            vec_results = []

    if vec_results:
        merged = _rrf_merge(bm25_results, vec_results)
        method = "hybrid (BM25 + semantic)"
    else:
        merged = bm25_results
        method = "keyword (BM25)"

    results: list[ProductSearchResult] = []
    for pid, score in merged[:top_k]:
        row = lookup_product_by_id(conn, pid)
        if row is None:
            continue
        results.append(
            ProductSearchResult(
                product=Product(
                    id=row["id"],
                    name=row["name"],
                    price=float(row["price"]),
                    description=row["description"],
                    product_type=row["product_type"],
                    stock=int(row["stock"]),
                ),
                score=round(score, 4),
                match_reason=method,
            )
        )
    return results
