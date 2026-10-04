# src/retrieval/hybrid_search.py

import sqlite3
import numpy as np
from rank_bm25 import BM25Okapi

from src.db.database import vector_search, lookup_product_by_id
from src.retrieval.embedder import embed
from src.models.schemas import Product, ProductSearchResult


# ═══════════════════════════════════════════════════════════════
# BM25 INDEX (in-memory, rebuilt on startup)
# ═══════════════════════════════════════════════════════════════

_bm25: BM25Okapi | None = None
_bm25_ids: list[str] = []


def build_bm25_index(conn: sqlite3.Connection) -> None:
    """Build BM25 index from all products. Called once at startup."""
    global _bm25, _bm25_ids

    rows = conn.execute(
        "SELECT id, name, product_type, description FROM products"
    ).fetchall()

    if not rows:
        print("[ bm25 ] no products found, index empty")
        _bm25 = None
        _bm25_ids = []
        return

    corpus = []
    _bm25_ids = []
    for r in rows:
        doc = f"{r['name']} {r['product_type']} {r['description']}".lower().split()
        corpus.append(doc)
        _bm25_ids.append(r["id"])

    _bm25 = BM25Okapi(corpus)
    print(f"[ bm25 ] indexed {len(corpus)} products")


def _bm25_search(query: str, top_k: int = 20) -> list[tuple[str, float]]:
    """Return list of (product_id, bm25_score)."""
    if _bm25 is None:
        return []
    tokens = query.lower().split()
    scores = _bm25.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [
        (_bm25_ids[i], float(scores[i]))
        for i in top_indices if scores[i] > 0
    ]


# ═══════════════════════════════════════════════════════════════
# RECIPROCAL RANK FUSION
# ═══════════════════════════════════════════════════════════════

def _rrf_merge(
    *ranked_lists: list[tuple[str, float]],
    k: int = 60
) -> list[tuple[str, float]]:
    """
    Reciprocal Rank Fusion: menggabungkan multiple ranked lists.
    Setiap list berupa [(id, score), ...] sudah diurutkan desc.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (pid, _) in enumerate(ranked):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ═══════════════════════════════════════════════════════════════
# HYBRID SEARCH (public API)
# ═══════════════════════════════════════════════════════════════

def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5
) -> list[ProductSearchResult]:
    """
    Combine BM25 (keyword) + vector (semantic) via RRF.
    Falls back to BM25-only if vector search is unavailable.
    """
    # BM25 leg
    bm25_results = _bm25_search(query, top_k=20)

    # Vector leg
    try:
        query_vec = embed(query)
        vec_results = vector_search(conn, query_vec, k=20)
        # vector_search returns (id, distance) — lower is better
        # flip to score so RRF works (sort desc)
        vec_results = [(pid, 1.0 / (1.0 + dist)) for pid, dist in vec_results]
    except Exception:
        vec_results = []

    # Merge
    if vec_results:
        merged = _rrf_merge(bm25_results, vec_results)
        method = "hybrid (BM25 + semantic)"
    else:
        merged = bm25_results
        method = "keyword (BM25)"

    # Hydrate top-k results from DB
    results = []
    for pid, score in merged[:top_k]:
        row = lookup_product_by_id(conn, pid)
        if row is None:
            continue
        results.append(ProductSearchResult(
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
        ))

    return results