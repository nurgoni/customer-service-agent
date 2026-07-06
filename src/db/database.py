import sqlite3
import struct
import numpy as np
from pathlib import Path


DB_PATH = Path("data/cs_agent.db")


def _serialize_vector(v: np.ndarray) -> bytes:
    """
    Pack a float32 numpy vector into the binary format sqlite-vec expects.
    """
    return struct.pack(f"{len(v)}f", *v.astype(np.float32))


def get_connection() -> sqlite3.Connection:
    """
    Open (or reuse) the SQLite connection with sqlite-vec loaded.
    WAL mode: readers never block writers; writers never block readers.
    This is essential for an async FastAPI server where multiple
    requests may hit the DB simultaneously.
    """


def init_db(conn: sqlite3.Connection) -> None:
    """
    Create schema if it doesn't exist.

    Two tables:
        products     - relationale, the source of truth for price stock
        product_vecs - virtual vec0 table for ANN vector search 
    
    The vec0 table stores 384-dim float32 embeddings (matching
    all-MiniLM-L6-v2 output dimension). Each row is keyed by the
    products.id rowid so a join retrieves the full product record.
    """


def vector_search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    k: int = 20
) -> list[tuple[str, float]]:
    """
    ANN search over the vec0 table.
    Returns list of (product_id, distance) - lower distance = more similar.
    """
    query_bytes = _serialize_vector(query_embedding)
    rows = conn.execute("""
        SELECT pr.pid, v.distance
        FROM product_vecs v
        JOIN product_rowids pr ON pr.rowid = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
    """, (query_bytes, k)).fetchall()
    return [(r["pid"], r["distance"]) for r in rows]
