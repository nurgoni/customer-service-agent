# src/db/database.py

import sqlite3
import struct
import numpy as np
from pathlib import Path

from src.config import DB_PATH

# ═══════════════════════════════════════════════════════════════
# CONNECTION
# ═══════════════════════════════════════════════════════════════

def _serialize_vector(v: np.ndarray) -> bytes:
    return struct.pack(f"{len(v)}f", *v.astype(np.float32))


def get_connection() -> sqlite3.Connection:
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row          # ← akses kolom by name
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    # Load sqlite-vec jika tersedia (opsional, untuk vector search)
    try:
        import sqlite_vec
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except (ImportError, Exception):
        print("[ db ] sqlite-vec not available, vector search disabled")

    return conn


# ═══════════════════════════════════════════════════════════════
# SCHEMA
# ═══════════════════════════════════════════════════════════════

def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS products (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            price        REAL NOT NULL CHECK(price > 0),
            description  TEXT NOT NULL DEFAULT '',
            product_type TEXT NOT NULL DEFAULT '',
            stock        INTEGER NOT NULL DEFAULT 0
        );
    """)

    # Coba buat vec0 table (skip jika sqlite-vec tidak ter-load)
    try:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS product_vecs
            USING vec0(embedding float[384])
        """)
        # Mapping rowid → product.id
        conn.execute("""
            CREATE TABLE IF NOT EXISTS product_rowids (
                rowid INTEGER PRIMARY KEY,
                pid   TEXT NOT NULL REFERENCES products(id)
            )
        """)
    except Exception:
        print("[ db ] skipping vector table creation")

    conn.commit()


# ═══════════════════════════════════════════════════════════════
# QUERY HELPERS
# ═══════════════════════════════════════════════════════════════

def lookup_product_by_id(
    conn: sqlite3.Connection,
    product_id: str
) -> sqlite3.Row | None:
    """Exact lookup by primary key."""
    return conn.execute(
        "SELECT * FROM products WHERE id = ?", (product_id,)
    ).fetchone()


def search_products_by_keyword(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10
) -> list[sqlite3.Row]:
    """
    Fuzzy keyword search across name, description, product_type.
    Ini adalah fallback jika vector search tidak tersedia.
    """
    wildcard = f"%{query}%"
    return conn.execute("""
        SELECT * FROM products
        WHERE LOWER(name)         LIKE LOWER(?)
           OR LOWER(description)  LIKE LOWER(?)
           OR LOWER(product_type) LIKE LOWER(?)
        LIMIT ?
    """, (wildcard, wildcard, wildcard, limit)).fetchall()


def vector_search(
    conn: sqlite3.Connection,
    query_embedding: np.ndarray,
    k: int = 20
) -> list[tuple[str, float]]:
    """ANN search over vec0 table."""
    query_bytes = _serialize_vector(query_embedding)
    try:
        rows = conn.execute("""
            SELECT pr.pid, v.distance
            FROM product_vecs v
            JOIN product_rowids pr ON pr.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
        """, (query_bytes, k)).fetchall()
        return [(r["pid"], r["distance"]) for r in rows]
    except Exception:
        return []