import logging
import sqlite3
import struct
from pathlib import Path

import numpy as np

from src.config import DB_PATH

logger = logging.getLogger("cs_agent.db")

# Diset True kalau ekstensi sqlite-vec berhasil dimuat pada koneksi.
VEC_AVAILABLE = False


def _serialize_vector(v: np.ndarray) -> bytes:
    """Pack float32 vector ke format biner yang diminta sqlite-vec."""
    return struct.pack(f"{len(v)}f", *v.astype(np.float32))


def get_connection() -> sqlite3.Connection:
    """
    Buka koneksi SQLite (WAL mode) dan coba muat sqlite-vec.
    Jika sqlite-vec tidak tersedia, sistem tetap jalan dengan BM25 saja.
    """
    global VEC_AVAILABLE

    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        VEC_AVAILABLE = True
    except Exception as e:  # ImportError, AttributeError (build tanpa extension), dll
        VEC_AVAILABLE = False
        logger.warning("sqlite-vec tidak tersedia (%s); vector search dimatikan, pakai BM25 saja", e)

    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """
    Buat schema relasional: products (source of truth) + product_rowids.
    Tabel vektor `product_vecs` dibuat oleh seed (recreate_vector_table) mengikuti dimensi model embedding.
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS products (
            id           TEXT PRIMARY KEY,
            name         TEXT NOT NULL,
            price        REAL NOT NULL CHECK(price > 0),
            description  TEXT NOT NULL DEFAULT '',
            product_type TEXT NOT NULL DEFAULT '',
            stock        INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS product_rowids (
            rowid INTEGER PRIMARY KEY,
            pid   TEXT NOT NULL REFERENCES products(id)
        );
        """
    )
    conn.commit()


def recreate_vector_table(conn: sqlite3.Connection, dim: int) -> None:
    """Buat ulang tabel vektor dengan dimensi yang sesuai model embedding saat ini."""
    conn.execute("DROP TABLE IF EXISTS product_vecs")
    conn.execute(f"CREATE VIRTUAL TABLE product_vecs USING vec0(embedding float[{int(dim)}])")
    conn.execute("DELETE FROM product_rowids")
    conn.commit()


def has_vector_index(conn: sqlite3.Connection) -> bool:
    """True bila sqlite-vec aktif dan tabel vektor sudah diisi oleh seed."""
    if not VEC_AVAILABLE:
        return False
    try:
        return conn.execute("SELECT COUNT(*) FROM product_rowids").fetchone()[0] > 0 and bool(
            conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'product_vecs'").fetchone()
        )
    except Exception:
        return False


def lookup_product_by_id(conn: sqlite3.Connection, product_id: str) -> sqlite3.Row | None:
    """Exact lookup (case-insensitive) berdasarkan primary key."""
    return conn.execute(
        "SELECT * FROM products WHERE UPPER(id) = UPPER(?)", (product_id.strip(),)
    ).fetchone()


def search_products_by_keyword(conn: sqlite3.Connection, query: str, limit: int = 10) -> list[sqlite3.Row]:
    """Fallback pencarian LIKE di name/description/product_type."""
    wildcard = f"%{query}%"
    return conn.execute(
        """
        SELECT * FROM products
        WHERE LOWER(name) LIKE LOWER(?)
           OR LOWER(description) LIKE LOWER(?)
           OR LOWER(product_type) LIKE LOWER(?)
        LIMIT ?
        """,
        (wildcard, wildcard, wildcard, limit),
    ).fetchall()


def vector_search(conn: sqlite3.Connection, query_embedding: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
    """ANN search. Return [(product_id, distance)], distance lebih kecil = lebih mirip."""
    if not VEC_AVAILABLE:
        return []
    try:
        rows = conn.execute(
            """
            SELECT pr.pid AS pid, v.distance AS distance
            FROM product_vecs v
            JOIN product_rowids pr ON pr.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_vector(query_embedding), k),
        ).fetchall()
        return [(r["pid"], r["distance"]) for r in rows]
    except Exception as e:
        logger.warning("vector_search gagal: %s", e)
        return []