"""
Load CSV/Excel produk ke database (+ embedding bila sqlite-vec tersedia).

Pemakaian (jalankan dari folder root project, yaitu yang berisi folder `src`):
    python -m src.db.seed                              # default: data/products.csv
    python -m src.db.seed --file data/products.xlsx

Jalankan ulang seed setiap kali data produk ATAU model embedding diganti.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import BASE_DIR, EMBEDDING_MODEL
from src.db import database
from src.db.database import _serialize_vector, get_connection, init_db, recreate_vector_table

REQUIRED_COLUMNS = {"id", "name", "price"}


def load_dataframe(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Format file tidak didukung: {ext}")

    df.columns = [c.strip().lower() for c in df.columns]
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan: {sorted(missing)}. Kolom yang ada: {list(df.columns)}")

    for col, default in (("description", ""), ("product_type", ""), ("stock", 0)):
        if col not in df.columns:
            df[col] = default
    df["description"] = df["description"].fillna("")
    df["product_type"] = df["product_type"].fillna("")
    df["stock"] = df["stock"].fillna(0).astype(int)
    return df


def seed_from_dataframe(df: pd.DataFrame) -> None:
    conn = get_connection()
    init_db(conn)

    for _, row in df.iterrows():
        conn.execute(
            """
            INSERT OR REPLACE INTO products (id, name, price, description, product_type, stock)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(row["id"]).strip(),
                str(row["name"]).strip(),
                float(row["price"]),
                str(row["description"]),
                str(row["product_type"]),
                int(row["stock"]),
            ),
        )
    conn.commit()
    print(f"[seed] {len(df)} produk dimasukkan ke tabel products")

    if not database.VEC_AVAILABLE:
        print("[seed] sqlite-vec tidak tersedia -> embedding dilewati (BM25 tetap berfungsi)")
        conn.close()
        return

    try:
        from src.retrieval.embedder import embed_batch

        texts = [
            f"{r['name']} - {r['product_type']} - {r['description']}" for _, r in df.iterrows()
        ]
        print(f"[seed] membuat embedding dengan model '{EMBEDDING_MODEL}'...")
        vectors = embed_batch(texts)
        dim = int(vectors.shape[1])

        recreate_vector_table(conn, dim)
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            conn.execute(
                "INSERT INTO product_vecs (rowid, embedding) VALUES (?, ?)",
                (i, _serialize_vector(vectors[i - 1])),
            )
            conn.execute(
                "INSERT INTO product_rowids (rowid, pid) VALUES (?, ?)",
                (i, str(row["id"]).strip()),
            )
        conn.commit()
        print(f"[seed] {len(vectors)} embedding tersimpan (dimensi {dim})")
    except Exception as e:
        print(f"[seed] embedding dilewati: {e}")
        print("[seed] pastikan Ollama berjalan dan model embedding sudah di-pull; BM25 tetap berfungsi")
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=str(BASE_DIR / "data" / "products.csv"))
    args = parser.parse_args()
    seed_from_dataframe(load_dataframe(Path(args.file)))
