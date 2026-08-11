# src/db/seed.py

"""
Pemakaian:
    python -m src.db.seed --file data/products.csv
    python -m src.db.seed --file data/products.xlsx
"""

import argparse
import pandas as pd
import numpy as np

from src.db.database import get_connection, init_db, _serialize_vector
from src.retrieval.embedder import embed_batch


SAMPLE_DATA = [
    ("SKU001", "Nike Air Max 270",      2199000, "Sepatu lari ringan dengan bantalan Air Max",         "Sepatu",     45),
    ("SKU002", "Adidas Ultraboost 23",  2899000, "Sepatu lari premium dengan teknologi Boost",         "Sepatu",     30),
    ("SKU003", "Uniqlo Airism T-Shirt",  199000, "Kaos teknologi pendingin, cepat kering, anti bau",   "Pakaian",   200),
    ("SKU004", "Samsung Galaxy S24",   13999000, "Smartphone flagship AI Galaxy, kamera 200MP",        "Elektronik", 15),
    ("SKU005", "Sony WH-1000XM5",       4999000, "Headphone wireless noise cancelling 30 jam battery", "Elektronik", 25),
    ("SKU006", "Erigo Flannel Shirt",    259000, "Kemeja flannel kotak-kotak casual katun premium",    "Pakaian",    80),
]


def seed_from_dataframe(df: pd.DataFrame) -> None:
    """
    Expects columns: id, name, price, description, product_type, stock
    """
    conn = get_connection()
    init_db(conn)

    # ── Insert products ──
    for _, row in df.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO products (id, name, price, description, product_type, stock)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            str(row["id"]),
            str(row["name"]),
            float(row["price"]),
            str(row.get("description", "")),
            str(row.get("product_type", "")),
            int(row.get("stock", 0)),
        ))
    conn.commit()
    print(f"✅ {len(df)} products inserted into database")

    # ── Generate & store embeddings ──
    texts = [
        f"{row['name']} - {row.get('product_type', '')} - {row.get('description', '')}"
        for _, row in df.iterrows()
    ]

    try:
        print("⏳ Generating embeddings...")
        vectors = embed_batch(texts)

        for i, (_, row) in enumerate(df.iterrows()):
            vec_bytes = _serialize_vector(vectors[i])
            conn.execute(
                "INSERT OR REPLACE INTO product_vecs (rowid, embedding) VALUES (?, ?)",
                (i + 1, vec_bytes)
            )
            conn.execute(
                "INSERT OR REPLACE INTO product_rowids (rowid, pid) VALUES (?, ?)",
                (i + 1, str(row["id"]))
            )
        conn.commit()
        print(f"✅ {len(vectors)} embeddings stored")
    except Exception as e:
        print(f"⚠️  Embedding storage skipped: {e}")

    conn.close()


def seed_sample():
    """Load built-in sample data."""
    df = pd.DataFrame(SAMPLE_DATA, columns=[
        "id", "name", "price", "description", "product_type", "stock"
    ])
    seed_from_dataframe(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default=None)
    args = parser.parse_args()

    if args.file:
        ext = args.file.rsplit(".", 1)[-1].lower()
        if ext == "csv":
            df = pd.read_csv(args.file)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(args.file)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        seed_from_dataframe(df)
    else:
        print("No file provided, using sample data...")
        seed_sample()