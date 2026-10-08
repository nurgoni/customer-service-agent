import json
import sqlite3

from src.db.database import lookup_product_by_id
from src.models.schemas import (
    PriceLookupResult,
    Product,
    ProductDetailResult,
    format_rupiah,
)
from src.retrieval.hybrid_search import hybrid_search

# --- TOOL SCHEMAS (format OpenAI) ---

PRODUCT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_product_catalog",
            "description": (
                "Cari produk di katalog berdasarkan nama, tipe, atau deskripsi. "
                "GUNAKAN tool ini ketika customer menyebut NAMA produk tetapi kamu belum tahu product_id-nya, "
                "atau ketika customer bertanya produk apa saja yang tersedia. "
                "Hasilnya sudah berisi harga dan stok resmi."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Kata kunci: nama produk, tipe, atau kata dari deskripsi.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": (
                "Ambil harga dan stok terkini sebuah produk berdasarkan product_id (mis. SKU001). "
                "JANGAN PERNAH menyebut harga tanpa data dari tool ini atau search_product_catalog."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "ID produk, mis. SKU001."}
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": "Ambil detail lengkap produk (nama, deskripsi, tipe, harga, stok) berdasarkan product_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string", "description": "ID produk yang tepat."}
                },
                "required": ["product_id"],
            },
        },
    },
]


# --- TOOL IMPLEMENTATIONS ---

def tool_get_product_price(conn: sqlite3.Connection, product_id: str) -> dict:
    row = lookup_product_by_id(conn, product_id)
    if row is None:
        return PriceLookupResult(
            found=False, product_id=product_id, error=f"Produk '{product_id}' tidak ditemukan."
        ).model_dump()

    price = round(float(row["price"]), 2)
    return PriceLookupResult(
        found=True,
        product_id=row["id"],
        product_name=row["name"],
        price=price,
        formatted_price=format_rupiah(price),
        stock=int(row["stock"]),
    ).model_dump()


def tool_get_product_detail(conn: sqlite3.Connection, product_id: str) -> dict:
    row = lookup_product_by_id(conn, product_id)
    if row is None:
        return ProductDetailResult(found=False, error=f"Produk '{product_id}' tidak ditemukan.").model_dump()

    product = Product(
        id=row["id"],
        name=row["name"],
        price=round(float(row["price"]), 2),
        description=row["description"],
        product_type=row["product_type"],
        stock=int(row["stock"]),
    )
    return ProductDetailResult(
        found=True, product=product, formatted_price=product.price_display()
    ).model_dump()


def tool_search_product_catalog(conn: sqlite3.Connection, query: str) -> dict:
    results = hybrid_search(conn, query, top_k=5)
    if not results:
        return {"found": False, "message": f"Tidak ada produk yang cocok dengan '{query}'."}

    return {
        "found": True,
        "count": len(results),
        "products": [
            {
                "id": r.product.id,
                "name": r.product.name,
                "product_type": r.product.product_type,
                "description": r.product.description,
                "price": r.product.price,
                "formatted_price": r.product.price_display(),
                "stock": r.product.stock,
            }
            for r in results
        ],
    }


# --- DISPATCHER ---

TOOL_IMPLEMENTATION = {
    "search_product_catalog": tool_search_product_catalog,
    "get_product_price": tool_get_product_price,
    "get_product_detail": tool_get_product_detail,
}


def execute_tool(conn: sqlite3.Connection, tool_name: str, tool_input: dict) -> str:
    fn = TOOL_IMPLEMENTATION.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return json.dumps(fn(conn, **tool_input), default=str, ensure_ascii=False)
    except TypeError:
        return json.dumps({"error": f"Invalid arguments for {tool_name}"})
