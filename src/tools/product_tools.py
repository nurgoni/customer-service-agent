# src/tools/product_tools.py

import json
import sqlite3
from src.db.database import lookup_product_by_id, search_products_by_keyword
from src.models.schemas import PriceLookupResult, ProductDetailResult, Product


# ═══════════════════════════════════════════════════════════════
# TOOL SCHEMAS — FORMAT OPENAI
# ═══════════════════════════════════════════════════════════════

PRODUCT_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": (
                "Look up the current price and stock of a product by its ID. "
                "ALWAYS use this when the customer asks about price or cost. "
                "NEVER state a price without calling this tool first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The product ID (e.g. SKU001)"
                    }
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_detail",
            "description": (
                "Look up full product details (name, description, price, "
                "stock, type) by product ID. Use when the customer asks "
                "about a specific product."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "The exact product ID to look up."
                    }
                },
                "required": ["product_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_product_catalog",
            "description": (
                "Search products by keyword across name, type, and description. "
                "Use when the customer asks what products are available, "
                "searches by category, or describes what they need."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search keyword: product name, type, or description words"
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════
# TOOL IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════

def tool_get_product_price(conn: sqlite3.Connection, product_id: str) -> dict:
    row = lookup_product_by_id(conn, product_id)
    if row is None:
        return PriceLookupResult(
            found=False, product_id=product_id,
            error=f"Product '{product_id}' not found."
        ).model_dump()

    price = round(float(row["price"]), 2)
    return PriceLookupResult(
        found=True, product_id=product_id,
        product_name=row["name"], price=price,
        formatted_price=f"Rp {price:,.0f}",
        stock=int(row["stock"])
    ).model_dump()


def tool_get_product_detail(conn: sqlite3.Connection, product_id: str) -> dict:
    row = lookup_product_by_id(conn, product_id)
    if row is None:
        return ProductDetailResult(
            found=False, error=f"Product '{product_id}' not found."
        ).model_dump()

    return ProductDetailResult(
        found=True,
        product=Product(
            id=row["id"], name=row["name"],
            price=round(float(row["price"]), 2),
            description=row["description"],
            product_type=row["product_type"],
            stock=int(row["stock"]),
        )
    ).model_dump()


def tool_search_product_catalog(conn: sqlite3.Connection, query: str) -> dict:
    rows = search_products_by_keyword(conn, query, limit=5)
    if not rows:
        return {"found": False, "message": f"No products matching '{query}'"}

    products = []
    for r in rows:
        p = {
            "id": r["id"], "name": r["name"],
            "price": round(float(r["price"]), 2),
            "formatted_price": f"Rp {float(r['price']):,.0f}",
            "product_type": r["product_type"],
            "description": r["description"],
            "stock": int(r["stock"]),
        }
        products.append(p)

    return {"found": True, "count": len(products), "products": products}


# ═══════════════════════════════════════════════════════════════
# DISPATCHER
# ═══════════════════════════════════════════════════════════════

_TOOL_FUNCTIONS = {
    "get_product_price": tool_get_product_price,
    "get_product_detail": tool_get_product_detail,
    "search_product_catalog": tool_search_product_catalog,
}

def execute_tool(conn: sqlite3.Connection, tool_name: str, tool_input: dict) -> str:
    fn = _TOOL_FUNCTIONS.get(tool_name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        result = fn(conn, **tool_input)
        return json.dumps(result, default=str, ensure_ascii=False)
    except TypeError:
        return json.dumps({"error": f"Invalid arguments for {tool_name}"})