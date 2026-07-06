import json
import sqlite3
from src.db.database import lookup_product_by_id
from src.models.schemas import PriceLookupResult, ProductDetailResult, Product


# --- TOOL SCHEMAS ---

PRODUCT_TOOL_SCHEMAS = [
    {
        "name": "get_product_price",
        "description": (

        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": (

                    ),
                }
            },
            "required": ["product_id"],
        },
    },
    {
        "name": "get_product_detail",
        "description": (

        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {
                    "type": "string",
                    "description": "The exact product ID to look up.",
                }
            },
            "required": ["product_id"],
        },
    },
]


# --- TOOL IMPLEMENTATION ---

def tool_get_product_price(
    conn: sqlite3.Connection,
    product_id: str
) -> dict:
    """
    
    """
    row = lookup_product_by_id(conn, product_id)
    if row is None:
        result = PriceLookupResult(
            found=False,
            product_id=product_id,
            error=f"Product '{product_id}' not found in catalog."
        )
    else:
        price = round(float(row["price"]), 2)
        result = PriceLookupResult(
            found=True,
            product_id=product_id,
            product_name=row["name"],
            price=price,
            formatted_price=f"${price:.2f}",
            stock=int(row["stock"])
        )
    return result.model_dump()


def tool_get_product_detail(
    conn: sqlite3.Connection,
    product_id: str
) -> dict:
    """
    
    """
    row = lookup_product_by_id(conn, product_id)
    if row is None:
        result = ProductDetailResult(
            found=False,
            error=f"Product '{product_id}' not found."
        )
    else:
        result = ProductDetailResult(
            found=True,
            product=Product(
                id=row["id"],
                name=row["name"],
                price=round(float(row["price"]), 2),
                description=row["description"],
                product_type=row["product_type"],
                stock=int(row["stock"]),
            )
        )
    return result.model_dump()


# --- TOOL DISPATCHER ---

TOOL_IMPLEMENTATION = {
    "get_product_price": tool_get_product_price,
    "get_product_detail": tool_get_product_detail
}

def execute_tool(
    conn: sqlite3.Connection,
    tool_name: str,
    tool_input: dict
) -> str:
    """
    
    """
    fn = TOOL_IMPLEMENTATION.get(tool_name)
    if fn is None:
        return json.dumps({
            "error": f"Unknown tool: {tool_name}"
        })
    try:
        result = fn(conn, **tool_input)
        return json.dumps(result, default=str)
    except TypeError as e:
        return json.dumps({
            "error": f"Invalid arguments for {tool_name}"
        })
