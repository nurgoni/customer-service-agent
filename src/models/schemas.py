from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


def format_rupiah(value: float) -> str:
    """Format kanonik harga: 2199000 -> 'Rp 2.199.000'. Dipakai di SEMUA tempat yang menampilkan harga."""
    return "Rp " + f"{value:,.0f}".replace(",", ".")


# --- INTENT ---

class Intent(str, Enum):
    GENERAL = "general"
    PRODUCT_SEARCH = "product_search"
    EXACT_FACT = "exact_fact"


class IntentResult(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0.0, le=1.0)
    extracted_query: str = Field(description="Cleaned query to pass downstream")
    product_id: str | None = Field(
        default=None,
        description="Extracted product ID for EXACT_FACT queries (if present)",
    )


# --- PRODUCT DOMAIN ---

class Product(BaseModel):
    """Mirrors the product database row. Price always comes from the DB, never from the LLM."""
    id: str
    name: str
    price: float = Field(gt=0.0, description="Price in IDR - must come from DB, never generated")
    description: str
    product_type: str
    stock: int = Field(ge=0)

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Price must be positive")
        return round(v, 2)

    def price_display(self) -> str:
        return format_rupiah(self.price)


class ProductSearchResult(BaseModel):
    product: Product
    score: float = Field(description="Hybrid RRF score (higher = more relevant)")
    match_reason: str = Field(description="Why this product was retrieved")


# --- CHAT API ---

class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str = Field(default="default")
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    reply: str
    intent: Intent
    products_shown: list[str] = Field(default_factory=list, description="Product IDs surfaced in this turn")
    sources: list[str] = Field(
        default_factory=list,
        description="How the answer was grounded: 'db_lookup', 'hybrid_search', 'llm_only'",
    )
    session_id: str


# --- TOOL CALL ---

class PriceLookupResult(BaseModel):
    found: bool
    product_id: str
    product_name: str | None = None
    price: float | None = None
    formatted_price: str | None = None
    stock: int | None = None
    error: str | None = None


class ProductDetailResult(BaseModel):
    found: bool
    product: Product | None = None
    formatted_price: str | None = None
    error: str | None = None


# --- INTERNAL ---

class AgentState(BaseModel):
    session_id: str
    user_message: str
    intent: Intent | None = None
    retrieved_products: list[ProductSearchResult] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    final_reply: str = ""
    sources: list[str] = Field(default_factory=list)
    shown_product_ids: list[str] = Field(default_factory=list)
