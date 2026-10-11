"""
Inisialisasi bersama untuk server API (src/main.py) dan chat CLI (src/cli.py):
logging, database, embedding, index BM25, client LLM, dan Langfuse.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from dataclasses import dataclass

from src import observability
from src.agent.core import CustomerServiceAgent
from src.config import (
    EMBEDDING_MODEL,
    LANGFUSE_BASE_URL,
    LLM_REASONING_EFFORT,
    LOG_LEVEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_MODEL_FAST,
)
from src.db.database import get_connection, has_vector_index, init_db
from src.retrieval.hybrid_search import build_bm25_index, set_vector_enabled

logger = logging.getLogger("cs_agent.startup")


@dataclass
class AppContext:
    agent: CustomerServiceAgent
    conn: sqlite3.Connection
    vector_enabled: bool
    tracing_enabled: bool
    tracing_connected: bool  # True bila server Langfuse terhubung dan key valid saat startup


def setup_logging(level: str | None = None) -> None:
    """Log aplikasi memakai logger 'cs_agent' (tidak mengubah logger uvicorn/httpx)."""
    app_logger = logging.getLogger("cs_agent")
    app_logger.setLevel((level or LOG_LEVEL).upper())
    if not app_logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(name)s] %(message)s", "%H:%M:%S"))
        app_logger.addHandler(handler)
    app_logger.propagate = False


def build_app_context() -> AppContext:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY belum diset. Isi file .env (lihat .env.example). Untuk Ollama isi nilai bebas, mis. 'ollama'."
        )

    logger.info(
        "LLM base_url=%s model=%s intent_model=%s reasoning_effort=%s",
        OPENAI_BASE_URL or "OpenAI",
        OPENAI_MODEL,
        OPENAI_MODEL_FAST,
        LLM_REASONING_EFFORT or "-",
    )

    conn = get_connection()
    init_db(conn)

    vector_ok = False
    if has_vector_index(conn):
        try:
            from src.retrieval.embedder import embed

            dim = len(embed("ping"))
            vector_ok = True
            logger.info("embedding '%s' siap (dimensi %d)", EMBEDDING_MODEL, dim)
        except Exception as e:
            logger.warning("embedding tidak bisa dipanggil (%s); pakai BM25 saja", e)
    else:
        logger.warning("index vektor belum ada (jalankan seed) atau sqlite-vec tidak aktif; pakai BM25 saja")
    set_vector_enabled(vector_ok)

    build_bm25_index(conn)

    tracing_connected = False
    if observability.is_enabled():
        tracing_connected = bool(observability.check_connection())
        if tracing_connected:
            logger.info("Langfuse tracing aktif -> %s", LANGFUSE_BASE_URL)
        else:
            logger.warning(
                "Langfuse tracing aktif tetapi server/key belum valid (%s); agent tetap berjalan", LANGFUSE_BASE_URL
            )
    else:
        logger.info("Langfuse tracing nonaktif (LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY kosong)")

    agent = CustomerServiceAgent(
        client=observability.create_openai_client(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
        conn=conn,
    )
    return AppContext(
        agent=agent,
        conn=conn,
        vector_enabled=vector_ok,
        tracing_enabled=observability.is_enabled(),
        tracing_connected=tracing_connected,
    )


def close_app_context(ctx: AppContext | None) -> None:
    observability.shutdown()  # kirim sisa trace sebelum keluar
    if ctx is not None:
        ctx.conn.close()