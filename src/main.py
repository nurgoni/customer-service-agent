from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from src.agent.core import CustomerServiceAgent
from src.api.routes import router
from src.config import (
    EMBEDDING_MODEL,
    LLM_REASONING_EFFORT,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
    OPENAI_MODEL_FAST,
)
from src.db.database import get_connection, has_vector_index, init_db
from src.retrieval.hybrid_search import build_bm25_index, set_vector_enabled


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY belum diset. Isi file .env (lihat .env.example). Untuk Ollama isi nilai bebas, mis. 'ollama'."
        )

    print(
        f"[ startup ] LLM base_url={OPENAI_BASE_URL or 'OpenAI'} model={OPENAI_MODEL} "
        f"intent_model={OPENAI_MODEL_FAST} reasoning_effort={LLM_REASONING_EFFORT or '-'}",
        flush=True,
    )

    print("[ startup ] connecting to database...", flush=True)
    conn = get_connection()
    init_db(conn)
    app.state.db_conn = conn

    print("[ startup ] checking embedding model...", flush=True)
    vector_ok = False
    if has_vector_index(conn):
        try:
            from src.retrieval.embedder import embed

            dim = len(embed("ping"))
            vector_ok = True
            print(f"[ startup ] embedding '{EMBEDDING_MODEL}' siap (dimensi {dim})", flush=True)
        except Exception as e:
            print(f"[ startup ] embedding tidak bisa dipanggil ({e}); pakai BM25 saja", flush=True)
    else:
        print("[ startup ] index vektor belum ada (jalankan seed) atau sqlite-vec tidak aktif; pakai BM25 saja", flush=True)
    set_vector_enabled(vector_ok)

    print("[ startup ] building BM25 index...", flush=True)
    build_bm25_index(conn)

    print("[ startup ] initialising agent...", flush=True)
    app.state.agent = CustomerServiceAgent(
        client=OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL),
        conn=conn,
    )

    print("[ startup ] ready.", flush=True)
    yield

    print("[ shutdown ] closing database connection...", flush=True)
    conn.close()


app = FastAPI(
    title="Customer Service Agent API",
    description="Three-lane chatbot: general, product search, exact fact lookup",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
