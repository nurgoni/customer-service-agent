# src/main.py

import os
from contextlib import asynccontextmanager
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.config import OPENAI_API_KEY
from src.db.database import get_connection, init_db
from src.retrieval.embedder import get_embedder
from src.retrieval.hybrid_search import build_bm25_index
from src.agent.core import CustomerServiceAgent
from src.api.routes import router


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    # if not OPENAI_API_KEY:
    #     raise RuntimeError("OPENAI_API_KEY not set")

    print("[ startup ] connecting to database...")
    conn = get_connection()
    init_db(conn)
    app.state.db_conn = conn

    print("[ startup ] loading embedding model...")
    get_embedder()

    print("[ startup ] building BM25 index...")
    build_bm25_index(conn)

    print("[ startup ] initialising agent...")
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )
    app.state.agent = CustomerServiceAgent(client=client, conn=conn)

    print("[ startup ] ready ✅")
    yield

    # --- Shutdown ---
    conn.close()


app = FastAPI(
    title="Customer Service Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ← fixed: single =
    allow_methods=["*"],       # ← fixed: allow_methods (plural)
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")