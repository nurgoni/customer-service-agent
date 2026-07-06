import os
from contextlib import asynccontextmanager
import anthropic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from src.db.database import get_connection, init_db
from src.retrieval.embedder import get_embedder
from src.retrieval.hybrid_search import build_bm25_index
from src.agent.core import CustomerServiceAgent
from src.api.routes import router


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""

    # --- Startup ---

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("")
    
    print("[ startup ] connecting to database...")
    conn = get_connection()
    init_db(conn)
    app.state.db_conn = conn

    print("[ startup ] loading embedding model...")
    get_embedder()

    print("[ startup ] building BM25 index...")
    build_bm25_index(conn)

    print("[ startup ] initialising agent...")
    client = anthropic.Anthropic(api_key=api_key)
    app.state.agent = CustomerServiceAgent(client=client, conn=conn)

    print("[ startup ] ready.")
    yield

    # --- Shutdown ---
    print("[ shutdown ] closing database connection...")
    conn.close()

app = FastAPI(
    title="Customer Service Agent API",
    description="Three-lane chatbot: general, product search, exact fact lookup",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins==["*"],
    allow_method=["*"],
    allow_headers=["*"]
)

app.include_router(router, prefix="/api/v1")
