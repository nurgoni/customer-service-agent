import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.bootstrap import build_app_context, close_app_context, setup_logging

logger = logging.getLogger("cs_agent.startup")


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    ctx = build_app_context()
    app.state.ctx = ctx
    app.state.agent = ctx.agent
    app.state.db_conn = ctx.conn
    logger.info("ready.")
    yield
    logger.info("shutdown: flush trace & menutup database...")
    close_app_context(ctx)


app = FastAPI(
    title="Customer Service Agent API",
    description="Three-lane chatbot: general, product search, exact fact lookup",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")