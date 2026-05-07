"""
WikiRisk – FastAPI application entry point.

Endpoints:
    GET  /health            Service health + external dependency status
    GET  /stats             Aggregate risk statistics
    GET  /edits/recent      Paginated list of recent edits (filterable)
    GET  /edits/{id}        Single edit record
    GET  /explain/{id}      AI-generated risk explanation (cached)
    POST /explain           AI explanation (body: ExplanationRequest)

Run locally:
    uvicorn src.serving.main:app --reload --port 8000
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.common.logger import configure_logging, get_logger
from src.config import get_settings
from src.serving.database import init_db
from src.serving.routes import edits, explain, health, notify

log = get_logger(__name__)
cfg = get_settings()
configure_logging(cfg.log_level, cfg.log_format)

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api_startup", version="1.0.0", db=cfg.db_path)
    await init_db(cfg.db_path)
    yield
    log.info("api_shutdown")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="WikiRisk API",
    description=(
        "Real-Time Wikipedia Edit Risk & Knowledge Integrity Monitoring. "
        "Provides risk-scored edit predictions from Spark Structured Streaming."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.api_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.monotonic()
    response = await call_next(request)
    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=elapsed_ms,
    )
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    return response


# ── Exception handlers ────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        type=type(exc).__name__,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": type(exc).__name__},
    )


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(edits.router)
app.include_router(explain.router)
app.include_router(notify.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "WikiRisk API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
