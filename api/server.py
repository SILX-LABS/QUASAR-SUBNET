"""Quasar Subnet 24 API. App creation, middleware, startup."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from config import ALLOWED_ORIGINS, API_DESCRIPTION
from helpers.rate_limit import _rate_limiter
from helpers.cache import _bg_refresh
from helpers.fetch import _fetch_metagraph, _fetch_commitments, _fetch_price, _fetch_weights

# Import routers
from routes.health import router as health_router
from routes.miners import router as miners_router
from routes.evaluation import router as evaluation_router
from routes.market import router as market_router
from routes.chat import router as chat_router
from routes.debugging import router as debugging_router
from routes.telemetry import router as telemetry_router
from routes.dashboard import router as dashboard_router
from routes.state_ingest import router as state_ingest_router


app = FastAPI(
    title="Quasar - Subnet 24 API",
    description=API_DESCRIPTION,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {"name": "Overview", "description": "API info and health checks"},
        {"name": "Metagraph", "description": "On-chain subnet data - UIDs, stakes, weights, incentive"},
        {"name": "Miners", "description": "Miner model commitments and scores"},
        {"name": "Evaluation", "description": "Live eval progress, composite rounds, and score history"},
        {"name": "Market", "description": "Token pricing, emission, and market data"},
        {"name": "Chat", "description": "Chat with the current king model (when GPU is available)"},
        {"name": "Telemetry", "description": "Dashboard telemetry — composite axes, DQs, validator events, pod health"},
        {"name": "Dashboard", "description": "Compact UI payloads with chain-weight consensus, eval state, submissions, and validators"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Rate limiting middleware for all endpoints ────────────────────────────────

class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        # Skip rate limiting for docs
        if request.url.path in ("/docs", "/redoc", "/openapi.json"):
            return await call_next(request)
        # Chat/OpenAI endpoints have their own stricter limiter applied in the handler
        if request.url.path in ("/api/chat", "/v1/chat/completions", "/v1/models"):
            return await call_next(request)
        client_ip = request.client.host if request.client else "unknown"
        # Exempt localhost - dashboard SSR makes many internal requests
        if client_ip in ("127.0.0.1", "::1", "localhost"):
            return await call_next(request)
        if not _rate_limiter.is_allowed(client_ip):
            return JSONResponse(status_code=429, content={"error": "rate limit exceeded"})
        return await call_next(request)

app.add_middleware(RateLimitMiddleware)


# ── Include routers ──────────────────────────────────────────────────────────

app.include_router(health_router)
app.include_router(miners_router)
app.include_router(evaluation_router)
app.include_router(market_router)
app.include_router(chat_router)
app.include_router(debugging_router)
app.include_router(telemetry_router)
app.include_router(dashboard_router)
app.include_router(state_ingest_router)

# ── Startup: prime caches ────────────────────────────────────────────────────

@app.on_event("startup")
def prime_caches():
    """On startup, kick off background refreshes so first request is fast."""
    _bg_refresh("metagraph", _fetch_metagraph)
    _bg_refresh("commitments", _fetch_commitments)
    _bg_refresh("weights", _fetch_weights)
    _bg_refresh("price", _fetch_price)
