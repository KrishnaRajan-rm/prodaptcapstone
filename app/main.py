"""FastAPI Application Entry Point for AI-Powered Restaurant Operations Assistant.
Exposes REST endpoints on Port 8000 connecting the React frontend with the multi-agent system.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.core.logging import logger
from app.api.routes import router as api_router
from app.api.persistence import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initializes databases and resources on startup."""
    logger.info("Initializing Restaurant Operations Assistant API Gateway...")
    init_db()
    logger.info("SQLite persistence initialized.")
    yield
    logger.info("Shutting down API Gateway.")


app = FastAPI(
    title="AI-Powered Restaurant Operations Review Assistant API",
    description="Operational diagnostic, multi-agent synthesis, and Quality Gate API for restaurant operations.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for React/Vite development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://localhost:8501",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(api_router)


@app.get("/")
async def root():
    return {
        "app": "AI-Powered Restaurant Operations Review Assistant",
        "status": "online",
        "docs_url": "/docs",
        "api_v1": "/api/v1",
    }


def main():
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
