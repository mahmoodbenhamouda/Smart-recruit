from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ATS_SERVICE_ROOT = WORKSPACE_ROOT / "ats_service"
RESUME_SERVICE_ROOT = WORKSPACE_ROOT / "resume_service"

SEARCH_PATHS = (
    WORKSPACE_ROOT,
    ATS_SERVICE_ROOT,
    ATS_SERVICE_ROOT / "app",
    RESUME_SERVICE_ROOT,
    RESUME_SERVICE_ROOT / "app",
)

for path in SEARCH_PATHS:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import importlib

# Ensure the ATS package-relative imports (e.g. `from app.models`) resolve.
ats_package = importlib.import_module("ats_service.app")
sys.modules["app"] = ats_package

ats_module = importlib.import_module("ats_service.app.main")
resume_module = importlib.import_module("resume_service.app.main")

def get_ats_app():
    return ats_module.get_fastapi_app()


def get_resume_app():
    return resume_module.get_fastapi_app()

ats_app = get_ats_app()
resume_app = get_resume_app()

app = FastAPI(
    title="Talent Bridge AI Gateway",
    description="Unified FastAPI gateway exposing ATS analysis and resume intelligence endpoints.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ats_app.router, prefix="/ats", tags=["ATS Service"])
app.include_router(resume_app.router, prefix="/resume", tags=["Resume Intelligence"])


@app.get("/health", tags=["Gateway"])
async def gateway_health() -> Dict[str, Any]:
    ats_status = ats_module.health()
    resume_status = await resume_module.health()
    return {
        "gateway": "ok",
        "ats": ats_status.dict(),
        "resume": resume_status.dict(),
    }


@app.get("/", tags=["Gateway"])
def root_summary() -> Dict[str, str]:
    return {
        "message": "Talent Bridge AI Gateway is running.",
        "docs_url": "/docs",
        "ats_base": "/ats",
        "resume_base": "/resume",
    }


def get_fastapi_app() -> FastAPI:
    return app


