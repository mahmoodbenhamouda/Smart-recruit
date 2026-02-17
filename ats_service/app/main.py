from __future__ import annotations

import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .models import AnalysisResponse, HealthResponse
from .pipeline import AtsService, serialize_response
from .settings import get_settings

app = FastAPI(
    title="ATS Pipeline Service",
    description="FastAPI wrapper around the ATS resume analysis pipeline with Explainable AI",
    version="2.0.0"
)

service = AtsService()

# Include XAI routes
try:
    from .xai_routes import router as xai_router
    app.include_router(xai_router)
    print("✅ XAI routes loaded successfully")
except Exception as e:
    print(f"⚠️ XAI routes not available: {e}")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", timestamp=int(time.time()))


@app.post("/analyze", response_model=AnalysisResponse, summary="Run ATS analysis on a resume and job description")
async def analyze_resume(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text")
) -> AnalysisResponse:
    if resume.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file.")

    resume_bytes = await resume.read()
    if not resume_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume is empty.")

    try:
        response = await service.analyze(resume_bytes, job_description)
    except Exception as exc:  # pragma: no cover - runtime error
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return response


def get_fastapi_app() -> FastAPI:
    """Helper for uvicorn entrypoint."""
    return app

