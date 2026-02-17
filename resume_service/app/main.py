from __future__ import annotations

import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ATS_SERVICE_ROOT = WORKSPACE_ROOT / "ats_service"

for path in (WORKSPACE_ROOT, ATS_SERVICE_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import ats_service.app.models as ats_models  # type: ignore
import ats_service.app.pipeline as ats_pipeline  # type: ignore

from .models import (
    FeedbackRequest,
    FeedbackResponse,
    FeedbackTrace,
    HealthResponse as ResumeHealthResponse,
    OCRDiagnostics,
    ParseResumeResponse,
    RetrievedJobModel,
    SectionEvidence,
    SectionModel,
    SkillGapInsight,
)
from .service import FeedbackService, ResumeParserService

APP_TITLE = "Resume Intelligence Service"
APP_DESCRIPTION = (
    "FastAPI facade for resume OCR, LayoutLM section detection, personalised feedback, "
    "and ATS resume/job analysis."
)
APP_VERSION = "1.1.0"

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

parser_service = ResumeParserService()
feedback_service = FeedbackService()
ats_service = ats_pipeline.AtsService()
ats_router = APIRouter(prefix="/ats", tags=["ATS Analysis"])

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "into",
    "your",
    "have",
    "will",
    "about",
    "using",
    "such",
    "through",
    "their",
    "each",
    "over",
    "been",
    "also",
    "within",
}

SECTION_HIGHLIGHT_KEYWORDS = {
    "experience": {"managed", "developed", "led", "project", "team", "delivered"},
    "education": {"bachelor", "master", "degree", "university", "certification"},
    "skills": {"skills", "expertise", "languages", "tools", "technologies"},
    "summary": {"summary", "profile", "overview", "professional"},
    "projects": {"project", "implemented", "built", "designed"},
    "certifications": {"certified", "certification", "license"},
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def _extract_top_keywords(tokens: List[str], limit: int = 6) -> List[str]:
    meaningful_tokens = [token for token in tokens if token not in _STOPWORDS]
    if not meaningful_tokens:
        return []
    counter = Counter(meaningful_tokens)
    return [word for word, _ in counter.most_common(limit)]


def _derive_section_evidence(section_name: str, content: str) -> tuple[float, SectionEvidence]:
    tokens = _tokenize(content)
    top_keywords = _extract_top_keywords(tokens)
    keyword_hits = 0
    for key, keywords in SECTION_HIGHLIGHT_KEYWORDS.items():
        if key in section_name.lower():
            keyword_hits = sum(1 for kw in top_keywords if kw in keywords)
            break

    bullet_signals = len(re.findall(r"(\n[-•\u2022])", content))
    token_score = min(len(tokens) / 400.0, 1.0)
    keyword_score = min(keyword_hits * 0.12, 0.36)
    bullet_score = 0.08 if bullet_signals else 0.0
    confidence = max(0.35, min(0.98, 0.35 + token_score * 0.35 + keyword_score + bullet_score))

    rationale_parts = []
    if keyword_hits:
        rationale_parts.append(f"Detected {keyword_hits} domain keywords for {section_name.lower()}.")
    if bullet_signals:
        rationale_parts.append("Bullet-style formatting reinforces the section structure.")
    if len(tokens) > 80:
        rationale_parts.append("Sufficient content length for reliable classification.")
    if not rationale_parts:
        rationale_parts.append("Section detected via LayoutLM clustering and lexical cues.")

    snippet = content.strip().split("\n", 1)[0][:220]

    evidence = SectionEvidence(
        rationale=" ".join(rationale_parts),
        keywords=top_keywords[:5],
        snippet=snippet,
    )
    return confidence, evidence


def _build_ocr_diagnostics(raw_text: str, ocr_used: bool, errors: List[str]) -> OCRDiagnostics:
    warnings = [
        err for err in errors if "ocr" in err.lower() or "text extraction" in err.lower()
    ]
    method = "ocr_fallback" if ocr_used else "direct_pdf"
    return OCRDiagnostics(
        method=method,
        characters_extracted=len(raw_text),
        warnings=warnings,
    )


def _extract_skill_gaps(resume_text: str, job_description: str, limit: int = 8) -> List[SkillGapInsight]:
    if not job_description:
        return []

    resume_tokens = set(_tokenize(resume_text))
    job_tokens = _tokenize(job_description)
    counter = Counter(
        token for token in job_tokens if token not in _STOPWORDS and len(token) > 2
    )

    gap_candidates = [
        (token, freq)
        for token, freq in counter.items()
        if token not in resume_tokens
    ]
    if not gap_candidates:
        return []

    total = sum(freq for _, freq in gap_candidates) or 1
    insights: List[SkillGapInsight] = []
    for skill, freq in sorted(gap_candidates, key=lambda item: item[1], reverse=True)[:limit]:
        importance = min(1.0, freq / total * 1.5)
        rationale = f"Skill '{skill}' appears {freq} times in the job description but was not detected in the resume."
        insights.append(
            SkillGapInsight(
                skill=skill,
                importance=round(importance, 4),
                rationale=rationale,
                evidence=[f"Frequency in job description: {freq}"],
            )
        )

    return insights


def _build_feedback_breakdown(
    intermediate_sections: Dict[str, str],
    skill_insights: List[SkillGapInsight],
    retrieved_jobs: List[RetrievedJobModel],
) -> List[FeedbackTrace]:
    traces: List[FeedbackTrace] = []
    skill_lookup = [insight.skill for insight in skill_insights[:5]]
    job_indices = [job.index for job in retrieved_jobs[:3]]

    for key, text in intermediate_sections.items():
        cleaned_lines = [
            line.strip(" -•\u2022")
            for line in text.splitlines()
            if line.strip()
        ]
        if not cleaned_lines:
            continue
        summary = cleaned_lines[0]
        bullets = cleaned_lines[1:] or cleaned_lines[:1]
        traces.append(
            FeedbackTrace(
                section=key,
                summary=summary,
                bullets=bullets,
                supporting_skills=skill_lookup,
                supporting_jobs=job_indices,
            )
        )
    return traces


@app.get("/health")
async def health() -> Dict[str, Any]:
    parser_ready = parser_service.is_ready
    feedback_ready = feedback_service.is_ready
    if parser_ready and feedback_ready:
        overall = "ok"
    elif parser_ready:
        overall = "degraded"
    else:
        overall = "error"

    resume_health = ResumeHealthResponse(
        status=overall,
        timestamp=int(time.time()),
        parser_ready=parser_ready,
        feedback_ready=feedback_ready,
        feedback_error=feedback_service.startup_error,
    )

    ats_health = {"status": "ok", "timestamp": int(time.time())}

    return {
        "timestamp": int(time.time()),
        "resume_service": resume_health.dict(),
        "ats_service": ats_health.dict(),
    }


@app.post(
    "/parse-resume",
    response_model=ParseResumeResponse,
    summary="Parse a resume PDF and return structured sections and skills.",
)
async def parse_resume(resume: UploadFile = File(...)) -> ParseResumeResponse:
    if resume.content_type not in (
        "application/pdf",
        "application/octet-stream",
    ):
        raise HTTPException(status_code=400, detail="Resume must be uploaded as a PDF.")

    resume_bytes = await resume.read()
    if not resume_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume is empty.")

    try:
        parsed = await run_in_threadpool(parser_service.parse, resume_bytes, resume.filename)
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    sections_payload: List[SectionModel] = []
    for section in parsed.sections:
        if not isinstance(section, dict):
            continue
        name = str(section.get("section_name", "Section")).strip() or "Section"
        content = str(section.get("content", "")).strip()
        confidence, evidence = _derive_section_evidence(name, content)
        sections_payload.append(
            SectionModel(
                section_name=name,
                content=content,
                confidence=round(confidence, 4),
                evidence=evidence,
            )
        )

    ocr_diagnostics = _build_ocr_diagnostics(parsed.raw_text, parsed.ocr_used, parsed.errors)

    return ParseResumeResponse(
        filename=resume.filename,
        source_path=str(parsed.source_path) if parsed.source_path else None,
        raw_text=parsed.raw_text,
        sections=sections_payload,
        skills=parsed.skill_entities,
        ocr_used=parsed.ocr_used,
        ocr_diagnostics=ocr_diagnostics,
        errors=parsed.errors,
    )


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Generate personalised feedback using resume text and optional job context.",
)
async def generate_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    if not payload.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text is required.")

    try:
        result: Dict[str, Any] = await run_in_threadpool(
            feedback_service.generate_feedback,
            payload.resume_text,
            payload.job_description,
            payload.user_question,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    retrieved_jobs_payload = []
    for job in result.get("retrieved_jobs", []):
        if isinstance(job, dict):
            try:
                retrieved_jobs_payload.append(RetrievedJobModel(**job))
            except (TypeError, ValueError):
                continue

    skill_gap_payload: List[SkillGapInsight] = []
    for insight in result.get("skill_gap_insights", []):
        if isinstance(insight, dict):
            try:
                skill_gap_payload.append(SkillGapInsight(**insight))
            except (TypeError, ValueError):
                continue

    if not skill_gap_payload and payload.job_description:
        skill_gap_payload = _extract_skill_gaps(
            payload.resume_text, payload.job_description
        )

    feedback_breakdown_payload: List[FeedbackTrace] = []
    for trace in result.get("feedback_breakdown", []):
        if isinstance(trace, dict):
            try:
                feedback_breakdown_payload.append(FeedbackTrace(**trace))
            except (TypeError, ValueError):
                continue

    intermediate_sections = {
        str(key): str(value)
        for key, value in (result.get("intermediate_sections") or {}).items()
        if isinstance(key, str) and isinstance(value, str)
    }
    if not feedback_breakdown_payload and intermediate_sections:
        feedback_breakdown_payload = _build_feedback_breakdown(
            intermediate_sections=intermediate_sections,
            skill_insights=skill_gap_payload,
            retrieved_jobs=retrieved_jobs_payload,
        )

    return FeedbackResponse(
        final_answer=str(result.get("final_answer", "")),
        retrieved_jobs=retrieved_jobs_payload,
        skill_gap_insights=skill_gap_payload,
        feedback_breakdown=feedback_breakdown_payload,
        intermediate_sections=intermediate_sections,
        error=result.get("error"),
    )


@ats_router.get("/health", response_model=ats_models.HealthResponse)
def ats_health() -> ats_models.HealthResponse:
    return ats_models.HealthResponse(status="ok", timestamp=int(time.time()))


@ats_router.post(
    "/analyze",
    response_model=ats_models.AnalysisResponse,
    summary="Run ATS analysis on a resume and job description.",
)
async def analyze_resume(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text"),
) -> ats_models.AnalysisResponse:
    if resume.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file.")

    resume_bytes = await resume.read()
    if not resume_bytes:
        raise HTTPException(status_code=400, detail="Uploaded resume is empty.")

    try:
        response = await ats_service.analyze(resume_bytes, job_description)
    except Exception as exc:  # pragma: no cover - runtime env dependent
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return response


app.include_router(ats_router)


def get_fastapi_app() -> FastAPI:
    """Helper to provide the ASGI application to uvicorn."""
    return app


