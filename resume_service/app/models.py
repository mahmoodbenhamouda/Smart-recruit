from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(
        description="Overall status flag: ok, degraded, or error.",
    )
    timestamp: int = Field(
        description="Unix timestamp when the health snapshot was produced.",
    )
    parser_ready: bool = Field(
        description="True when the resume parser backend is initialised.",
    )
    feedback_ready: bool = Field(
        description="True when the personalised feedback engine is initialised.",
    )
    feedback_error: Optional[str] = Field(
        default=None,
        description="Startup error message for the feedback engine, if any.",
    )


class SectionEvidence(BaseModel):
    rationale: str = Field(
        description="Natural language summary describing why the section was classified."
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Top discriminative keywords that influenced the section label.",
    )
    snippet: str = Field(
        description="Excerpt from the resume illustrating the detected section."
    )


class SectionModel(BaseModel):
    section_name: str
    content: str
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the detected section."
    )
    evidence: SectionEvidence = Field(
        ..., description="Explainability artefacts for the section classification."
    )


class OCRDiagnostics(BaseModel):
    method: str = Field(
        description="Extraction strategy used: direct_pdf or ocr_fallback."
    )
    characters_extracted: int = Field(
        description="Total number of characters recovered from the resume."
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warnings or errors encountered during extraction.",
    )


class ParseResumeResponse(BaseModel):
    filename: str
    source_path: Optional[str] = Field(
        default=None, description="Server-side persisted path for the uploaded resume."
    )
    raw_text: str = Field(description="Full text extracted from the resume.")
    sections: List[SectionModel]
    skills: List[str] = Field(default_factory=list)
    ocr_used: bool = Field(
        default=False, description="True when OCR fallback was used to extract text."
    )
    ocr_diagnostics: Optional[OCRDiagnostics] = Field(
        default=None,
        description="Diagnostic information about the text extraction process.",
    )
    errors: List[str] = Field(
        default_factory=list,
        description="Non-fatal issues encountered while parsing the resume.",
    )


class FeedbackRequest(BaseModel):
    resume_text: str = Field(..., description="Plain text extracted from the resume.")
    job_description: Optional[str] = Field(
        default=None, description="Target job description used for context."
    )
    user_question: Optional[str] = Field(
        default=None, description="Optional question from the candidate."
    )


class RetrievedJobModel(BaseModel):
    index: int
    similarity: float
    job_title: str
    job_description: str
    reason_for_decision: str


class SkillGapInsight(BaseModel):
    skill: str
    importance: float = Field(
        ..., ge=0.0, le=1.0, description="Relative importance of the missing skill."
    )
    rationale: str = Field(description="Explanation of why the skill matters.")
    evidence: List[str] = Field(
        default_factory=list,
        description="Supportive snippets or dataset references highlighting the gap.",
    )


class FeedbackTrace(BaseModel):
    section: str = Field(
        description="Portion of the feedback message (feedback, coaching, alternative_roles)."
    )
    summary: str = Field(description="High-level summary of the section.")
    bullets: List[str] = Field(
        default_factory=list, description="Individual bullet points generated for this section."
    )
    supporting_skills: List[str] = Field(
        default_factory=list,
        description="Skill identifiers that informed this part of the feedback.",
    )
    supporting_jobs: List[int] = Field(
        default_factory=list,
        description="Indices of retrieved dataset rows that influenced the section.",
    )


class FeedbackResponse(BaseModel):
    final_answer: str
    retrieved_jobs: List[RetrievedJobModel] = Field(default_factory=list)
    skill_gap_insights: List[SkillGapInsight] = Field(
        default_factory=list, description="Structured view of the key skill gaps."
    )
    feedback_breakdown: List[FeedbackTrace] = Field(
        default_factory=list,
        description="Explainable mapping of generated feedback to influencing factors.",
    )
    intermediate_sections: Dict[str, str] = Field(
        default_factory=dict,
        description="Raw intermediate sections (feedback/coaching/alternatives) prior to synthesis.",
    )
    error: Optional[str] = None


