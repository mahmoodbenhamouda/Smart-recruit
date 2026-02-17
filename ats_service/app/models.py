from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    role: str = Field(..., description="Predicted role label")
    probability: float = Field(..., ge=0.0, le=1.0, description="Probability assigned to this role")


class JobPrediction(BaseModel):
    predicted_role: Optional[str] = Field(None, description="Most likely role label")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence for predicted role")
    top_predictions: List[TopPrediction] = Field(default_factory=list, description="Top-N job role probabilities")


class SkillExplanation(BaseModel):
    skill: str = Field(description="Skill or keyword evaluated by the ATS.")
    status: str = Field(description="matched or missing relative to the job description.")
    weight: float = Field(
        ge=-1.0,
        le=1.0,
        description="Relative contribution of the skill to the final similarity score.",
    )
    rationale: str = Field(description="Short textual justification for the contribution.")


class SimilarityContribution(BaseModel):
    metric: str = Field(description="Name of the similarity metric reported by the pipeline.")
    score: float = Field(description="Numeric score returned by the pipeline.")
    description: Optional[str] = Field(
        default=None, description="Human-readable interpretation of what the metric measures."
    )


class JobPredictionKeyword(BaseModel):
    keyword: str = Field(description="Keyword or phrase influential in the job prediction.")
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Relative importance of the keyword within the prediction context.",
    )


class ATSExplanation(BaseModel):
    skill_explanations: List[SkillExplanation] = Field(
        default_factory=list,
        description="Explainability artefacts for matched and missing skills.",
    )
    similarity_contributions: List[SimilarityContribution] = Field(
        default_factory=list,
        description="Breakdown of the similarity metrics contributing to the overall score.",
    )
    job_prediction_keywords: List[JobPredictionKeyword] = Field(
        default_factory=list,
        description="Top resume keywords that influenced job prediction.",
    )
    summary: Optional[str] = Field(
        default=None,
        description="High-level natural language summary of the ATS rationale.",
    )


class AnalysisResponse(BaseModel):
    success: bool = Field(True, description="Indicates whether the pipeline executed successfully")
    overall_match: Optional[float] = Field(None, description="Overall resume/job match percentage (0-100)")
    match_level: Optional[str] = Field(None, description="Textual interpretation of the overall match")
    skills_match_rate: Optional[float] = Field(
        None, description="Skills match rate (0-1) from ATS similarity scores"
    )
    matched_skills: List[str] = Field(default_factory=list, description="Skills detected in both resume and job")
    missing_skills: List[str] = Field(default_factory=list, description="Skills missing from the resume")
    job_prediction: Optional[JobPrediction] = Field(None, description="AI job prediction from skill extraction")
    explanations: Optional[ATSExplanation] = Field(
        default=None,
        description="Explainability payload describing how the ATS reached its conclusions.",
    )
    raw: Optional[dict] = Field(None, description="Complete raw payload returned by the ATS pipeline")


class HealthResponse(BaseModel):
    status: str
    timestamp: int

