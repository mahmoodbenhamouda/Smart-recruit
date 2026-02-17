from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

from .models import (
    ATSExplanation,
    AnalysisResponse,
    JobPrediction,
    JobPredictionKeyword,
    SimilarityContribution,
    SkillExplanation,
    TopPrediction,
)
from .settings import get_settings


class AtsService:
    """
    Wrapper around the legacy ATS pipeline that keeps heavy models in memory
    and exposes an async-friendly API for FastAPI.
    """

    def __init__(self) -> None:
        settings = get_settings()
        ats_dir = settings.ats_agent_path

        if str(ats_dir) not in sys.path:
            sys.path.insert(0, str(ats_dir))

        try:
            from ats_pipeline import ATSPipeline  # type: ignore
            from job_role_predictor import JobRolePredictor  # type: ignore
        except ImportError as exc:  # pragma: no cover - startup failure
            raise RuntimeError(f"Failed to import ATS modules: {exc}") from exc

        self.pipeline = ATSPipeline(use_spacy=True)

        model_path = settings.job_prediction_model_path or ats_dir / "JobPrediction_Model"
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
        with redirect_stdout(StringIO()):
            self.job_predictor = JobRolePredictor(model_path=str(model_path))

    async def analyze(self, resume_bytes: bytes, job_description: str) -> AnalysisResponse:
        """
        Execute the ATS pipeline on provided resume bytes and job description text.
        Runs the heavy work in a thread executor to avoid blocking the event loop.
        """
        return await asyncio.to_thread(self._analyze_sync, resume_bytes, job_description)

    def _analyze_sync(self, resume_bytes: bytes, job_description: str) -> AnalysisResponse:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_resume:
            tmp_resume.write(resume_bytes)
            tmp_resume_path = Path(tmp_resume.name)

        try:
            results = self.pipeline.analyze(
                str(tmp_resume_path),
                job_description,
                verbose=False,
                analyze_format=False,
            )
        finally:
            tmp_resume_path.unlink(missing_ok=True)

        resume_keywords = results.get("resume_analysis", {}).get("keywords", {})
        keyword_list = (
            resume_keywords.get("technical_skills")
            or resume_keywords.get("all_keywords")
            or []
        )
        skills_text = " ".join(keyword_list[:120])

        job_prediction = None
        if skills_text:
            with redirect_stdout(StringIO()):
                prediction = self.job_predictor.predict_job_role(skills_text, top_n=5)
            job_prediction = JobPrediction(
                predicted_role=prediction.get("predicted_role"),
                confidence=prediction.get("confidence"),
                top_predictions=[
                    TopPrediction(role=role, probability=prob)
                    for role, prob in prediction.get("top_predictions", [])
                ],
            )

        similarity_scores: Dict[str, Any] = results.get("similarity_scores", {}) or {}
        detailed_scores: Dict[str, Any] = similarity_scores.get("detailed_scores", {}) or {}
        matched_skills: List[str] = similarity_scores.get("matched_skills") or []
        missing_skills: List[str] = similarity_scores.get("missing_skills") or []

        job_prediction_keywords = self._extract_job_prediction_keywords(skills_text, matched_skills)
        explanations = self._build_explanations(
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            detailed_scores=detailed_scores,
            job_prediction_keywords=job_prediction_keywords,
            job_prediction=job_prediction,
        )

        response = AnalysisResponse(
            success=bool(results.get("success", True)),
            overall_match=similarity_scores.get("overall_percentage"),
            match_level=similarity_scores.get("match_level"),
            skills_match_rate=detailed_scores.get("skills_match_rate"),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            job_prediction=job_prediction,
            explanations=explanations,
            raw=results,
        )
        return response

    def _build_explanations(
        self,
        matched_skills: List[str],
        missing_skills: List[str],
        detailed_scores: Dict[str, Any],
        job_prediction_keywords: List[JobPredictionKeyword],
        job_prediction: JobPrediction | None,
    ) -> ATSExplanation:
        skill_explanations: list[SkillExplanation] = []

        if matched_skills or missing_skills:
            all_skills = [(skill, "matched") for skill in matched_skills] + [
                (skill, "missing") for skill in missing_skills
            ]
            total = max(len(all_skills), 1)
            for skill, status in all_skills:
                weight = 1.0 / total if status == "matched" else -1.0 / total
                rationale = (
                    f"Skill '{skill}' was present in the resume and aligns with the job description."
                    if status == "matched"
                    else f"Skill '{skill}' appears important in the job description but was not detected in the resume."
                )
                skill_explanations.append(
                    SkillExplanation(
                        skill=skill,
                        status=status,
                        weight=round(weight, 4),
                        rationale=rationale,
                    )
                )

        metric_descriptions = {
            "skills_match_rate": "Overlap between resume skills and the job description skills.",
            "experience_alignment": "Alignment between stated experience and the target role.",
            "education_alignment": "Alignment of education or certifications with job requirements.",
            "format_score": "ATS-friendly formatting and structure of the resume.",
        }

        similarity_contributions = [
            SimilarityContribution(
                metric=metric,
                score=float(score),
                description=metric_descriptions.get(metric),
            )
            for metric, score in detailed_scores.items()
            if isinstance(score, (int, float))
        ]

        summary_parts: List[str] = []
        if similarity_contributions:
            top_metric = max(similarity_contributions, key=lambda item: item.score)
            summary_parts.append(
                f"Top contributing metric: {top_metric.metric} scored {top_metric.score:.2f}."
            )
        if missing_skills:
            summary_parts.append(
                f"{len(missing_skills)} important skills are missing; address them to raise the match score."
            )
        if job_prediction and job_prediction.predicted_role:
            confidence_display = job_prediction.confidence or 0.0
            summary_parts.append(
                f"Resume is closest to the '{job_prediction.predicted_role}' profile (confidence {confidence_display:.2f})."
            )

        summary = " ".join(summary_parts) if summary_parts else None

        return ATSExplanation(
            skill_explanations=skill_explanations,
            similarity_contributions=similarity_contributions,
            job_prediction_keywords=job_prediction_keywords,
            summary=summary,
        )

    def _extract_job_prediction_keywords(
        self, skills_text: str, matched_skills: List[str]
    ) -> List[JobPredictionKeyword]:
        if not skills_text:
            return []

        tokens = re.findall(r"[a-zA-Z]{3,}", skills_text.lower())
        counter = Counter(tokens)
        total = sum(counter.values()) or 1
        top_items = counter.most_common(6)
        keywords: list[JobPredictionKeyword] = []
        for word, count in top_items:
            weight = count / total
            keywords.append(
                JobPredictionKeyword(
                    keyword=word,
                    weight=round(weight, 4),
                )
            )

        # Ensure matched skills are surfaced even if not in top tokens.
        existing_keywords = {entry.keyword for entry in keywords}
        missing_from_keywords = [
            skill for skill in matched_skills[:3] if skill.lower() not in existing_keywords
        ]
        for skill in missing_from_keywords:
            keywords.append(
                JobPredictionKeyword(
                    keyword=skill.lower(),
                    weight=round(0.05, 4),
                )
            )

        return keywords


def serialize_response(response: AnalysisResponse) -> Dict[str, Any]:
    json_str = response.json(by_alias=True, exclude_none=False)
    return json.loads(json_str)

