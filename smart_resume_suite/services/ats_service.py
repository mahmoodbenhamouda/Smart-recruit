from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from .. import config

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
ATS_AGENT_SRC = WORKSPACE_ROOT / "ATS-agent" / "ATS-agent"

if str(ATS_AGENT_SRC) not in sys.path:
    sys.path.insert(0, str(ATS_AGENT_SRC))

from ats_pipeline import ATSPipeline  # type: ignore  # noqa: E402
from pdf_extractor import PDFExtractor  # type: ignore  # noqa: E402
from rag_skills_extractor import RAGSkillsExtractor  # type: ignore  # noqa: E402


@lru_cache(maxsize=1)
def get_pipeline() -> ATSPipeline:
    return ATSPipeline(use_spacy=True)


@lru_cache(maxsize=1)
def get_rag_extractor() -> Optional[RAGSkillsExtractor]:
    skills_csv = ATS_AGENT_SRC / "data" / "skills_exploded (2).csv"
    if not skills_csv.exists():
        return None
    return RAGSkillsExtractor(skills_csv_path=str(skills_csv), max_skills=10000)


def run_ats_analysis(resume_pdf_path: Path, job_description: str) -> Dict[str, Any]:
    pipeline = get_pipeline()
    results = pipeline.analyze(
        str(resume_pdf_path),
        job_description,
        verbose=False,
        analyze_format=True,
    )

    rag_skills: list[str] = []
    rag_extractor = get_rag_extractor()
    if rag_extractor and resume_pdf_path.exists():
        try:
            resume_text = PDFExtractor().extract_text_safe(str(resume_pdf_path)) or ""
            if resume_text:
                rag_skills = rag_extractor.extract_skills_rag(resume_text, threshold=0.65)
        except Exception:  # pragma: no cover
            rag_skills = []

    results["rag_skills"] = rag_skills
    return results

