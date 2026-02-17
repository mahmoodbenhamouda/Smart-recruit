from __future__ import annotations

import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from .. import config
from . import path_utils

# ---------------------------------------------------------------------------
# Import legacy modules with dynamic path injection
# ---------------------------------------------------------------------------

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

ATS_AGENT_SRC = WORKSPACE_ROOT / "ATS-agent" / "ATS-agent"
PI5_SRC = WORKSPACE_ROOT / "pi5eme1 - Copie (2)" / "pi5eme1 - Copie (2)"

for legacy_path in (str(ATS_AGENT_SRC), str(PI5_SRC)):
    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)

from pdf_extractor import PDFExtractor  # type: ignore  # noqa: E402
from models.layoutlm_detect_sections import detect_sections  # type: ignore  # noqa: E402
from models.nlp_extraction import extract_skills_with_spacy  # type: ignore  # noqa: E402
from ocr import extract_text as ocr_extract_text  # type: ignore  # noqa: E402

pdf_extractor = PDFExtractor()


@dataclass
class ParsedResume:
    """Structured representation of the resume after the parsing stage."""

    source_path: Path
    raw_text: str
    sections: List[Dict[str, str]]
    skill_entities: List[str]
    ocr_used: bool
    errors: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "source_path": str(self.source_path),
            "raw_text": self.raw_text,
            "sections": self.sections,
            "skill_entities": self.skill_entities,
            "ocr_used": self.ocr_used,
            "errors": self.errors,
        }


def _attempt_pdf_text_extraction(pdf_path: Path) -> str:
    text = pdf_extractor.extract_text_safe(str(pdf_path))
    return text or ""


def _attempt_ocr(pdf_path: Path) -> str:
    try:
        poppler_path = config.resolve_poppler_path()
        if not poppler_path:
            raise RuntimeError(
                "Poppler path is not configured. Set POPPLER_PATH env variable."
            )
        return ocr_extract_text.extract_text_from_pdf(
            str(pdf_path), poppler_path=str(poppler_path)
        )
    except Exception as exc:  # pragma: no cover - depends on environment
        st.warning(f"OCR fallback failed: {exc}")
        return ""


def parse_resume(uploaded_file: io.BytesIO, filename: str) -> ParsedResume:
    """
    Parse a resume PDF:
        1. Save it to the temp directory.
        2. Attempt text extraction via PyPDF2.
        3. Fallback to OCR if necessary.
        4. Run LayoutLMv3 section detection.
        5. Extract skills via spaCy model (if available).
    """
    file_bytes = uploaded_file.read()
    stored_path = path_utils.save_uploaded_file(filename, file_bytes, suffix=".pdf")

    errors: List[str] = []

    raw_text = _attempt_pdf_text_extraction(stored_path)
    ocr_used = False
    if len(raw_text) < 400:
        ocr_text = _attempt_ocr(stored_path)
        if ocr_text:
            raw_text = ocr_text
            ocr_used = True
        else:
            errors.append("PDF text extraction failed and OCR fallback returned empty.")

    raw_text = raw_text.strip()

    sections: List[Dict[str, str]] = []
    if raw_text:
        try:
            sections = detect_sections(raw_text) or []
        except Exception as exc:  # pragma: no cover
            errors.append(f"LayoutLM section detection failed: {exc}")
    else:
        errors.append("No text available to detect sections.")

    try:
        skill_entities = extract_skills_with_spacy(raw_text) if raw_text else []
    except Exception as exc:  # pragma: no cover
        errors.append(f"spaCy skill extraction failed: {exc}")
        skill_entities = []

    return ParsedResume(
        source_path=stored_path,
        raw_text=raw_text,
        sections=sections,
        skill_entities=skill_entities,
        ocr_used=ocr_used,
        errors=errors,
    )

