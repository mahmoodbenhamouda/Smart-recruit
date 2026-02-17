from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import sys

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from smart_resume_suite.services import cv_parser
from smart_resume_suite.services.cv_parser import ParsedResume
from smart_resume_suite.services.feedback_service import FeedbackEngine


class ResumeParserService:
    """Thin wrapper around the Streamlit-oriented parser for FastAPI use."""

    def __init__(self) -> None:
        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def parse(self, file_bytes: bytes, filename: str) -> ParsedResume:
        buffer = io.BytesIO(file_bytes)
        buffer.seek(0)
        return cv_parser.parse_resume(buffer, filename)


class FeedbackService:
    """Lazily initialises the dataset-backed feedback engine."""

    def __init__(self) -> None:
        self._engine: Optional[FeedbackEngine] = None
        self._startup_error: Optional[str] = None
        self._initialise_engine()

    @property
    def is_ready(self) -> bool:
        return self._engine is not None

    @property
    def startup_error(self) -> Optional[str]:
        return self._startup_error

    def _initialise_engine(self) -> None:
        try:
            self._engine = FeedbackEngine()
        except Exception as exc:  # pragma: no cover - depends on local dataset/API keys
            self._startup_error = str(exc)
            self._engine = None

    def generate_feedback(
        self, resume_text: str, job_description: Optional[str], user_question: Optional[str]
    ) -> dict:
        if not self._engine:
            raise RuntimeError(
                self._startup_error
                or "Feedback engine failed to initialise; check dataset and API keys."
            )
        return self._engine.generate_feedback(
            cv_text=resume_text,
            job_description=job_description or "",
            user_question=user_question or "",
        )


