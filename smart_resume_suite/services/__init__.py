"""
Service layer for the Smart Resume Suite.

This package exposes cohesive utilities that combine functionality from the
legacy projects (CV parser, ATS analyzer, and SmartRecruiter feedback agent)
into reusable components for the new multi-page Streamlit application.
"""

from . import ats_service, cv_parser, feedback_service, path_utils  # noqa: F401

__all__ = [
    "ats_service",
    "cv_parser",
    "feedback_service",
    "path_utils",
]

