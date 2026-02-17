"""
Smart Resume Suite
==================

Unified package that orchestrates resume parsing, ATS-style analysis, and
career feedback generation.  The modules in this package wrap the existing
project code so it can be reused inside a single Streamlit multi-page
application.
"""

from importlib import metadata


def get_version() -> str:
    """Return the installed package version if available."""
    try:
        return metadata.version("smart_resume_suite")
    except metadata.PackageNotFoundError:
        return "0.0.0"

