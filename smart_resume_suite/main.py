from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_resume_suite import get_version

st.set_page_config(
    page_title="Smart Resume Suite",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Smart Resume Suite")
st.markdown(
    """
Welcome to the unified experience that combines the CV parser, ATS analyzer, and
SmartRecruiter feedback assistant into a single workflow.

Follow the steps below using the sidebar navigation:

1. **Resume Parsing** – upload a PDF and extract structured sections & skills.
2. **ATS Analysis** – compare the resume against a job description to get scores.
3. **Career Feedback** – generate personalised feedback and coaching suggestions.

> Tip: The app stores intermediate results in the current session so you can
> seamlessly continue through the stages without re-uploading your resume.
"""
)

st.info(f"Version: {get_version()} • Use the left sidebar to select a page.")

