from __future__ import annotations

import io
import sys
from pathlib import Path

import streamlit as st

PAGE_PATH = Path(__file__).resolve()
PROJECT_ROOT = PAGE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_resume_suite.services import cv_parser
from smart_resume_suite.session_keys import (
    OCR_USED,
    PARSER_ERRORS,
    RESUME_PARSED,
    RESUME_PATH,
    RESUME_SECTIONS,
    RESUME_SKILLS,
    RESUME_TEXT,
)

st.set_page_config(page_title="Resume Parsing", page_icon="🧠", layout="wide")

st.title("🧠 Step 1 – Resume Parsing")
st.markdown(
    "Upload a resume PDF. The parser combines PyPDF2 extraction, OCR fallback, "
    "LayoutLM-based section clustering, and spaCy skill recognition."
)

uploaded_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Parsing resume..."):
        parsed = cv_parser.parse_resume(
            io.BytesIO(uploaded_file.getbuffer()), uploaded_file.name
        )

    st.session_state[RESUME_PARSED] = True
    st.session_state[RESUME_TEXT] = parsed.raw_text
    st.session_state[RESUME_SECTIONS] = parsed.sections
    st.session_state[RESUME_SKILLS] = parsed.skill_entities
    st.session_state[RESUME_PATH] = str(parsed.source_path)
    st.session_state[OCR_USED] = parsed.ocr_used
    st.session_state[PARSER_ERRORS] = parsed.errors

    st.success("Resume parsed successfully. Continue to Step 2 from the sidebar.")

    if parsed.ocr_used:
        st.warning("OCR fallback was used because direct text extraction was limited.")

    if parsed.errors:
        with st.expander("Warnings and errors"):
            for err in parsed.errors:
                st.write(f"- {err}")

    st.subheader("Extracted Sections")
    if parsed.sections:
        for section in parsed.sections:
            with st.expander(section.get("section_name", "Section")):
                st.write(section.get("content", "")[:2000] + "...")
    else:
        st.info("No sections detected. Proceed to ATS analysis to continue.")

    st.subheader("Detected Skills (spaCy)")
    if parsed.skill_entities:
        st.write(", ".join(sorted(parsed.skill_entities)))
    else:
        st.info("No skill entities detected by the spaCy model.")

    st.subheader("Raw Text Preview")
    st.text_area(
        "First 4000 characters",
        parsed.raw_text[:4000],
        height=240,
    )
else:
    if st.session_state.get(RESUME_PARSED):
        st.success("Resume already parsed. You can re-upload to start over.")
    else:
        st.info("Upload a resume PDF to begin.")

