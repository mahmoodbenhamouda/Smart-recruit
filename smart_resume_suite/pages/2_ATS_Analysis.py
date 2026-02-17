from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PAGE_PATH = Path(__file__).resolve()
PROJECT_ROOT = PAGE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_resume_suite.services import ats_service
from smart_resume_suite.session_keys import (
    ATS_RESULTS,
    JOB_DESCRIPTION,
    RESUME_PATH,
    RESUME_SKILLS,
    RESUME_TEXT,
)

st.set_page_config(page_title="ATS Analysis", page_icon="📊", layout="wide")

st.title("📊 Step 2 – ATS Analysis")

resume_path = st.session_state.get(RESUME_PATH)
resume_text = st.session_state.get(RESUME_TEXT, "")

if not resume_path:
    st.error("No resume available. Please complete Step 1 first.")
    st.stop()

st.markdown(
    """
Paste a job description to evaluate how well the parsed resume aligns. The ATS
pipeline replicates the original ATS-agent behaviour: keyword extraction,
similarity scoring, and optional RAG skill detection.
"""
)

job_description = st.text_area(
    "Job description",
    value=st.session_state.get(JOB_DESCRIPTION, ""),
    height=240,
    placeholder="Paste the target job description here...",
)

if st.button("Run ATS Analysis", type="primary", use_container_width=True):
    if not job_description.strip():
        st.warning("Provide a job description to continue.")
    else:
        with st.spinner("Running ATS pipeline..."):
            results = ats_service.run_ats_analysis(
                resume_pdf_path=Path(resume_path), job_description=job_description
            )
        st.session_state[ATS_RESULTS] = results
        st.session_state[JOB_DESCRIPTION] = job_description
        st.success("ATS analysis completed. Review the results below.")

results = st.session_state.get(ATS_RESULTS)

if not results:
    st.info("Run the analysis to see detailed results.")
    st.stop()

similarity = results.get("similarity_scores", {})
overall_match = similarity.get("overall_percentage", 0.0)
match_level = similarity.get("match_level", "N/A")
skills_match_rate = similarity.get("detailed_scores", {}).get("skills_match_rate", 0.0)
rag_skills = results.get("rag_skills", [])

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Overall Match", f"{overall_match:.1f}%")
    st.write(f"**Match Level:** {match_level}")
with col2:
    st.metric("Skills Match", f"{skills_match_rate*100:.1f}%")
    ats_format = results.get("format_analysis", {})
    if ats_format:
        st.metric("ATS Score", f"{ats_format.get('ats_friendly_score', 0)}/100")
    else:
        st.info("No format analysis data.")
with col3:
    st.metric("Detected Skills (RAG)", len(rag_skills))
    st.metric(
        "spaCy Skills",
        len(st.session_state.get(RESUME_SKILLS, [])),
    )

tabs = st.tabs(
    [
        "Matched vs Missing Skills",
        "Scoring Breakdown",
        "RAG Skills",
        "Resume Format",
    ]
)

matched = similarity.get("matched_skills", [])
missing = similarity.get("missing_skills", [])

with tabs[0]:
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("✅ Matched Skills")
        if matched:
            st.write(", ".join(sorted(matched)))
        else:
            st.info("No matched skills detected.")
    with col_b:
        st.subheader("❌ Missing Skills")
        if missing:
            st.write(", ".join(sorted(missing)))
        else:
            st.success("No missing skills detected.")

with tabs[1]:
    st.subheader("Component Scores")
    detailed = similarity.get("detailed_scores", {})
    st.progress(detailed.get("skills_match_rate", 0.0), text="Skills match rate")
    st.progress(detailed.get("all_keywords_match_rate", 0.0), text="Keyword overlap")
    st.progress(detailed.get("tfidf_match_rate", 0.0), text="TF-IDF similarity")
    st.progress(detailed.get("text_similarity", 0.0), text="Text similarity")
    st.subheader("Recommendations")
    for item in results.get("recommendations", []):
        st.write(f"- {item}")

with tabs[2]:
    st.subheader("RAG Skills")
    if rag_skills:
        st.write(", ".join(rag_skills[:200]))
        if len(rag_skills) > 200:
            st.caption(f"...and {len(rag_skills) - 200} more")
    else:
        st.info("No RAG skills detected. Ensure the CSV corpus is present.")

with tabs[3]:
    st.subheader("Resume Format Insights")
    format_analysis = results.get("format_analysis", {})
    if not format_analysis:
        st.info("Format analysis unavailable.")
    else:
        st.json(format_analysis)

st.markdown("---")
st.info(
    "Proceed to Step 3 for personalised feedback leveraging the SmartRecruiter dataset."
)

