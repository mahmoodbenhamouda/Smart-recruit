from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PAGE_PATH = Path(__file__).resolve()
PROJECT_ROOT = PAGE_PATH.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from smart_resume_suite.services.feedback_service import FeedbackEngine
from smart_resume_suite.session_keys import (
    ATS_RESULTS,
    FEEDBACK_RESULTS,
    JOB_DESCRIPTION,
    RESUME_TEXT,
    USER_QUESTION,
)

st.set_page_config(page_title="Career Feedback", page_icon="🤝", layout="wide")

st.title("🤝 Step 3 – Career Feedback & Coaching")

resume_text = st.session_state.get(RESUME_TEXT, "")
job_description = st.session_state.get(JOB_DESCRIPTION, "")

if not resume_text:
    st.error("Resume text unavailable. Complete Steps 1 and 2 first.")
    st.stop()

@st.cache_resource
def get_engine():
    return FeedbackEngine()


try:
    engine = get_engine()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.markdown(
    """
Using the SmartRecruiter dataset and Groq-powered prompts, this stage generates
tailored rejection feedback, coaching advice, and alternative role suggestions.
"""
)

default_question = st.session_state.get(
    USER_QUESTION, "What improvements should I focus on next?"
)
user_question = st.text_input("Candidate question", value=default_question)

if st.button("Generate Feedback", type="primary", use_container_width=True):
    with st.spinner("Generating feedback..."):
        feedback = engine.generate_feedback(
            cv_text=resume_text,
            job_description=job_description,
            user_question=user_question,
        )
    st.session_state[FEEDBACK_RESULTS] = feedback
    st.session_state[USER_QUESTION] = user_question
    st.success("Feedback generated.")

feedback_results = st.session_state.get(FEEDBACK_RESULTS)

if not feedback_results:
    st.info("Generate feedback to view results.")
    st.stop()

st.subheader("Final Response")
st.write(feedback_results.get("final_answer", "No feedback available."))

retrieved_jobs = feedback_results.get("retrieved_jobs", [])
if retrieved_jobs:
    st.subheader("Top Retrieved Dataset Jobs")
    for job in retrieved_jobs:
        with st.expander(f"{job['job_title']} (similarity {job['similarity']:.2f})"):
            st.write(f"**Reason:** {job['reason_for_decision']}")
            st.write(job["job_description"])

if "error" in feedback_results:
    st.warning(feedback_results["error"])
    st.info(
        "Set GROQ_API_KEY in a .env file or environment variable to enable AI-generated messaging."
    )

st.markdown("---")
if st.session_state.get(ATS_RESULTS):
    st.caption("ATS insights from Step 2 are retained in session state for further use.")

