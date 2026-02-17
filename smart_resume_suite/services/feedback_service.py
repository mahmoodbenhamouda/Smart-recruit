from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from .. import config

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "your",
    "will",
    "have",
    "about",
    "using",
    "such",
    "their",
    "each",
    "into",
    "over",
    "also",
}


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z]{3,}", text.lower())


def _derive_skill_gaps(cv_text: str, job_description: str, limit: int = 8) -> List[Dict[str, Any]]:
    if not job_description:
        return []

    resume_tokens = set(_tokenize(cv_text))
    job_tokens = [
        token for token in _tokenize(job_description) if token not in _STOPWORDS
    ]
    counter = Counter(job_tokens)
    candidates = [
        (token, freq) for token, freq in counter.items() if token not in resume_tokens
    ]
    if not candidates:
        return []

    total = sum(freq for _, freq in candidates) or 1
    insights: List[Dict[str, Any]] = []
    for token, freq in sorted(candidates, key=lambda item: item[1], reverse=True)[:limit]:
        importance = min(1.0, freq / total * 1.5)
        insights.append(
            {
                "skill": token,
                "importance": round(importance, 4),
                "rationale": f"Skill '{token}' appears {freq} times in the job description but not in the resume.",
                "evidence": [f"Job description mentions '{token}' {freq} times."],
            }
        )
    return insights


def _build_feedback_breakdown(
    feedback_text: str,
    coaching_text: str,
    matches_text: str,
    skill_gaps: List[Dict[str, Any]],
    retrieved_jobs: List["RetrievedJob"],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    intermediate_sections = {
        "feedback": feedback_text,
        "coaching": coaching_text,
        "alternative_roles": matches_text,
    }

    skill_lookup = [entry["skill"] for entry in skill_gaps[:5]]
    job_indices = [job.index for job in retrieved_jobs[:3]]

    breakdown: List[Dict[str, Any]] = []
    for key, text in intermediate_sections.items():
        lines = [
            line.strip(" -•\u2022")
            for line in text.splitlines()
            if line.strip()
        ]
        if not lines:
            continue
        summary = lines[0]
        bullets = lines[1:] or lines[:1]
        breakdown.append(
            {
                "section": key,
                "summary": summary,
                "bullets": bullets,
                "supporting_skills": skill_lookup,
                "supporting_jobs": job_indices,
            }
        )

    return breakdown, intermediate_sections

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = WORKSPACE_ROOT / "deep_Learning_Project" / "resume_screening_dataset_train.csv"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5


@dataclass
class RetrievedJob:
    index: int
    similarity: float
    job_title: str
    job_description: str
    reason_for_decision: str


class FeedbackEngine:
    """
    Lightweight wrapper around the SmartRecruiter LangGraph pipeline.
    It reuses the same prompts but avoids the Streamlit-specific orchestration.
    """

    def __init__(self):
        if not DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at {DATASET_PATH}. "
                "Place resume_screening_dataset_train.csv inside deep_Learning_Project."
            )
        self.df = self._load_dataset()
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.nn = self._build_index()

    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(DATASET_PATH)
        df.columns = [c.lower().strip() for c in df.columns]
        if "job_title" not in df.columns:
            if "role" in df.columns:
                df = df.rename(columns={"role": "job_title"})
            else:
                df["job_title"] = [f"Job {i+1}" for i in range(len(df))]
        if "job_description" not in df.columns:
            desc_col = next(
                (c for c in df.columns if "description" in c), None
            )
            if desc_col:
                df = df.rename(columns={desc_col: "job_description"})
            else:
                text_cols = [c for c in df.columns if df[c].dtype == "object"]
                df["job_description"] = df[text_cols].astype(str).agg(" ".join, axis=1)
        if "reason_for_decision" not in df.columns:
            df["reason_for_decision"] = "Not specified"
        df = df[df["job_description"].astype(str).str.strip() != ""].reset_index(drop=True)
        return df

    def _build_index(self):
        corpus = (
            self.df["job_title"].astype(str)
            + " | "
            + self.df["job_description"].astype(str)
            + " | Reason: "
            + self.df["reason_for_decision"].astype(str)
        ).tolist()
        vectors = self.embedder.encode(corpus, convert_to_numpy=True)
        nn = NearestNeighbors(
            n_neighbors=min(TOP_K, len(vectors)), metric="cosine"
        )
        nn.fit(vectors)
        self._vectors = vectors
        return nn

    def retrieve_jobs(self, query: str, top_k: int = TOP_K) -> List[RetrievedJob]:
        qv = self.embedder.encode([query], convert_to_numpy=True)
        dist, idx = self.nn.kneighbors(qv, n_neighbors=min(top_k, len(self.df)))
        retrieved: List[RetrievedJob] = []
        for i, d in zip(idx[0], dist[0]):
            row = self.df.iloc[int(i)]
            retrieved.append(
                RetrievedJob(
                    index=int(i),
                    similarity=1 - float(d),
                    job_title=row["job_title"],
                    job_description=str(row["job_description"]),
                    reason_for_decision=str(row["reason_for_decision"]),
                )
            )
        return retrieved

    def _context_block(self, jobs: List[RetrievedJob]) -> str:
        lines = []
        for job in jobs:
            snippet = job.job_description.replace("\n", " ")
            if len(snippet) > 420:
                snippet = snippet[:420] + "..."
            lines.append(
                f"- {job.job_title} (sim {job.similarity:.2f})\n  Desc: {snippet}\n  Reason: {job.reason_for_decision}"
            )
        return "\n".join(lines)

    def _call_groq(self, prompt: str) -> str:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "Groq API key is missing. Provide GROQ_API_KEY to enable AI feedback."
            )
        import groq
        import httpx

        http_client = httpx.Client(trust_env=False, timeout=60)
        try:
            client = groq.Groq(api_key=config.GROQ_API_KEY, http_client=http_client)
            chat = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an empathetic HR assistant who gives concrete, constructive feedback.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=900,
            )
            return chat.choices[0].message.content
        except groq.PermissionDeniedError as exc:
            raise RuntimeError("Groq API request denied. Check network/proxy settings or API key.") from exc
        except groq.GroqError as exc:
            raise RuntimeError(f"Groq API error: {exc}") from exc
        except httpx.HTTPError as exc:
            raise RuntimeError(f"HTTP error contacting Groq: {exc}") from exc

    def generate_feedback(
        self,
        cv_text: str,
        job_description: str,
        user_question: str,
    ) -> Dict[str, Any]:
        if not cv_text.strip():
            raise ValueError("CV text is required to generate feedback.")

        query = cv_text + "\n\n" + job_description + "\n\n" + user_question
        retrieved = self.retrieve_jobs(query)
        selected_job = retrieved[0] if retrieved else None
        skill_gap_insights = _derive_skill_gaps(cv_text, job_description)
        if not selected_job:
            breakdown, intermediate_sections = _build_feedback_breakdown(
                feedback_text="Unable to generate detailed feedback without relevant comparables.",
                coaching_text="Review and expand the resume with quantifiable achievements and targeted keywords.",
                matches_text="Consider broadening job search filters to find closer matches.",
                skill_gaps=skill_gap_insights,
                retrieved_jobs=retrieved,
            )
            return {
                "final_answer": "Could not find relevant jobs in the dataset to generate feedback.",
                "retrieved_jobs": [job.__dict__ for job in retrieved],
                "skill_gap_insights": skill_gap_insights,
                "feedback_breakdown": breakdown,
                "intermediate_sections": intermediate_sections,
            }

        context_block = self._context_block(retrieved)

        prompt_feedback = f"""
The candidate is evaluated for **{selected_job.job_title}** role.
Dataset rejection reason sample: "{selected_job.reason_for_decision}".
Candidate CV (truncated):
{cv_text[:1500]}
Top matched dataset records:
{context_block}
Candidate question:
{user_question or 'Why was I rejected?'}
TASK: Provide 6-8 lines of respectful, actionable rejection feedback grounded in the context.
"""

        prompt_coaching = f"""
Context:
- Target job: {selected_job.job_title}
- Common rejection reason: {selected_job.reason_for_decision}
- Retrieved matches:
{context_block}
- Candidate CV (truncated): {cv_text[:800]}
TASK: Suggest 3-5 concrete upskilling steps (skills, projects, certifications) achievable in 6-12 weeks.
"""

        prompt_matches = f"""
We compared the candidate to: {selected_job.job_title}.
Using the retrieved context:
{context_block}
TASK: Recommend 2-3 alternative job roles that currently fit the candidate's profile better, with one-sentence rationale each.
"""

        prompt_synth = f"""
You are the Lead HR Assistant at SmartRecruiter.
Merge the sections below into a cohesive 10-14 line response with headings Feedback, Coaching, Alternative Roles.
Avoid repetition and remain empathetic yet specific.

FEEDBACK:
{{feedback}}

COACHING:
{{coaching}}

ALTERNATIVE ROLES:
{{matches}}
"""

        try:
            feedback = self._call_groq(prompt_feedback)
            coaching = self._call_groq(prompt_coaching)
            matches = self._call_groq(prompt_matches)
            final_prompt = prompt_synth.format(
                feedback=feedback, coaching=coaching, matches=matches
            )
            final_answer = self._call_groq(final_prompt)
            breakdown, intermediate_sections = _build_feedback_breakdown(
                feedback_text=feedback,
                coaching_text=coaching,
                matches_text=matches,
                skill_gaps=skill_gap_insights,
                retrieved_jobs=retrieved,
            )
            return {
                "retrieved_jobs": [job.__dict__ for job in retrieved],
                "feedback": feedback,
                "coaching": coaching,
                "matches": matches,
                "final_answer": final_answer,
                "skill_gap_insights": skill_gap_insights,
                "feedback_breakdown": breakdown,
                "intermediate_sections": intermediate_sections,
            }
        except RuntimeError as exc:
            summary_lines = [
                "AI feedback unavailable (missing GROQ_API_KEY).",
                f"Closest dataset role: {selected_job.job_title}",
                f"Typical rejection reason: {selected_job.reason_for_decision}",
                "Consider improving alignment with job description keywords and highlighting measurable achievements.",
            ]
            fallback_feedback = "\n".join(summary_lines)
            breakdown, intermediate_sections = _build_feedback_breakdown(
                feedback_text=fallback_feedback,
                coaching_text="Focus on the highlighted skill gaps to align more closely with the role.",
                matches_text="Review alternative roles suggested by the dataset retrieval step.",
                skill_gaps=skill_gap_insights,
                retrieved_jobs=retrieved,
            )
            return {
                "retrieved_jobs": [job.__dict__ for job in retrieved],
                "final_answer": "\n".join(summary_lines),
                "error": str(exc),
                "skill_gap_insights": skill_gap_insights,
                "feedback_breakdown": breakdown,
                "intermediate_sections": intermediate_sections,
            }

