import os
import random
import textwrap
from typing import TypedDict, List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from dotenv import load_dotenv
import plotly.graph_objects as go

# Load environment variables from .env file
load_dotenv()

# ===== XAI Integration =====
try:
    from xai_explainer import XAIExplainer
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("⚠️ XAI Explainer not available. Install xgboost, shap, lime.")

# ===== LangGraph (lightweight orchestration) =====
from langgraph.graph import StateGraph, END

# =========================
# Branding / Page Config
# =========================
st.set_page_config(
    page_title="SmartRecruiter — Candidate Feedback Assistant",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_TITLE = "SmartRecruiter"
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FALLBACK_PATH = os.path.join(SCRIPT_DIR, "resume_screening_dataset_train.csv")
TOP_K = 5
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"

# 🔑 Load Groq API Key from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# =========================
# Corporate CSS
# =========================
CORP_CSS = """
<style>
:root {
  --bg: #f6f8fb; --card:#fff; --text:#0f172a; --muted:#64748b;
  --accent:#2563eb; --accent-weak:#eff6ff; --shadow:0 6px 24px rgba(15,23,42,.06); --radius:14px;
}
html, body [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--text); font-family:Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
.block-container{ padding-top:1rem; padding-bottom:2rem; }

/* Header */
.brand-bar{display:flex;align-items:center;gap:12px;margin:0 0 18px 0;padding:14px 18px;background:linear-gradient(135deg,#fff,#f3f7ff);border:1px solid #e6eefc;border-radius:var(--radius);box-shadow:var(--shadow);}
.brand-logo{width:42px;height:42px;border-radius:10px;background:linear-gradient(135deg,#2563eb,#60a5fa);display:inline-flex;align-items:center;justify-content:center;color:#fff;font-weight:700;}
.brand-title{font-size:20px;font-weight:700;color:#0f172a;}
.brand-sub{font-size:13px;color:var(--muted);}

/* Job card */
.job-card{background:var(--card);border:1px solid #e6eefc;border-radius:var(--radius);padding:18px;box-shadow:var(--shadow);}
.job-title{font-weight:700;font-size:18px;margin-bottom:6px;color:#0f172a;}
.job-chip{display:inline-block;font-size:12px;color:#1e40af;background:var(--accent-weak);border:1px solid #c7dcff;padding:4px 8px;border-radius:999px;margin-right:8px;}
.job-desc{color:#0f172a;line-height:1.5;}
.job-reason{color:#334155;font-size:13px;background:#eef2ff;padding:6px 10px;border-radius:10px;display:inline-block;}

/* Input panel */
.panel{background:var(--card);border:1px solid #e6eefc;border-radius:var(--radius);padding:16px;box-shadow:var(--shadow);}

/* Chat */
.chat-wrap{background:var(--card);border:1px solid #e6eefc;border-radius:var(--radius);box-shadow:var(--shadow);padding:14px;height:520px;overflow-y:auto;}
.msg{max-width:78%;padding:10px 14px;border-radius:16px;margin:8px 0;line-height:1.45;}
.msg-user{background:#e7f0ff;color:#0f172a;border:1px solid #cfe3ff;margin-left:auto;border-bottom-right-radius:6px;}
.msg-bot{background:#f8fafc;border:1px solid #e2e8f0;color:#0f172a;margin-right:auto;border-bottom-left-radius:6px;}
.meta{font-size:12px;color:var(--muted);margin-top:2px;}
.footer-note{color:var(--muted);font-size:12px;margin-top:6px;}
.send-btn button{width:100%;}
</style>
"""
st.markdown(CORP_CSS, unsafe_allow_html=True)

# =========================
# Utils
# =========================
def ensure_groq():
    if not GROQ_API_KEY or GROQ_API_KEY.startswith("gsk_your_real"):
        st.error("Groq API key is not set. Set GROQ_API_KEY env var or edit it in app.py.")
        st.stop()

def call_groq(prompt: str) -> str:
    """Call Groq API with proxy handling"""
    import groq
    
    # Try simple initialization first
    try:
        client = groq.Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        return f"[Groq API Error: Could not initialize client - {str(e)}]"

    try:
        chat = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an empathetic HR assistant who gives concrete, constructive feedback."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=900,
        )
        return chat.choices[0].message.content
    except Exception as e:
        return f"[Groq API Error: {str(e)}]"

def extract_text_from_pdf(uploaded_pdf) -> str:
    text = ""
    reader = PdfReader(uploaded_pdf)
    for page in reader.pages:
        page_txt = page.extract_text() or ""
        text += page_txt + "\n"
    return text.strip()

# =========================
# Data loading & indexing
# =========================
@st.cache_resource(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    if not os.path.exists(FALLBACK_PATH):
        st.error(f"❌ Dataset not found at {FALLBACK_PATH}. Place it next to app.py.")
        st.stop()
    df = pd.read_csv(FALLBACK_PATH)
    df.columns = [c.lower().strip() for c in df.columns]
    if "job_title" not in df.columns:
        if "role" in df.columns:
            df = df.rename(columns={"role":"job_title"})
        else:
            for c in df.columns:
                if "title" in c: df = df.rename(columns={c:"job_title"}); break
            if "job_title" not in df.columns:
                df["job_title"] = [f"Job {i+1}" for i in range(len(df))]
    if "job_description" not in df.columns:
        for c in df.columns:
            if "description" in c: df = df.rename(columns={c:"job_description"}); break
        if "job_description" not in df.columns:
            text_cols = [c for c in df.columns if df[c].dtype=="object"]
            df["job_description"] = df[text_cols].astype(str).agg(" ".join, axis=1)
    if "reason_for_decision" not in df.columns:
        df["reason_for_decision"] = "Not specified"

    df["job_title"] = df["job_title"].fillna("").astype(str)
    df["job_description"] = df["job_description"].fillna("").astype(str)
    df["reason_for_decision"] = df["reason_for_decision"].fillna("Not specified").astype(str)
    df = df[df["job_description"].str.strip()!=""].reset_index(drop=True)
    return df

df = load_dataset()

# =========================
# XAI Explainer
# =========================
@st.cache_resource(show_spinner=False)
def load_xai_explainer():
    """Load XAI explainer for job role prediction"""
    if not XAI_AVAILABLE:
        return None
    try:
        model_path = os.path.join(SCRIPT_DIR, 'JobPrediction_Model')
        explainer = XAIExplainer(model_path=model_path)
        return explainer
    except Exception as e:
        st.warning(f"⚠️ XAI explainer could not be loaded: {e}")
        return None

xai_explainer = load_xai_explainer() if XAI_AVAILABLE else None

@st.cache_resource(show_spinner=False)
def build_index(df: pd.DataFrame):
    embedder = SentenceTransformer(EMBED_MODEL)
    corpus = (
        df["job_title"].astype(str) + " | " +
        df["job_description"].astype(str) + " | Reason: " +
        df["reason_for_decision"].astype(str)
    ).tolist()
    vectors = embedder.encode(corpus, convert_to_numpy=True)
    nn = NearestNeighbors(n_neighbors=min(TOP_K, len(vectors)), metric="cosine")
    nn.fit(vectors)
    return {"embedder": embedder, "vectors": vectors, "nn": nn}

index = build_index(df)

def retrieve_similar(query: str, index, top_k=TOP_K):
    qv = index["embedder"].encode([query], convert_to_numpy=True)
    dist, idx = index["nn"].kneighbors(qv, n_neighbors=min(top_k, len(df)))
    return [(int(i), 1 - float(d)) for i, d in zip(idx[0], dist[0])]

# =========================
# LangGraph state & nodes
# =========================
class SRState(TypedDict):
    cv_text: str
    user_question: str
    selected_job_idx: int
    retrieved: List[Tuple[int, float]]  # (row_index, similarity)
    # XAI fields
    xai_prediction: Dict[str, Any]  # XGBoost prediction with confidence
    xai_shap: Dict[str, Any]  # SHAP explanation
    xai_lime: Dict[str, Any]  # LIME explanation
    xai_missing_skills: Dict[str, Any]  # Missing skills analysis
    # Agent outputs
    feedback: str
    coaching: str
    matches: str
    final_answer: str

def node_retrieve(state: SRState) -> Dict[str, Any]:
    """RAG retrieval node: compute top-k similar job rows from CV + question."""
    query = (state["cv_text"] or "") + "\n\n" + (state["user_question"] or "")
    hits = retrieve_similar(query, index, TOP_K)
    return {"retrieved": hits}

def node_xai_predict(state: SRState) -> Dict[str, Any]:
    """XAI prediction node: predict job role with explainability"""
    if not xai_explainer:
        return {
            "xai_prediction": {},
            "xai_shap": {},
            "xai_lime": {},
            "xai_missing_skills": {}
        }
    
    try:
        # Extract skills/keywords from CV
        cv_text = state["cv_text"] or ""
        # Simple skill extraction (take technical terms)
        import re
        # Remove common boilerplate
        cv_clean = re.sub(r'(email|phone|linkedin|address|summary|objective):\s*\S+', '', cv_text, flags=re.IGNORECASE)
        cv_clean = re.sub(r'(professional|summary|experience|education|skills):', '', cv_clean, flags=re.IGNORECASE)
        cv_clean = ' '.join(cv_clean.split())[:2000]  # Limit to 2000 chars
        
        # Get XGBoost prediction
        prediction = xai_explainer.predict_with_xgboost(cv_clean, top_n=5)
        
        # Get SHAP explanation
        shap_result = xai_explainer.explain_with_shap(cv_clean)
        
        # Get LIME explanation
        lime_result = xai_explainer.explain_with_lime(cv_clean, num_features=10)
        
        # Analyze missing skills for the target job (from retrieved context)
        missing_skills_analysis = {}
        if state.get("selected_job_idx") is not None:
            target_job = df.iloc[state["selected_job_idx"]]
            target_role_guess = target_job.get('job_title', '')
            
            # Try to match target role to one of the predicted roles
            # If not found, analyze against the top predicted role
            if target_role_guess not in xai_explainer.job_roles:
                target_role_guess = prediction.get('predicted_role', 'Data Scientist')
            
            missing_skills_analysis = xai_explainer.analyze_missing_skills(
                cv_clean, 
                target_role_guess, 
                top_n=8
            )
        
        # Debug logging
        print(f"\n🔍 XAI Prediction Results:")
        print(f"   Predicted Role: {prediction.get('predicted_role', 'N/A')}")
        print(f"   Confidence: {prediction.get('confidence', 0):.2%}")
        print(f"   SHAP features: {len(shap_result.get('top_features', []))} features")
        if shap_result.get('top_features'):
            print(f"   Top SHAP feature: {shap_result['top_features'][0]}")
        if missing_skills_analysis:
            print(f"   Missing skills analyzed: {len(missing_skills_analysis.get('gap_analysis', {}).get('top_missing_skills', []))} skills")
        
        return {
            "xai_prediction": prediction,
            "xai_shap": shap_result,
            "xai_lime": lime_result,
            "xai_missing_skills": missing_skills_analysis
        }
    except Exception as e:
        # Return empty results on error (don't use st.warning in threads)
        print(f"XAI prediction failed: {e}")  # Log to console instead
        import traceback
        traceback.print_exc()
        return {
            "xai_prediction": {},
            "xai_shap": {},
            "xai_lime": {},
            "xai_missing_skills": {}
        }

def _context_block(hits: List[Tuple[int,float]]) -> str:
    chunks = []
    for idx, sim in hits:
        r = df.iloc[idx]
        chunks.append(
            f"- {r['job_title']} (sim {sim:.2f})\n  Desc: {r['job_description'][:420]}...\n  Reason: {r['reason_for_decision']}"
        )
    return "\n".join(chunks)

def node_feedback(state: SRState) -> Dict[str, Any]:
    job = df.iloc[state["selected_job_idx"]]
    
    # Add XAI insights to the prompt
    xai_info = ""
    if state.get("xai_prediction") and state["xai_prediction"]:
        pred = state["xai_prediction"]
        xai_info = f"""
**AI-Powered Job Role Analysis:**
- Predicted Best Fit Role: **{pred.get('predicted_role', 'N/A')}** (Confidence: {pred.get('confidence', 0):.1%})
- Top 3 Role Matches:
"""
        for i, p in enumerate(pred.get('top_predictions', [])[:3], 1):
            xai_info += f"  {i}. {p['role']}: {p['probability']:.1%}\n"
        
        # Add SHAP insights
        if state.get("xai_shap") and state["xai_shap"].get('top_features'):
            xai_info += "\n**Key Skills Driving the Prediction (SHAP Analysis):**\n"
            for feat in state["xai_shap"]['top_features'][:5]:
                impact = "✅" if feat['impact'] == 'positive' else "❌"
                xai_info += f"  {impact} {feat['feature']}: {feat['shap_value']*100:+.2f}pp\n"
    
    prompt = f"""
The candidate appears to be evaluated against the position: **{job['job_title']}**.
A common decision reason in similar cases: "{job['reason_for_decision']}".

{xai_info}

Candidate CV:
{state['cv_text'][:1500]}

Top matching job descriptions with real rejection reasons:
{_context_block(state['retrieved'])}

Candidate's question:
{state['user_question']}

TASK: As a Feedback Agent, kindly explain the most likely reasons for rejection grounded in the context and XAI analysis.
If the XAI predicted role differs from the target job, explain the skill gaps.
Limit to ~8-10 lines, specific, respectful, and actionable.
"""
    return {"feedback": call_groq(prompt)}

def node_coaching(state: SRState) -> Dict[str, Any]:
    job = df.iloc[state["selected_job_idx"]]
    
    # Add skill gap analysis from XAI
    skill_gaps = ""
    if state.get("xai_shap") and state["xai_shap"].get('top_features'):
        # Find negative impact features (missing skills)
        missing = [f for f in state["xai_shap"]['top_features'] if f['impact'] == 'negative']
        if missing:
            skill_gaps = "\n**Skill Gaps Identified by AI:**\n"
            for feat in missing[:3]:
                skill_gaps += f"  - {feat['feature']}\n"
    
    prompt = f"""
Context:
- Target job: {job['job_title']}
- Typical decision reasons: {job['reason_for_decision']}
- Retrieved matches:
{_context_block(state['retrieved'])}

{skill_gaps}

- Candidate CV (truncated): {state['cv_text'][:800]}

TASK: As a Career Coach Agent, propose 4–6 concrete steps (skills, courses, project ideas, certifications) to close gaps.
Focus on the skill gaps identified by the AI analysis above.
Keep it realistic for the next 6–12 weeks with specific resources (Coursera, Udemy, etc.).
"""
    return {"coaching": call_groq(prompt)}

def node_match(state: SRState) -> Dict[str, Any]:
    job = df.iloc[state["selected_job_idx"]]
    
    # Use XAI predictions for alternative roles
    alt_roles_from_ai = ""
    if state.get("xai_prediction") and state["xai_prediction"].get('top_predictions'):
        alt_roles_from_ai = "\n**AI-Suggested Alternative Roles (Based on Your Skills):**\n"
        for pred in state["xai_prediction"]['top_predictions'][1:4]:  # Skip the top one, show next 3
            alt_roles_from_ai += f"  - {pred['role']}: {pred['probability']:.1%} match\n"
    
    prompt = f"""
We compared the candidate to: {job['job_title']}.

{alt_roles_from_ai}

Using the retrieved context:
{_context_block(state['retrieved'])}

TASK: As a Matcher Agent, suggest 3–4 alternative job roles or tracks that fit the candidate's current profile better.
Prioritize the AI-suggested roles above, but also consider the retrieved matches.
For each role, give a one-sentence rationale based on the candidate's demonstrated skills.
Keep it concise and practical.
"""
    return {"matches": call_groq(prompt)}

def node_synthesize(state: SRState) -> Dict[str, Any]:
    job = df.iloc[state["selected_job_idx"]]
    
    # Build XAI summary for the final output
    xai_summary = ""
    if state.get("xai_prediction") and state["xai_prediction"]:
        pred = state["xai_prediction"]
        xai_summary = f"""
═══════════════════════════════════════════════════════════════
📊 **XAI JOB ROLE ANALYSIS** (Explainable AI Insights)
═══════════════════════════════════════════════════════════════

🎯 **AI-Predicted Best Fit Role:** {pred.get('predicted_role', 'N/A')}
   • Confidence Score: {pred.get('confidence', 0):.1%}
   • Model: XGBoost Classifier (45 job roles, 96.27% accuracy)

📈 **Top 5 Predicted Roles:**
"""
        for i, p in enumerate(pred.get('top_predictions', [])[:5], 1):
            xai_summary += f"   {i}. {p['role']}: {p['probability']:.2%}\n"
        
        # Add SHAP feature importance
        if state.get("xai_shap") and state["xai_shap"].get('top_features'):
            xai_summary += "\n🔍 **SHAP Feature Importance Analysis:**\n"
            xai_summary += "   (Percentage point contribution to prediction probability)\n\n"
            for i, feat in enumerate(state["xai_shap"]['top_features'][:8], 1):
                impact_icon = "✅" if feat['impact'] == 'positive' else "⚠️"
                impact_label = "Strength" if feat['impact'] == 'positive' else "Skill Gap"
                xai_summary += f"   {i}. {impact_icon} **{feat['feature']}**: {feat['shap_value']*100:+.2f}pp ({impact_label})\n"
        
        # Add LIME explanation
        if state.get("xai_lime") and state["xai_lime"].get('explanation'):
            xai_summary += f"\n💡 **LIME Local Interpretability:**\n   {state['xai_lime']['explanation']}\n"
        
        # Add Missing Skills Analysis
        if state.get("xai_missing_skills") and state["xai_missing_skills"].get('gap_analysis'):
            gap = state["xai_missing_skills"]['gap_analysis']
            xai_summary += f"\n\n⚠️ **MISSING SKILLS ANALYSIS** (SHAP-Based Gap Detection)\n"
            xai_summary += f"═══════════════════════════════════════════════════════════════\n"
            xai_summary += f"Target Role: {state['xai_missing_skills'].get('target_role', 'N/A')}\n"
            xai_summary += f"Current Match: {state['xai_missing_skills'].get('current_match_probability', 0):.1%}\n"
            xai_summary += f"Potential Improvement: +{gap.get('total_potential_improvement', 0)*100:.1f}%\n\n"
            xai_summary += "**Top Missing Skills (If Added to Profile):**\n"
            
            for i, skill in enumerate(gap.get('top_missing_skills', [])[:8], 1):
                priority_emoji = "🔴" if skill['priority'] == 'High' else "🟡" if skill['priority'] == 'Medium' else "🟢"
                xai_summary += f"   {i}. {priority_emoji} **{skill['skill']}**: +{skill['impact_percentage']:.2f}% impact ({skill['priority']} Priority)\n"
            
            if state['xai_missing_skills'].get('recommendation'):
                xai_summary += f"\n💡 **Recommendation:**\n{state['xai_missing_skills']['recommendation']}\n"
        
        xai_summary += "\n═══════════════════════════════════════════════════════════════\n"
    
    prompt = f"""
You are the Lead HR Assistant at SmartRecruiter.
Create a comprehensive performance review and development plan.

CRITICAL INSTRUCTION: You MUST include ALL sections below in your response, preserving the XAI analysis data exactly as provided.

**REQUIRED OUTPUT STRUCTURE:**

1. **XAI Analysis** - Include the complete XAI Job Role Analysis below with ALL predicted roles and SHAP feature impacts
2. **Feedback** - Explain rejection reasons based on XAI insights
3. **Coaching** - Provide actionable steps to close skill gaps identified in SHAP analysis
4. **Alternative Roles** - Suggest roles based on predicted matches

---

TARGET JOB: {job['job_title']}

{xai_summary}

RETRIEVED CONTEXT:
{_context_block(state['retrieved'])}

FEEDBACK SECTION INPUT:
{state['feedback']}

COACHING SECTION INPUT:
{state['coaching']}

ALTERNATIVE ROLES INPUT:
{state['matches']}

---

FORMAT REQUIREMENTS:
- Start with "**Performance Review and Development Plan for [Candidate Name]**"
- Include the XAI Analysis section FIRST with all predicted roles and SHAP values
- Use bullet points (•) for lists
- Keep percentages and pp (percentage point) values exact
- Use ✅ for strengths and ⚠️ for gaps
- Be professional, empathetic, and data-driven
"""
    return {"final_answer": call_groq(prompt)}

# Build the LangGraph
graph = StateGraph(SRState)
graph.add_node("retrieve", node_retrieve)
graph.add_node("xai_predict", node_xai_predict)
graph.add_node("feedback", node_feedback)
graph.add_node("coaching", node_coaching)
graph.add_node("matches", node_match)
graph.add_node("synthesize", node_synthesize)

# Edges: start -> retrieve -> xai_predict -> (feedback, coaching, matches) -> synthesize -> END
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "xai_predict")
graph.add_edge("xai_predict", "feedback")
graph.add_edge("xai_predict", "coaching")
graph.add_edge("xai_predict", "matches")
graph.add_edge("feedback", "synthesize")
graph.add_edge("coaching", "synthesize")
graph.add_edge("matches", "synthesize")
graph.add_edge("synthesize", END)

compiled_graph = graph.compile()

# =========================
# Session state (chat)
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role":"user"/"assistant","content":str}
if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

# =========================
# Header
# =========================
st.markdown(
    f"""
<div class="brand-bar">
  <div class="brand-logo">SR</div>
  <div>
    <div class="brand-title">{APP_TITLE}</div>
    <div class="brand-sub">LangGraph Orchestration • RAG • Groq • Analytics</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Random job on reload
random_row = df.sample(1, random_state=None).iloc[0]
selected_idx = int(random_row.name)

# =========================
# Tabs: Chat | XAI Analysis | Analytics
# =========================
tab_chat, tab_xai, tab_dash = st.tabs(["💬 Chat", "🔍 XAI Analysis", "📊 Analytics"])

with tab_chat:
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.markdown(
            f"""
<div class="job-card">
  <div class="job-title">{random_row['job_title']}</div>
  <div style="margin-bottom: 8px;">
    <span class="job-chip">Selected automatically</span>
    <span class="job-chip">Randomized on reload</span>
  </div>
  <div class="job-desc">{textwrap.shorten(random_row['job_description'], width=550, placeholder="…")}</div>
  <div style="margin-top:10px;">
    <span class="job-reason"><b>Dataset Decision Reason:</b> {random_row['reason_for_decision']}</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Upload your CV (PDF)**")
        uploaded_pdf = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
        if uploaded_pdf is not None:
            st.session_state.cv_text = extract_text_from_pdf(uploaded_pdf)
            st.success("CV loaded successfully.")

        st.markdown("---")
        user_input = st.text_input("Ask a question", placeholder="Why was I rejected? What should I improve?")
        send = st.button("Send", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="chat-wrap" id="chat-window">', unsafe_allow_html=True)
        # Render history
        for m in st.session_state.messages:
            role_class = "msg-user" if m["role"] == "user" else "msg-bot"
            st.markdown(f'<div class="msg {role_class}">{m["content"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="footer-note">Conversation is grounded on the selected job plus top-matched roles from your dataset (via RAG + LangGraph).</div>', unsafe_allow_html=True)

    # Send action
    if send:
        ensure_groq()
        if not st.session_state.cv_text:
            st.warning("Please upload your CV (PDF) first for personalization.")
        elif not user_input.strip():
            st.warning("Please type a question.")
        else:
            # Append user message
            st.session_state.messages.append({"role":"user","content":user_input})

            # Run the LangGraph pipeline
            init_state: SRState = {
                "cv_text": st.session_state.cv_text,
                "user_question": user_input,
                "selected_job_idx": selected_idx,
                "retrieved": [],
                "xai_prediction": {},
                "xai_shap": {},
                "xai_lime": {},
                "xai_missing_skills": {},
                "feedback": "",
                "coaching": "",
                "matches": "",
                "final_answer": "",
            }
            
            with st.spinner("🤖 Analyzing your CV with AI..."):
                result: SRState = compiled_graph.invoke(init_state)
            
            answer = result.get("final_answer","(no answer)")

            # Append assistant message
            st.session_state.messages.append({"role":"assistant","content":answer})
            
            # Store XAI results in session for visualization
            if result.get("xai_prediction"):
                st.session_state["last_xai_prediction"] = result["xai_prediction"]
                st.session_state["last_xai_shap"] = result.get("xai_shap", {})
                st.session_state["last_xai_lime"] = result.get("xai_lime", {})
                st.session_state["last_xai_missing_skills"] = result.get("xai_missing_skills", {})
            
            st.rerun()

with tab_xai:
    st.subheader("🔍 Explainable AI Analysis")
    
    if not XAI_AVAILABLE:
        st.warning("⚠️ XAI features not available. Install xgboost, shap, lime.")
    elif "last_xai_prediction" not in st.session_state:
        st.info("📤 Upload your CV and ask a question in the Chat tab to see XAI analysis.")
    else:
        pred = st.session_state.get("last_xai_prediction", {})
        shap_res = st.session_state.get("last_xai_shap", {})
        lime_res = st.session_state.get("last_xai_lime", {})
        
        if pred:
            # Prediction Results
            st.markdown("### 🎯 Job Role Prediction")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Role", pred.get('predicted_role', 'N/A'))
            with col2:
                st.metric("Confidence", f"{pred.get('confidence', 0):.1%}")
            with col3:
                conf_level = "High" if pred.get('confidence', 0) > 0.5 else "Medium" if pred.get('confidence', 0) > 0.3 else "Low"
                st.metric("Confidence Level", conf_level)
            
            # Top predictions
            st.markdown("#### 📊 Top 5 Predictions")
            top_preds = pred.get('top_predictions', [])
            if top_preds:
                for i, p in enumerate(top_preds, 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{i}. {p['role']}**")
                    with col2:
                        st.progress(p['probability'])
                        st.write(f"{p['probability']:.1%}")
        
        # SHAP Explanation
        if shap_res and shap_res.get('top_features'):
            st.markdown("---")
            st.markdown("### 📊 SHAP Explanation (Feature Importance)")
            st.info("💡 SHAP values show how much each feature contributes to the prediction. Larger values = stronger influence.")
            
            features = shap_res['top_features'][:10]
            if features:
                df_shap = pd.DataFrame(features)
                df_shap['abs_shap'] = df_shap['shap_value'].abs()
                df_shap = df_shap.sort_values('abs_shap', ascending=False)
                df_shap['shap_scaled'] = df_shap['shap_value'] * 100
                
                # Bar chart
                colors = ['#2ecc71' if imp == 'positive' else '#e74c3c' for imp in df_shap['impact']]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=df_shap['shap_scaled'],
                        y=df_shap['feature'],
                        orientation='h',
                        marker=dict(color=colors),
                        text=[f"{val:+.3f}pp" for val in df_shap['shap_scaled']],
                        textposition='outside',
                        hovertemplate=(
                            '<b>%{y}</b><br>' +
                            'Contribution: %{x:.4f} percentage points<br>' +
                            '<extra></extra>'
                        )
                    )
                ])
                
                fig.update_layout(
                    title="SHAP Feature Contributions (percentage points)",
                    xaxis_title="Contribution to Prediction Probability (percentage points)",
                    yaxis_title="Feature",
                    height=max(400, len(df_shap) * 30),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Table
                st.markdown("##### Top Contributing Features:")
                display_df = df_shap[['feature', 'shap_scaled', 'impact']].copy()
                display_df.columns = ['Feature', 'Contribution (pp)', 'Impact']
                display_df['Contribution (pp)'] = display_df['Contribution (pp)'].apply(lambda x: f"{x:+.3f}")
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                st.caption("**pp** = percentage points. E.g., +0.250pp means this feature increases the probability by 0.25%")
        
        # LIME Explanation
        if lime_res and lime_res.get('explanation'):
            st.markdown("---")
            st.markdown("### 📊 LIME Explanation (Local Interpretability)")
            st.info("💡 LIME explains predictions by testing variations of your input. Shows which words/phrases most influenced the decision.")
            
            lime_features = lime_res['explanation'][:10]
            if lime_features:
                df_lime = pd.DataFrame(lime_features)
                df_lime = df_lime.sort_values('weight', key=abs, ascending=False)
                
                # Bar chart
                colors_lime = ['#3498db' if w > 0 else '#e67e22' for w in df_lime['weight']]
                
                fig_lime = go.Figure(data=[
                    go.Bar(
                        x=df_lime['weight'],
                        y=df_lime['feature'],
                        orientation='h',
                        marker=dict(color=colors_lime),
                        text=[f"{val:+.4f}" for val in df_lime['weight']],
                        textposition='outside'
                    )
                ])
                
                fig_lime.update_layout(
                    title="LIME Feature Weights",
                    xaxis_title="Weight (Impact on Prediction)",
                    yaxis_title="Feature",
                    height=max(400, len(df_lime) * 30),
                    showlegend=False
                )
                
                st.plotly_chart(fig_lime, use_container_width=True)
                
                st.markdown("**Summary:** " + lime_res.get('summary', 'N/A'))
        
        # Missing Skills Analysis
        missing_skills_res = st.session_state.get("last_xai_missing_skills", {})
        if missing_skills_res and missing_skills_res.get('gap_analysis'):
            st.markdown("---")
            st.markdown("### ⚠️ Missing Skills Analysis (SHAP-Based Gap Detection)")
            st.warning("💡 This analysis shows which skills, if added to your profile, would most improve your match for the target role.")
            
            gap = missing_skills_res['gap_analysis']
            target_role = missing_skills_res.get('target_role', 'N/A')
            current_match = missing_skills_res.get('current_match_probability', 0)
            potential_improvement = gap.get('total_potential_improvement', 0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Target Role", target_role)
            with col2:
                st.metric("Current Match", f"{current_match:.1%}")
            with col3:
                st.metric("Potential Gain", f"+{potential_improvement*100:.1f}%", delta="If top skills added")
            
            st.markdown("#### 🎯 Top Missing Skills (High Impact)")
            missing_skills = gap.get('top_missing_skills', [])
            
            if missing_skills:
                df_missing = pd.DataFrame(missing_skills)
                df_missing = df_missing.sort_values('impact_if_added', ascending=False)
                
                # Bar chart for missing skills impact
                colors_priority = []
                for priority in df_missing['priority']:
                    if priority == 'High':
                        colors_priority.append('#e74c3c')  # Red
                    elif priority == 'Medium':
                        colors_priority.append('#f39c12')  # Orange
                    else:
                        colors_priority.append('#3498db')  # Blue
                
                fig_missing = go.Figure(data=[
                    go.Bar(
                        x=df_missing['impact_percentage'],
                        y=df_missing['skill'],
                        orientation='h',
                        marker=dict(color=colors_priority),
                        text=[f"+{val:.2f}%" for val in df_missing['impact_percentage']],
                        textposition='outside',
                        hovertemplate=(
                            '<b>%{y}</b><br>' +
                            'Impact if added: +%{x:.2f}%<br>' +
                            '<extra></extra>'
                        )
                    )
                ])
                
                fig_missing.update_layout(
                    title=f"Skills Missing from Your Profile (Target: {target_role})",
                    xaxis_title="Impact on Match Probability (%)",
                    yaxis_title="Skill",
                    height=max(400, len(df_missing) * 35),
                    showlegend=False
                )
                
                st.plotly_chart(fig_missing, use_container_width=True)
                
                # Table view
                st.markdown("##### 📋 Detailed Missing Skills Breakdown")
                display_missing = df_missing[['skill', 'impact_percentage', 'priority']].copy()
                display_missing.columns = ['Skill', 'Impact if Added (%)', 'Priority']
                display_missing['Impact if Added (%)'] = display_missing['Impact if Added (%)'].apply(lambda x: f"+{x:.2f}%")
                st.dataframe(display_missing, use_container_width=True, hide_index=True)
                
                # Recommendation
                if missing_skills_res.get('recommendation'):
                    st.markdown("##### 💡 Personalized Recommendation")
                    st.info(missing_skills_res['recommendation'])
            
            # Show existing strengths
            existing_strengths = missing_skills_res.get('existing_strengths', [])
            if existing_strengths:
                st.markdown("---")
                st.markdown("#### ✅ Your Current Strengths")
                strengths_list = ", ".join([f"**{s['skill']}**" for s in existing_strengths[:10]])
                st.success(f"Strong skills detected in your profile: {strengths_list}")

with tab_dash:
    st.subheader("Candidate Analytics Dashboard")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Rejection Reasons (Top 10)**")
        reason_counts = df["reason_for_decision"].value_counts().head(10).reset_index()
        reason_counts.columns = ["reason_for_decision", "count"]
        st.bar_chart(reason_counts.set_index("reason_for_decision"))

    with colB:
        st.markdown("**Most Common Roles (Top 10)**")
        role_counts = df["job_title"].value_counts().head(10).reset_index()
        role_counts.columns = ["job_title", "count"]
        st.bar_chart(role_counts.set_index("job_title"))

    st.markdown("---")
    st.markdown("**Description Length Distribution**")
    desc_len = df["job_description"].str.len().describe()[["mean","50%","min","max"]].to_frame("value")
    st.table(desc_len)

    st.markdown("---")
    st.markdown("**Sample of Dataset (Top 10)**")
    st.dataframe(df.head(10), use_container_width=True)

st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
st.caption("© SmartRecruiter — LangGraph Orchestration • RAG • Groq • Streamlit Analytics")