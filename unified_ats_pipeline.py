"""
Unified ATS Pipeline - Complete Resume Analysis System
Integrates: CV Parsing → Skills Extraction → ATS Scoring → Job Prediction → XAI Analysis → AI Feedback

Pipeline Flow:
1. Resume Upload → Section Extraction (LayoutLMv3 from pi5eme1)
2. Skills Extraction → ATS Score (from ATS-agent)
3. Job Role Prediction → XAI Analysis (XGBoost + SHAP/LIME)
4. AI-Powered Feedback (Groq LLM from deep_Learning_Project)
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Tuple
import streamlit as st
from PIL import Image
import torch
import pytesseract
from pdf2image import convert_from_path

# Configure paths
BASE_DIR = Path(r"C:\Users\Mahmoud\Desktop\integ")
PI5EME_DIR = BASE_DIR / "pi5eme1 - Copie (2)" / "pi5eme1 - Copie (2)"
ATS_AGENT_DIR = BASE_DIR / "ATS-agent" / "ATS-agent"
DEEP_LEARNING_DIR = BASE_DIR / "deep_Learning_Project"

# Add paths to system path
for path in [PI5EME_DIR, ATS_AGENT_DIR, DEEP_LEARNING_DIR]:
    sys.path.insert(0, str(path))

# ============================================================================
# STAGE 1: CV SECTION EXTRACTION (LayoutLMv3)
# ============================================================================
class CVSectionExtractor:
    """Extract structured sections from resume PDF using LayoutLMv3"""
    
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        self.poppler_path = r"C:\Release-25.07.0-0\poppler-25.07.0\Library\bin"
        self.model_available = False
        
        try:
            model_dir = PI5EME_DIR / "outputs" / "models" / "layoutlmv3_finetuned"
            
            if not model_dir.exists():
                print(f"⚠️ LayoutLMv3 model not found at {model_dir}")
                print("   Will use fallback text extraction")
                return
            
            from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.processor = LayoutLMv3Processor.from_pretrained(str(model_dir))
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_dir)).to(self.device)
            self.model_available = True
        except Exception as e:
            print(f"⚠️ Could not load LayoutLMv3 model: {e}")
            print("   Will use fallback text extraction")
            self.model_available = False
        
        self.label_list = [
            "O", "B-HEADER", "I-HEADER", "B-CONTACT", "I-CONTACT",
            "B-SUMMARY", "I-SUMMARY", "B-EDUCATION", "I-EDUCATION",
            "B-EXPERIENCE", "I-EXPERIENCE", "B-SKILLS", "I-SKILLS",
            "B-PROJECTS", "I-PROJECTS", "B-CERTIFICATIONS", "I-CERTIFICATIONS",
            "B-LANGUAGES", "I-LANGUAGES", "B-PUBLICATIONS", "I-PUBLICATIONS",
            "B-REFERENCES", "I-REFERENCES", "B-OTHER", "I-OTHER"
        ]
        self.id2label = {i: l for i, l in enumerate(self.label_list)}
    
    def _fallback_extraction(self, pdf_path: str) -> Dict[str, str]:
        """Fallback extraction using simple text parsing"""
        import PyPDF2
        
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            return {"error": f"Failed to extract text: {str(e)}"}
        
        # Simple section detection based on common headers
        sections = {
            "Personal Information": "",
            "Education": "",
            "Experience": "",
            "Skills": "",
            "Projects": "",
            "Other": ""
        }
        
        lines = text.split('\n')
        current_section = "Personal Information"
        
        section_keywords = {
            "education": ["education", "academic", "qualification", "degree"],
            "experience": ["experience", "work history", "employment", "professional"],
            "skills": ["skills", "technical skills", "competencies", "expertise"],
            "projects": ["projects", "portfolio", "personal projects"]
        }
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Detect section headers
            section_detected = False
            for section_name, keywords in section_keywords.items():
                if any(keyword in line_lower for keyword in keywords):
                    current_section = section_name.title()
                    section_detected = True
                    break
            
            if not section_detected:
                sections[current_section] += line + "\n"
        
        # Combine short sections
        if len(sections["Personal Information"]) < 50:
            sections["Personal Information"] = text[:500]  # First 500 chars
        
        return {k: v.strip() for k, v in sections.items() if v.strip()}
    
    def extract_sections(self, pdf_path: str) -> Dict[str, str]:
        """Extract structured sections from resume PDF"""
        if not self.model_available:
            # Fallback: Use simple text extraction
            return self._fallback_extraction(pdf_path)
        
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, poppler_path=self.poppler_path)
            
            all_sections = {}
            for page_num, image in enumerate(images, 1):
                # Extract words and bounding boxes using OCR
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                words = []
                boxes = []
                
                for i, word in enumerate(ocr_data['text']):
                    if word.strip():
                        x, y, w, h = (
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['width'][i],
                            ocr_data['height'][i]
                        )
                        words.append(word)
                        boxes.append([x, y, x + w, y + h])
                
                if not words:
                    continue
                
                # Prepare input for LayoutLMv3
                encoding = self.processor(
                    image,
                    words,
                    boxes=boxes,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**encoding)
                    predictions = outputs.logits.argmax(-1).squeeze().tolist()
                
                # Group words by section
                page_sections = self._group_sections(words, predictions)
                
                # Merge with existing sections
                for section, text in page_sections.items():
                    if section in all_sections:
                        all_sections[section] += " " + text
                    else:
                        all_sections[section] = text
            
            return all_sections
            
        except Exception as e:
            print(f"Error during LayoutLMv3 extraction: {e}")
            print("Falling back to simple text extraction...")
            return self._fallback_extraction(pdf_path)
    
    def _group_sections(self, words: List[str], predictions: List[int]) -> Dict[str, str]:
        """Group words by predicted section labels"""
        sections = {}
        current_section = None
        current_text = []
        
        for word, pred_id in zip(words, predictions):
            if isinstance(pred_id, list):
                pred_id = pred_id[0]
            
            label = self.id2label.get(pred_id, "O")
            
            if label.startswith("B-"):
                # Start of new section
                if current_section and current_text:
                    sections[current_section] = " ".join(current_text)
                current_section = label[2:]  # Remove "B-" prefix
                current_text = [word]
            elif label.startswith("I-") and current_section:
                # Continuation of section
                current_text.append(word)
            elif label == "O" and current_section:
                # Outside any section, but we have a current section
                current_text.append(word)
        
        # Add last section
        if current_section and current_text:
            sections[current_section] = " ".join(current_text)
        
        return sections


# ============================================================================
# STAGE 2: SKILLS EXTRACTION & ATS SCORING
# ============================================================================
class ATSScorer:
    """Extract skills and calculate ATS match score"""
    
    def __init__(self):
        from pdf_extractor import PDFExtractor
        from ats_pipeline import ATSPipeline
        from rag_skills_extractor import RAGSkillsExtractor
        from job_role_predictor import JobRolePredictor
        
        self.pdf_extractor = PDFExtractor()
        self.ats_pipeline = ATSPipeline(use_spacy=True)
        self.rag_extractor = RAGSkillsExtractor(
            skills_csv_path=str(ATS_AGENT_DIR / "data" / "skills_exploded (2).csv"),
            max_skills=10000
        )
        
        # Job role predictor
        try:
            model_path = ATS_AGENT_DIR / "JobPrediction_Model"
            self.job_predictor = JobRolePredictor(model_dir=str(model_path))
            self.job_prediction_available = True
        except Exception as e:
            print(f"Warning: Job prediction not available: {e}")
            self.job_prediction_available = False
    
    def analyze_resume(self, pdf_path: str, job_description: str) -> Dict[str, Any]:
        """Extract skills and calculate ATS score"""
        try:
            # Extract text from PDF
            resume_text = self.pdf_extractor.extract_text(pdf_path)
            
            # Run ATS analysis - pass pdf_path, not text
            ats_result = self.ats_pipeline.analyze(pdf_path, job_description, verbose=False)
            
            # Extract skills using RAG
            print("Extracting resume skills with RAG...")
            resume_skills = self.rag_extractor.extract_skills_rag(resume_text)
            print(f"   Found {len(resume_skills)} resume skills")
            
            print("Extracting job description skills with RAG...")
            jd_skills = self.rag_extractor.extract_skills_rag(job_description)
            print(f"   Found {len(jd_skills)} JD skills")
            
            # Also get skills from ATS pipeline (more reliable)
            ats_missing_skills = []
            if ats_result.get("success"):
                similarity_scores = ats_result.get("similarity_scores", {})
                ats_missing_skills = similarity_scores.get("missing_skills", [])
            
            # Calculate match using RAG skills
            matched_skills = set(resume_skills) & set(jd_skills)
            missing_skills_rag = set(jd_skills) - set(resume_skills)
            
            # Combine both missing skills lists (prefer ATS pipeline's results)
            if ats_missing_skills:
                missing_skills = list(set(ats_missing_skills))
                print(f"   Using ATS pipeline missing skills: {len(missing_skills)}")
            else:
                missing_skills = list(missing_skills_rag)
                print(f"   Using RAG missing skills: {len(missing_skills)}")
            
            match_percentage = (len(matched_skills) / len(jd_skills) * 100) if jd_skills else 0
            
            # Extract ATS score from the correct path
            ats_score = 0
            if ats_result.get("success"):
                similarity_scores = ats_result.get("similarity_scores", {})
                ats_score = similarity_scores.get("overall_percentage", 0)
                
                # Debug logging
                if ats_score == 0:
                    print(f"⚠️ Debug: ATS result keys: {ats_result.keys()}")
                    print(f"⚠️ Debug: Similarity scores keys: {similarity_scores.keys() if similarity_scores else 'None'}")
                    if similarity_scores:
                        print(f"⚠️ Debug: overall_percentage value: {similarity_scores.get('overall_percentage', 'NOT FOUND')}")
            
            result = {
                "resume_text": resume_text,
                "ats_score": ats_score,
                "resume_skills": resume_skills,
                "jd_skills": jd_skills,
                "matched_skills": list(matched_skills),
                "missing_skills": list(missing_skills),
                "match_percentage": match_percentage,
                "keyword_frequency": ats_result.get("keyword_frequency", {}),
                "ats_full_result": ats_result,  # Keep full result for debugging
            }
            
            # Add job prediction if available
            if self.job_prediction_available:
                try:
                    predicted_role, confidence = self.job_predictor.predict(resume_text)
                    result["predicted_role"] = predicted_role
                    result["prediction_confidence"] = confidence
                except:
                    pass
            
            return result
            
        except Exception as e:
            print(f"Error in ATS analysis: {e}")
            import traceback
            traceback.print_exc()
            return {}


# ============================================================================
# STAGE 3: XAI ANALYSIS (SHAP/LIME)
# ============================================================================
class XAIAnalyzer:
    """Explainable AI analysis using SHAP and LIME"""
    
    def __init__(self):
        # Change to deep_Learning_Project directory temporarily
        original_dir = os.getcwd()
        os.chdir(str(DEEP_LEARNING_DIR))
        
        try:
            from xai_explainer import XAIExplainer
            model_path = DEEP_LEARNING_DIR / 'JobPrediction_Model'
            self.explainer = XAIExplainer(model_path=str(model_path))
            self.available = True
        except Exception as e:
            print(f"Warning: XAI not available: {e}")
            self.available = False
        finally:
            os.chdir(original_dir)
    
    def analyze(self, resume_text: str, target_role: str = None) -> Dict[str, Any]:
        """Perform XAI analysis with SHAP and LIME"""
        if not self.available:
            return {"error": "XAI not available"}
        
        try:
            # Clean text
            import re
            clean_text = re.sub(r'(email|phone|linkedin|address|summary|objective):\s*\S+', '', resume_text, flags=re.IGNORECASE)
            clean_text = re.sub(r'(professional|summary|experience|education|skills):', '', clean_text, flags=re.IGNORECASE)
            clean_text = ' '.join(clean_text.split())[:2000]
            
            # Validate text length for LIME (needs at least 10 words)
            word_count = len(clean_text.split())
            if word_count < 10:
                return {
                    "error": f"Resume text too short for analysis ({word_count} words). Need at least 10 words.",
                    "prediction": None,
                    "shap": None,
                    "lime": None,
                    "missing_skills": {}
                }
            
            # Get prediction
            prediction = self.explainer.predict_with_xgboost(clean_text, top_n=5)
            
            # Get SHAP explanation
            shap_result = self.explainer.explain_with_shap(clean_text)
            
            # Get LIME explanation (only if enough words)
            lime_result = None
            try:
                if word_count >= 20:  # LIME needs more text
                    lime_result = self.explainer.explain_with_lime(clean_text, num_features=min(10, word_count//2))
                else:
                    lime_result = {"warning": f"Text too short for LIME analysis ({word_count} words)"}
            except Exception as e:
                lime_result = {"error": f"LIME analysis failed: {str(e)}"}
            
            # Analyze missing skills if target role provided
            missing_skills_analysis = {}
            
            # Determine which role to analyze
            role_to_analyze = None
            if target_role:
                role_to_analyze = target_role.strip()
            elif prediction and "predicted_role" in prediction:
                role_to_analyze = prediction["predicted_role"]
                print(f"   ℹ️ No target role provided, using predicted role: {role_to_analyze}")
            
            if role_to_analyze:
                # Normalize target role and check if it exists
                target_role_normalized = role_to_analyze.strip()
                
                # Try exact match first
                if target_role_normalized in self.explainer.job_roles:
                    try:
                        missing_skills_analysis = self.explainer.analyze_missing_skills(
                            clean_text,
                            target_role_normalized,
                            top_n=8
                        )
                        print(f"   ✅ Missing skills analysis completed for: {target_role_normalized}")
                    except Exception as e:
                        print(f"   ⚠️ Error in missing skills analysis: {str(e)}")
                        print(f"   ✅ Missing skills analysis completed for: {target_role_normalized}")
                    except Exception as e:
                        print(f"   ⚠️ Error in missing skills analysis: {str(e)}")
                        missing_skills_analysis = {
                            "error": f"Failed to analyze missing skills: {str(e)}"
                        }
                else:
                    # Try case-insensitive match
                    target_role_lower = target_role_normalized.lower()
                    matched_role = None
                    
                    for role in self.explainer.job_roles:
                        if role.lower() == target_role_lower:
                            matched_role = role
                            break
                    
                    if matched_role:
                        try:
                            missing_skills_analysis = self.explainer.analyze_missing_skills(
                                clean_text,
                                matched_role,
                                top_n=8
                            )
                            print(f"   ✅ Missing skills analysis completed for: {matched_role}")
                        except Exception as e:
                            print(f"   ⚠️ Error in missing skills analysis: {str(e)}")
                            missing_skills_analysis = {
                                "error": f"Failed to analyze missing skills: {str(e)}"
                            }
                    else:
                        print(f"   ⚠️ Role '{role_to_analyze}' not found in {len(self.explainer.job_roles)} available roles")
                        missing_skills_analysis = {
                            "error": f"Target role '{role_to_analyze}' not found",
                            "available_roles": self.explainer.job_roles[:20]  # Show first 20
                        }
            else:
                print("   ⚠️ No target role available for missing skills analysis")
            
            return {
                "prediction": prediction,
                "shap": shap_result,
                "lime": lime_result,
                "missing_skills": missing_skills_analysis
            }
            
        except Exception as e:
            print(f"Error in XAI analysis: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}


# ============================================================================
# STAGE 4: AI FEEDBACK (Groq LLM)
# ============================================================================
class AIFeedbackGenerator:
    """Generate AI-powered feedback using Groq"""
    
    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv(DEEP_LEARNING_DIR / ".env")
        
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        try:
            import groq
            self.groq_client = groq.Groq(api_key=self.groq_api_key)
            self.available = True
        except Exception as e:
            print(f"Warning: Groq not available: {e}")
            self.available = False
    
    def generate_feedback(
        self,
        resume_sections: Dict[str, str],
        ats_results: Dict[str, Any],
        xai_results: Dict[str, Any],
        job_description: str
    ) -> str:
        """Generate comprehensive AI feedback"""
        if not self.available:
            return "AI feedback not available - Groq API not configured"
        
        try:
            # Build context
            prompt = self._build_feedback_prompt(
                resume_sections,
                ats_results,
                xai_results,
                job_description
            )
            
            # Call Groq API
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an expert HR assistant and career coach specializing in resume analysis and job matching."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating feedback: {e}")
            return f"Error generating feedback: {str(e)}"
    
    def _build_feedback_prompt(
        self,
        resume_sections: Dict[str, str],
        ats_results: Dict[str, Any],
        xai_results: Dict[str, Any],
        job_description: str
    ) -> str:
        """Build comprehensive prompt for Groq"""
        
        # XAI Summary
        xai_summary = ""
        if xai_results and "prediction" in xai_results:
            pred = xai_results["prediction"]
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
            
            # Add SHAP analysis
            if xai_results.get("shap") and xai_results["shap"].get('top_features'):
                xai_summary += "\n🔍 **SHAP Feature Importance Analysis:**\n"
                for i, feat in enumerate(xai_results["shap"]['top_features'][:8], 1):
                    impact_icon = "✅" if feat['impact'] == 'positive' else "⚠️"
                    impact_label = "Strength" if feat['impact'] == 'positive' else "Skill Gap"
                    xai_summary += f"   {i}. {impact_icon} **{feat['feature']}**: {feat['shap_value']*100:+.2f}pp ({impact_label})\n"
            
            # Add missing skills analysis
            if xai_results.get("missing_skills") and xai_results["missing_skills"].get("gap_analysis"):
                gap = xai_results["missing_skills"]["gap_analysis"]
                xai_summary += f"\n\n⚠️ **MISSING SKILLS ANALYSIS** (SHAP-Based)\n"
                xai_summary += f"Target Role: {xai_results['missing_skills'].get('target_role', 'N/A')}\n"
                xai_summary += f"Current Match: {xai_results['missing_skills'].get('current_match_probability', 0):.1%}\n"
                xai_summary += f"Potential Improvement: +{gap.get('total_potential_improvement', 0)*100:.1f}%\n\n"
                xai_summary += "**Top Missing Skills:**\n"
                for i, skill in enumerate(gap.get('top_missing_skills', [])[:5], 1):
                    priority = "🔴" if skill['priority'] == 'High' else "🟡" if skill['priority'] == 'Medium' else "🟢"
                    xai_summary += f"   {i}. {priority} **{skill['skill']}**: +{skill['impact_percentage']:.2f}% impact\n"
            
            xai_summary += "\n═══════════════════════════════════════════════════════════════\n"
        
        # Build prompt
        prompt = f"""
You are a senior HR consultant and career coach. Provide a comprehensive performance review and development plan.

**RESUME ANALYSIS DATA:**

{xai_summary}

**ATS SCORING:**
- ATS Match Score: {ats_results.get('ats_score', 0):.1f}%
- Skills Match: {ats_results.get('match_percentage', 0):.1f}%
- Matched Skills ({len(ats_results.get('matched_skills', []))}): {', '.join(ats_results.get('matched_skills', [])[:10])}
- Missing Skills ({len(ats_results.get('missing_skills', []))}): {', '.join(ats_results.get('missing_skills', [])[:10])}

**RESUME SECTIONS:**
{json.dumps({k: v[:200] + '...' if len(v) > 200 else v for k, v in resume_sections.items()}, indent=2)}

**TARGET JOB DESCRIPTION:**
{job_description[:500]}...

**YOUR TASK:**
Generate a comprehensive performance review with these sections:

1. **Executive Summary**: Brief overview of the candidate's fit for the role

2. **XAI Analysis Interpretation**: Explain what the AI predictions mean in plain language

3. **Strengths**: Key strengths based on SHAP positive features and matched skills

4. **Skill Gaps**: Critical missing skills with priority (High/Medium/Low) from SHAP analysis

5. **ATS Optimization Tips**: Specific recommendations to improve ATS score

6. **Development Plan**: Concrete action items:
   - Top 3 skills to learn (with course recommendations)
   - Projects to build
   - Timeline (4-12 weeks per skill)

7. **Alternative Roles**: 3-4 alternative job roles based on current skills

Format professionally with clear headings, bullet points, and actionable advice.
"""
        
        return prompt


# ============================================================================
# UNIFIED PIPELINE
# ============================================================================
class UnifiedATSPipeline:
    """Complete end-to-end pipeline"""
    
    def __init__(self):
        print("🔄 Initializing Unified ATS Pipeline...")
        
        print("   📄 Loading CV Section Extractor (LayoutLMv3)...")
        self.cv_extractor = CVSectionExtractor()
        
        print("   📊 Loading ATS Scorer...")
        self.ats_scorer = ATSScorer()
        
        print("   🔍 Loading XAI Analyzer (SHAP/LIME)...")
        self.xai_analyzer = XAIAnalyzer()
        
        print("   🤖 Loading AI Feedback Generator (Groq)...")
        self.feedback_generator = AIFeedbackGenerator()
        
        print("✅ Pipeline initialized successfully!")
    
    def analyze_resume(
        self,
        resume_pdf_path: str,
        job_description: str,
        target_role: str = None
    ) -> Dict[str, Any]:
        """Run complete pipeline analysis"""
        
        results = {}
        
        # Stage 1: Extract CV sections
        print("\n📄 Stage 1: Extracting CV sections...")
        resume_sections = self.cv_extractor.extract_sections(resume_pdf_path)
        results["resume_sections"] = resume_sections
        print(f"   ✅ Extracted {len(resume_sections)} sections")
        
        # Stage 2: ATS scoring and skills extraction
        print("\n📊 Stage 2: Calculating ATS score and extracting skills...")
        ats_results = self.ats_scorer.analyze_resume(resume_pdf_path, job_description)
        results["ats_results"] = ats_results
        print(f"   ✅ ATS Score: {ats_results.get('ats_score', 0):.1f}%")
        print(f"   ✅ Skills Match: {ats_results.get('match_percentage', 0):.1f}%")
        
        # Stage 3: XAI analysis
        print("\n🔍 Stage 3: Running XAI analysis (SHAP/LIME)...")
        resume_text = ats_results.get("resume_text", "")
        
        # First get prediction to determine target role
        xai_results = self.xai_analyzer.analyze(resume_text, target_role)
        results["xai_results"] = xai_results
        
        # If no missing skills analysis or it's empty, use the predicted role
        if xai_results.get("prediction") and (not xai_results.get("missing_skills") or not xai_results["missing_skills"].get("gap_analysis")):
            predicted_role = xai_results["prediction"].get("predicted_role")
            if predicted_role and not target_role:
                print(f"   🔄 Running missing skills analysis for predicted role: {predicted_role}")
                # Re-run with predicted role
                xai_results = self.xai_analyzer.analyze(resume_text, predicted_role)
                results["xai_results"] = xai_results
        
        if xai_results.get("prediction"):
            pred = xai_results["prediction"]
            print(f"   ✅ Predicted Role: {pred.get('predicted_role', 'N/A')} ({pred.get('confidence', 0):.1%})")
        
        # Stage 4: Generate AI feedback
        print("\n🤖 Stage 4: Generating AI-powered feedback...")
        feedback = self.feedback_generator.generate_feedback(
            resume_sections,
            ats_results,
            xai_results,
            job_description
        )
        results["ai_feedback"] = feedback
        print("   ✅ Feedback generated")
        
        return results


# ============================================================================
# STREAMLIT UI
# ============================================================================
def main():
    st.set_page_config(
        page_title="🚀 Unified ATS Pipeline",
        page_icon="🚀",
        layout="wide"
    )
    
    st.title("🚀 Unified ATS Pipeline - Complete Resume Analysis")
    st.markdown("""
    **Complete End-to-End Pipeline:**
    1. 📄 **CV Section Extraction** (LayoutLMv3) - Extract structured sections
    2. 📊 **ATS Scoring** - Skills matching and compatibility score
    3. 🔍 **XAI Analysis** - Job prediction with SHAP/LIME explanations
    4. 🤖 **AI Feedback** - Personalized career guidance (Groq LLM)
    """)
    
    # Initialize pipeline
    if 'pipeline' not in st.session_state:
        with st.spinner("🔄 Initializing pipeline..."):
            try:
                st.session_state.pipeline = UnifiedATSPipeline()
                st.success("✅ Pipeline ready!")
            except Exception as e:
                st.error(f"❌ Failed to initialize pipeline: {e}")
                st.stop()
    
    # Input section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Upload Resume")
        resume_file = st.file_uploader("Upload PDF Resume", type=['pdf'], key="resume")
        
        target_role = st.text_input(
            "🎯 Target Job Role (optional)",
            placeholder="e.g., Data Scientist, Software Engineer",
            help="Leave empty to use AI-predicted role"
        )
    
    with col2:
        st.subheader("📝 Job Description")
        job_description = st.text_area(
            "Paste Job Description",
            height=200,
            placeholder="Paste the full job description here..."
        )
    
    # Analyze button
    if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):
        if not resume_file:
            st.error("❌ Please upload a resume PDF")
            return
        
        if not job_description.strip():
            st.error("❌ Please provide a job description")
            return
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(resume_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Run pipeline
            with st.spinner("🔄 Running complete analysis..."):
                results = st.session_state.pipeline.analyze_resume(
                    tmp_path,
                    job_description,
                    target_role if target_role.strip() else None
                )
            
            # Display results
            st.success("✅ Analysis Complete!")
            
            # Tabs for results
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📄 Extracted Sections",
                "📊 ATS Score",
                "🔍 XAI Analysis",
                "🤖 AI Feedback",
                "📈 Full Report"
            ])
            
            with tab1:
                st.subheader("📄 Extracted Resume Sections")
                sections = results.get("resume_sections", {})
                for section_name, content in sections.items():
                    with st.expander(f"**{section_name.upper()}**"):
                        st.write(content)
            
            with tab2:
                st.subheader("📊 ATS Scoring Results")
                ats = results.get("ats_results", {})
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("ATS Score", f"{ats.get('ats_score', 0):.1f}%")
                with col2:
                    st.metric("Skills Match", f"{ats.get('match_percentage', 0):.1f}%")
                with col3:
                    matched = len(ats.get('matched_skills', []))
                    total = len(ats.get('jd_skills', []))
                    st.metric("Matched Skills", f"{matched}/{total}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**✅ Matched Skills:**")
                    for skill in ats.get('matched_skills', []):
                        st.success(f"• {skill}")
                
                with col2:
                    st.markdown("**❌ Missing Skills:**")
                    for skill in ats.get('missing_skills', [])[:10]:
                        st.error(f"• {skill}")
            
            with tab3:
                st.subheader("🔍 XAI Analysis (Explainable AI)")
                xai = results.get("xai_results", {})
                
                if xai.get("prediction"):
                    pred = xai["prediction"]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Role", pred.get('predicted_role', 'N/A'))
                    with col2:
                        st.metric("Confidence", f"{pred.get('confidence', 0):.1%}")
                    
                    st.markdown("**📈 Top 5 Predicted Roles:**")
                    for i, p in enumerate(pred.get('top_predictions', [])[:5], 1):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.write(f"{i}. **{p['role']}**")
                        with col2:
                            st.progress(p['probability'])
                            st.write(f"{p['probability']:.1%}")
                    
                    # SHAP Analysis
                    if xai.get("shap") and xai["shap"].get('top_features'):
                        st.markdown("---")
                        st.markdown("**🔍 SHAP Feature Importance:**")
                        
                        import plotly.graph_objects as go
                        import pandas as pd
                        
                        features = xai["shap"]['top_features'][:10]
                        df = pd.DataFrame(features)
                        df['shap_scaled'] = df['shap_value'] * 100
                        
                        colors = ['#2ecc71' if imp == 'positive' else '#e74c3c' for imp in df['impact']]
                        
                        fig = go.Figure(data=[
                            go.Bar(
                                x=df['shap_scaled'],
                                y=df['feature'],
                                orientation='h',
                                marker=dict(color=colors),
                                text=[f"{val:+.2f}pp" for val in df['shap_scaled']],
                                textposition='outside'
                            )
                        ])
                        
                        fig.update_layout(
                            title="SHAP Feature Contributions",
                            xaxis_title="Contribution (percentage points)",
                            yaxis_title="Feature",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Missing Skills
                    if xai.get("missing_skills") and xai["missing_skills"].get("gap_analysis"):
                        st.markdown("---")
                        st.markdown("**⚠️ Missing Skills Analysis (SHAP-Based):**")
                        
                        gap = xai["missing_skills"]["gap_analysis"]
                        st.info(f"Potential Improvement: +{gap.get('total_potential_improvement', 0)*100:.1f}%")
                        
                        for i, skill in enumerate(gap.get('top_missing_skills', [])[:8], 1):
                            priority_emoji = "🔴" if skill['priority'] == 'High' else "🟡" if skill['priority'] == 'Medium' else "🟢"
                            st.warning(f"{i}. {priority_emoji} **{skill['skill']}** - Impact: +{skill['impact_percentage']:.2f}% ({skill['priority']} Priority)")
            
            with tab4:
                st.subheader("🤖 AI-Powered Feedback")
                feedback = results.get("ai_feedback", "")
                st.markdown(feedback)
            
            with tab5:
                st.subheader("📈 Complete Analysis Report")
                st.json(results, expanded=False)
                
                # Download button
                report_json = json.dumps(results, indent=2, default=str)
                st.download_button(
                    label="📥 Download Full Report (JSON)",
                    data=report_json,
                    file_name="ats_analysis_report.json",
                    mime="application/json"
                )
        
        except Exception as e:
            st.error(f"❌ Error during analysis: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            # Cleanup
            try:
                os.unlink(tmp_path)
            except:
                pass


if __name__ == "__main__":
    main()
