"""
FastAPI wrapper for unified ATS pipeline with XAI and LayoutLMv3 section extraction
Provides REST API endpoint for Node.js server
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# LayoutLMv3 imports for section extraction
try:
    from PIL import Image
    import torch
    import pytesseract
    from pdf2image import convert_from_path
    from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
    from pytesseract import Output
    LAYOUTLM_AVAILABLE = True
    print("✅ LayoutLMv3 dependencies loaded")
except ImportError as e:
    LAYOUTLM_AVAILABLE = False
    print(f"⚠️ LayoutLMv3 not available: {e}")

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent / "ATS-agent" / "ATS-agent"))
sys.path.insert(0, str(Path(__file__).parent / "deep_Learning_Project"))

# Import pipeline components
from xai_explainer import XAIExplainer

# Import similarity calculator and keyword extractor
try:
    from similarity_calculator import SimilarityCalculator
    from keyword_extractor import KeywordExtractor
    
    similarity_calc = SimilarityCalculator()
    keyword_extractor = KeywordExtractor(use_spacy=False)  # Use keyword DB only, no spaCy for speed
    
    def calculate_similarity_score(cv_text, job_description):
        """Calculate similarity using proper technical skills extraction"""
        try:
            # Extract technical skills properly using KeywordExtractor
            resume_skills = keyword_extractor.extract_technical_skills(cv_text)
            job_skills = keyword_extractor.extract_technical_skills(job_description)
            
            print(f"   📊 CV Skills: {len(resume_skills)} detected")
            print(f"   📊 Job Skills: {len(job_skills)} required")
            
            # Calculate matched and missing skills
            matched_skills = list(resume_skills.intersection(job_skills))
            missing_skills = list(job_skills - resume_skills)
            
            # Get cosine similarity on full text
            cosine_score = similarity_calc.cosine_similarity_score(cv_text, job_description)
            
            # Calculate skills match rate
            skills_match_rate = len(matched_skills) / len(job_skills) if len(job_skills) > 0 else 0.0
            
            # Calculate overall score (weighted: 50% skills, 50% text similarity)
            overall = (skills_match_rate * 0.5 + cosine_score * 0.5) * 100
            
            # Determine match level
            if overall >= 75:
                match_level = "Excellent"
            elif overall >= 60:
                match_level = "Good"
            elif overall >= 40:
                match_level = "Medium"
            else:
                match_level = "Low"
            
            print(f"   ✅ Matched Skills: {len(matched_skills)}/{len(job_skills)}")
            print(f"   ⚠️ Missing Skills: {len(missing_skills)}")
            
            return {
                "overall_percentage": round(overall, 2),
                "match_level": match_level,
                "detailed_scores": {
                    "cosine_similarity": round(cosine_score * 100, 2),
                    "skills_match_rate": round(skills_match_rate * 100, 2),
                    "matched_count": len(matched_skills),
                    "required_count": len(job_skills),
                    "coverage_percentage": round(skills_match_rate * 100, 2)
                },
                "matched_skills": sorted(matched_skills)[:50],  # Return up to 50 skills
                "missing_skills": sorted(missing_skills)[:50]    # Return up to 50 missing skills
            }
        except Exception as e:
            print(f"❌ Similarity calculation error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "overall_percentage": 0.0,
                "match_level": "Error",
                "detailed_scores": {},
                "matched_skills": [],
                "missing_skills": []
            }
    
    print("✅ Loaded SimilarityCalculator and KeywordExtractor from ATS-agent")
except ImportError as e:
    print(f"⚠️ Could not import required modules: {e}")
    # Fallback similarity calculator
    def calculate_similarity_score(cv_text, job_description):
        return {
            "overall_percentage": 50.0,
            "match_level": "Medium",
            "detailed_scores": {},
            "matched_skills": [],
            "missing_skills": []
        }

# ============================================================================
# Configuration
# ============================================================================
MODEL_PATH = Path(__file__).parent / "deep_Learning_Project" / "JobPrediction_Model"
LAYOUTLM_MODEL_DIR = Path(__file__).parent / "pi5eme1 - Copie (2)" / "pi5eme1 - Copie (2)" / "outputs" / "models" / "layoutlmv3_finetuned"
POPPLER_PATH = r"C:\Release-25.07.0-0\poppler-25.07.0\Library\bin"

# Configure Tesseract
if LAYOUTLM_AVAILABLE:
    try:
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    except:
        pass

# Initialize LayoutLMv3 model if available
layoutlm_processor = None
layoutlm_model = None
DEVICE = None

if LAYOUTLM_AVAILABLE and LAYOUTLM_MODEL_DIR.exists():
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layoutlm_processor = LayoutLMv3Processor.from_pretrained(str(LAYOUTLM_MODEL_DIR))
        layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained(str(LAYOUTLM_MODEL_DIR)).to(DEVICE)
        print(f"✅ LayoutLMv3 model loaded on {DEVICE}")
    except Exception as e:
        print(f"⚠️ Could not load LayoutLMv3 model: {e}")
        LAYOUTLM_AVAILABLE = False

# Label mapping for LayoutLMv3
label_list = [
    "O",
    "B-HEADER", "I-HEADER",
    "B-CONTACT", "I-CONTACT",
    "B-SUMMARY", "I-SUMMARY",
    "B-EDUCATION", "I-EDUCATION",
    "B-EXPERIENCE", "I-EXPERIENCE",
    "B-SKILLS", "I-SKILLS",
    "B-PROJECTS", "I-PROJECTS",
    "B-CERTIFICATIONS", "I-CERTIFICATIONS",
    "B-LANGUAGES", "I-LANGUAGES",
    "B-PUBLICATIONS", "I-PUBLICATIONS",
    "B-REFERENCES", "I-REFERENCES",
    "B-OTHER", "I-OTHER"
]
id2label = {i: l for i, l in enumerate(label_list)}

# ============================================================================
# Helper Functions
# ============================================================================

def make_json_serializable(obj):
    """Convert complex objects to JSON-serializable format"""
    import numpy as np
    
    # Handle None
    if obj is None:
        return None
    
    # Handle basic types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Handle numpy types
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    
    # Handle collections
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle LIME/SHAP Explanation objects
    if hasattr(obj, 'as_list'):
        # LIME explanation
        try:
            return {
                'explanation': obj.as_list(),
                'score': float(obj.score) if hasattr(obj, 'score') else None,
                'intercept': float(obj.intercept[0]) if hasattr(obj, 'intercept') else None
            }
        except:
            pass
    
    # Try to extract dict
    if hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    
    # Last resort: convert to string
    try:
        return str(obj)
    except:
        return None


# ============================================================================
# Section Extraction Functions
# ============================================================================

def extract_words_and_boxes(image: Image.Image):
    """Use OCR to extract words and their normalized coordinates."""
    width, height = image.size
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    words, boxes = [], []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        box = [
            int(1000 * x / width),
            int(1000 * y / height),
            int(1000 * (x + w) / width),
            int(1000 * (y + h) / height),
        ]
        words.append(text)
        boxes.append(box)
    return words, boxes


def infer_sections_layoutlm(image_path):
    """Detect sections from an image using LayoutLMv3 and OCR."""
    if not LAYOUTLM_AVAILABLE or layoutlm_model is None:
        return {}
    
    try:
        image = Image.open(image_path).convert("RGB")
        words, boxes = extract_words_and_boxes(image)
        
        print(f"🔍 LayoutLMv3: {len(words)} words detected by OCR")
        if not words:
            print("⚠️ No words detected in image")
            return {}

        encoding = layoutlm_processor(
            images=image,
            text=words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512
        ).to(DEVICE)

        with torch.no_grad():
            outputs = layoutlm_model(**encoding)
            predictions = outputs.logits.argmax(-1).squeeze().tolist()
            predicted_labels = [id2label[p] for p in predictions]

        print(f"✅ LayoutLMv3: Generated {len(predicted_labels)} predictions")

        sections = {}
        for word, label in zip(words, predicted_labels):
            if label != "O":
                label_clean = label.replace("B-", "").replace("I-", "")
                sections.setdefault(label_clean, []).append(word)

        # Merge words into full sections
        sections = {k: " ".join(v) for k, v in sections.items()}
        print(f"✅ LayoutLMv3: Detected sections: {list(sections.keys())}")
        return sections
    except Exception as e:
        print(f"❌ LayoutLMv3 error: {e}")
        import traceback
        traceback.print_exc()
        return {}


def convert_pdf_to_images(pdf_path):
    """Convert a PDF into PNG images (one per page)."""
    try:
        poppler = POPPLER_PATH if os.path.exists(POPPLER_PATH) else None
        pages = convert_from_path(pdf_path, dpi=200, poppler_path=poppler)
        paths = []
        for i, p in enumerate(pages):
            tmp = f"temp_page_{i+1}.png"
            p.save(tmp, "PNG")
            paths.append(tmp)
        return paths
    except Exception as e:
        print(f"❌ PDF to image conversion failed: {e}")
        return []


def extract_sections_from_resume(resume_path: str, cv_text: str = None) -> dict:
    """
    Extract structured sections from resume using LayoutLMv3 (preferred) or regex fallback
    """
    print("📚 Extracting structured sections from resume...")
    
    # Try LayoutLMv3 first if available
    if LAYOUTLM_AVAILABLE and layoutlm_model is not None:
        try:
            print("🔍 Using LayoutLMv3 transformer model...")
            image_paths = convert_pdf_to_images(resume_path)
            
            if not image_paths:
                print("⚠️ Could not convert PDF to images, falling back to regex")
            else:
                all_sections = {}
                for img_path in image_paths:
                    try:
                        page_sections = infer_sections_layoutlm(img_path)
                        # Merge sections from all pages
                        for section_name, content in page_sections.items():
                            if section_name in all_sections:
                                all_sections[section_name] += " " + content
                            else:
                                all_sections[section_name] = content
                    except Exception as e:
                        print(f"⚠️ Error processing page: {e}")
                    finally:
                        # Cleanup temp image
                        if os.path.exists(img_path) and img_path.startswith("temp_page_"):
                            try:
                                os.remove(img_path)
                            except:
                                pass
                
                if all_sections:
                    print(f"✅ LayoutLMv3 extracted {len(all_sections)} sections")
                    return {"sections": all_sections, "method": "LayoutLMv3"}
                else:
                    print("⚠️ LayoutLMv3 found no sections, falling back to regex")
        except Exception as e:
            print(f"❌ LayoutLMv3 extraction failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback to simple text extraction
    print("⚠️ Using basic text extraction (no section detection)")
    return {"sections": {}, "method": "none"}


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ATS XAI Service",
    description="Explainable AI-powered ATS analysis with SHAP, LIME, and LayoutLMv3 section extraction",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize XAI explainer
xai_explainer = XAIExplainer(model_path=str(MODEL_PATH))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "xgboost_loaded": xai_explainer.xgb_model is not None,
        "job_roles_count": len(xai_explainer.job_roles),
        "layoutlmv3_available": LAYOUTLM_AVAILABLE and layoutlm_model is not None
    }


@app.post("/analyze")
async def analyze_application(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    target_role: Optional[str] = Form(None)
):
    """
    Complete ATS analysis pipeline with:
    - LayoutLMv3 section extraction
    - ATS similarity scoring
    - Job prediction with confidence
    - SHAP explanation
    - LIME explanation
    - Missing skills analysis
    """
    
    # Validate file type
    if not resume.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await resume.read()
        tmp_file.write(content)
        resume_path = tmp_file.name
    
    try:
        # STAGE 1: Extract CV text
        print("📄 Extracting text from PDF...")
        try:
            from pdf_extractor import extract_text_from_pdf
            cv_text = extract_text_from_pdf(resume_path)
        except ImportError:
            # Fallback: Use PyPDF2 directly
            import PyPDF2
            cv_text = ""
            with open(resume_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    cv_text += page.extract_text()
        
        if not cv_text or len(cv_text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Could not extract text from resume")
        
        print(f"✅ Text extracted: {len(cv_text)} characters")
        
        # STAGE 1.5: Extract structured sections using LayoutLMv3
        cv_sections_result = extract_sections_from_resume(resume_path, cv_text)
        cv_sections = cv_sections_result.get("sections", {})
        
        # STAGE 2: Calculate ATS score
        print(f"📊 Calculating similarity score...")
        similarity_scores = calculate_similarity_score(cv_text, job_description)
        print(f"✅ ATS Score: {similarity_scores['overall_percentage']}%")
        
        # STAGE 3: Job prediction with XAI
        print("🎯 Predicting job roles...")
        prediction = xai_explainer.predict_with_xgboost(cv_text, top_n=5)
        print(f"✅ Predicted: {prediction['predicted_role']}")
        
        # STAGE 4: SHAP explanation
        print("🔍 Generating SHAP explanation...")
        shap_explanation = xai_explainer.explain_with_shap(cv_text)
        print("✅ SHAP completed")
        
        # STAGE 5: LIME explanation (reduced samples for speed)
        print("🔍 Generating LIME explanation...")
        lime_explanation = xai_explainer.explain_with_lime(cv_text, num_features=10, num_samples=1000)
        print("✅ LIME completed")
        
        # STAGE 6: Missing skills analysis
        print("🔍 Analyzing missing skills...")
        analysis_role = target_role if target_role else prediction['predicted_role']
        missing_skills_analysis = xai_explainer.analyze_missing_skills(
            cv_text=cv_text,
            target_role=analysis_role,
            top_n=10
        )
        print("✅ Missing skills analysis completed")
        
        # Convert explanations to JSON-serializable format
        shap_serializable = make_json_serializable(shap_explanation)
        lime_serializable = make_json_serializable(lime_explanation)
        
        # Format response
        response = {
            "overall_match": similarity_scores["overall_percentage"],
            "match_level": similarity_scores["match_level"],
            "detailed_scores": similarity_scores["detailed_scores"],
            "matched_skills": similarity_scores["matched_skills"],
            "missing_skills": similarity_scores["missing_skills"],
            "job_prediction": {
                "predicted_role": prediction["predicted_role"],
                "confidence": prediction["confidence"],
                "top_predictions": prediction["top_predictions"]
            },
            "shap_explanation": shap_serializable,
            "lime_explanation": lime_serializable,
            "missing_skills_analysis": missing_skills_analysis,
            "cv_sections": cv_sections
        }
        
        print("✅ Analysis completed successfully")
        return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Error in analyze_application: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup
        try:
            os.unlink(resume_path)
        except:
            pass


@app.get("/roles")
async def get_available_roles():
    """Get list of all available job roles"""
    return {
        "roles": xai_explainer.job_roles,
        "count": len(xai_explainer.job_roles)
    }


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("="*80)
    print("🚀 Starting ATS XAI API Service with LayoutLMv3")
    print("="*80)
    print(f"Model Path: {MODEL_PATH}")
    print(f"XGBoost Loaded: {xai_explainer.xgb_model is not None}")
    print(f"Job Roles: {len(xai_explainer.job_roles)}")
    print(f"LayoutLMv3: {'✅ Loaded' if (LAYOUTLM_AVAILABLE and layoutlm_model) else '❌ Not available'}")
    print("="*80)
    
    try:
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
