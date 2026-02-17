"""
Extended API routes for Explainable AI features
Adds SHAP and LIME explanation endpoints to the ATS service
"""

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .models import AnalysisResponse
from .pipeline import AtsService


class XAIExplanationRequest(BaseModel):
    """Request for XAI explanation"""
    include_shap: bool = Field(True, description="Include SHAP explanations")
    include_lime: bool = Field(True, description="Include LIME explanations")
    compare_methods: bool = Field(False, description="Compare SHAP and LIME")


class XAIFeatureExplanation(BaseModel):
    """Feature explanation from XAI"""
    feature: str
    value: float
    shap_value: Optional[float] = None
    lime_weight: Optional[float] = None
    impact: str
    rationale: Optional[str] = None


class XAIPredictionExplanation(BaseModel):
    """Complete XAI prediction explanation"""
    method: str
    predicted_class: str
    confidence: float
    shap_features: Optional[list[XAIFeatureExplanation]] = None
    lime_features: Optional[list[XAIFeatureExplanation]] = None
    summary: str
    comparison: Optional[str] = None


class EnhancedAnalysisResponse(AnalysisResponse):
    """Analysis response with XAI explanations"""
    xai_explanation: Optional[XAIPredictionExplanation] = None


router = APIRouter(prefix="/api/v1/xai", tags=["Explainable AI"])


def get_xai_service():
    """Get or create XAI-enhanced ATS service"""
    # This should be initialized once at startup
    # For now, we'll import dynamically
    try:
        from pathlib import Path
        import sys
        
        # Add ATS-agent to path
        ats_path = Path(__file__).parent.parent.parent / "ATS-agent" / "ATS-agent"
        if str(ats_path) not in sys.path:
            sys.path.insert(0, str(ats_path))
        
        from enhanced_ats_service import EnhancedAtsService
        return EnhancedAtsService(ats_path)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize XAI service: {str(e)}"
        )


@router.post("/analyze", response_model=EnhancedAnalysisResponse)
async def analyze_with_xai(
    resume: UploadFile = File(..., description="Resume PDF file"),
    job_description: str = Form(..., description="Job description text"),
    include_shap: bool = Form(True, description="Include SHAP explanations"),
    include_lime: bool = Form(True, description="Include LIME explanations"),
    compare_methods: bool = Form(False, description="Compare SHAP and LIME methods")
):
    """
    Analyze resume with explainable AI features.
    
    Provides SHAP and/or LIME explanations for the job role prediction,
    helping understand which skills and features influenced the decision.
    
    Returns:
    - Standard ATS analysis (similarity scores, matched/missing skills)
    - Job role prediction with confidence
    - SHAP feature importance (if include_shap=True)
    - LIME feature weights (if include_lime=True)
    - Method comparison (if compare_methods=True)
    """
    # Validate file type
    if not resume.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Read resume
    resume_bytes = await resume.read()
    if len(resume_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")
    
    # Get XAI service
    try:
        service = get_xai_service()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    # Analyze with explanations
    try:
        result = await service.analyze_with_explanations(
            resume_bytes=resume_bytes,
            job_description=job_description,
            include_shap=include_shap,
            include_lime=include_lime,
            compare_methods=compare_methods
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/explain-prediction")
async def explain_prediction(
    skills_text: str = Form(..., description="Skills text to explain"),
    method: str = Form("both", description="Explanation method: shap, lime, or both")
):
    """
    Get explanation for a job role prediction based on skills.
    
    This endpoint provides detailed feature-level explanations without
    requiring a full resume upload.
    
    Args:
    - skills_text: Text containing skills (comma or space separated)
    - method: Which explanation method to use (shap, lime, or both)
    
    Returns:
    - Predicted job role with confidence
    - Feature-level explanations (SHAP and/or LIME)
    - Human-readable summary
    """
    if method.lower() not in ['shap', 'lime', 'both']:
        raise HTTPException(
            status_code=400,
            detail="Method must be 'shap', 'lime', or 'both'"
        )
    
    # Get XAI service
    try:
        service = get_xai_service()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    if not service.xai_available or not service.xai_explainer:
        raise HTTPException(
            status_code=503,
            detail="XAI service not available. Train XGBoost model first."
        )
    
    try:
        include_shap = method.lower() in ['shap', 'both']
        include_lime = method.lower() in ['lime', 'both']
        
        explanation = service._generate_xai_explanation(
            skills_text=skills_text,
            include_shap=include_shap,
            include_lime=include_lime,
            compare_methods=(method.lower() == 'both')
        )
        
        if explanation is None:
            raise HTTPException(
                status_code=503,
                detail="XGBoost model not trained. Run train_xai_model.py first."
            )
        
        return explanation
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Explanation generation failed: {str(e)}"
        )


@router.get("/model-info")
async def get_model_info():
    """
    Get information about the XAI models.
    
    Returns details about available models, their training status,
    and explainability capabilities.
    """
    try:
        service = get_xai_service()
        
        xai_status = {
            'xai_available': service.xai_available,
            'xgboost_trained': False,
            'job_roles': [],
            'explainability_methods': []
        }
        
        if service.xai_available and service.xai_explainer:
            xai_status['xgboost_trained'] = service.xai_explainer.xgb_model is not None
            xai_status['job_roles'] = service.xai_explainer.job_roles
            
            if service.xai_explainer.xgb_model:
                xai_status['explainability_methods'] = ['SHAP', 'LIME']
        
        return {
            'success': True,
            'ats_available': True,
            'xai_status': xai_status,
            'message': (
                'XAI fully operational' if xai_status['xgboost_trained']
                else 'XAI available but model needs training. Run train_xai_model.py'
            )
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


@router.post("/train-model")
async def trigger_training(
    dataset_path: str = Form(..., description="Path to training dataset CSV")
):
    """
    Trigger XGBoost model training (for development/admin use).
    
    WARNING: This is a long-running operation. In production, use a
    background task queue (Celery, RQ, etc.) instead.
    """
    import asyncio
    from pathlib import Path
    
    if not Path(dataset_path).exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset not found: {dataset_path}"
        )
    
    try:
        # This should be moved to a background task in production
        def train_sync():
            from train_xai_model import train_model
            train_model(dataset_path=dataset_path)
        
        await asyncio.to_thread(train_sync)
        
        return {
            'success': True,
            'message': 'Model training completed successfully'
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}"
        )
