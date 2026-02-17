"""
Extended ATS Service with Explainable AI Support
Adds SHAP and LIME explanations to the existing ATS pipeline
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class XAIFeatureExplanation(BaseModel):
    """Explanation for a single feature"""
    feature: str = Field(..., description="Feature name or keyword")
    value: float = Field(..., description="Feature value/weight")
    shap_value: Optional[float] = Field(None, description="SHAP contribution")
    lime_weight: Optional[float] = Field(None, description="LIME weight")
    impact: str = Field(..., description="positive or negative")
    rationale: Optional[str] = Field(None, description="Human-readable explanation")


class XAIPredictionExplanation(BaseModel):
    """Complete XAI explanation for a prediction"""
    method: str = Field(..., description="SHAP, LIME, or BOTH")
    predicted_class: str = Field(..., description="Predicted job role")
    confidence: float = Field(..., description="Prediction confidence")
    shap_features: Optional[List[XAIFeatureExplanation]] = Field(None, description="SHAP features")
    lime_features: Optional[List[XAIFeatureExplanation]] = Field(None, description="LIME features")
    summary: str = Field(..., description="Human-readable explanation summary")
    comparison: Optional[str] = Field(None, description="SHAP vs LIME comparison")


class EnhancedAnalysisResponse(BaseModel):
    """Enhanced analysis response with XAI explanations"""
    success: bool
    overall_match: Optional[float] = None
    match_level: Optional[str] = None
    skills_match_rate: Optional[float] = None
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    
    # Job prediction
    job_prediction: Optional[Dict[str, Any]] = None
    
    # XAI explanations
    xai_explanation: Optional[XAIPredictionExplanation] = None
    
    # Original explanations
    explanations: Optional[Dict[str, Any]] = None
    
    # Raw results
    raw: Optional[Dict[str, Any]] = None


class EnhancedAtsService:
    """
    Enhanced ATS service with explainable AI capabilities.
    Wraps the original ATS service and adds SHAP/LIME explanations.
    """
    
    def __init__(self, ats_agent_path: Path):
        """Initialize enhanced ATS service with XAI capabilities"""
        self.ats_agent_path = ats_agent_path
        
        # Add to path if needed
        if str(ats_agent_path) not in sys.path:
            sys.path.insert(0, str(ats_agent_path))
        
        # Import XAI explainer
        try:
            from xai_explainer import XAIExplainer
            self.xai_available = True
            self.xai_explainer = XAIExplainer(
                model_path=str(ats_agent_path / "JobPrediction_Model")
            )
            print("✅ XAI explainer loaded successfully")
        except Exception as e:
            print(f"⚠️ XAI explainer not available: {e}")
            self.xai_available = False
            self.xai_explainer = None
        
        # Import original ATS pipeline
        try:
            from ats_pipeline import ATSPipeline
            from job_role_predictor import JobRolePredictor
            
            self.pipeline = ATSPipeline(use_spacy=True)
            self.job_predictor = JobRolePredictor(
                model_path=str(ats_agent_path / "JobPrediction_Model")
            )
            print("✅ ATS pipeline loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load ATS pipeline: {e}")
    
    async def analyze_with_explanations(
        self,
        resume_bytes: bytes,
        job_description: str,
        include_shap: bool = True,
        include_lime: bool = True,
        compare_methods: bool = False
    ) -> EnhancedAnalysisResponse:
        """
        Analyze resume with XAI explanations.
        
        Args:
            resume_bytes: PDF resume as bytes
            job_description: Job description text
            include_shap: Include SHAP explanations
            include_lime: Include LIME explanations
            compare_methods: Compare SHAP and LIME methods
            
        Returns:
            Enhanced analysis response with XAI explanations
        """
        return await asyncio.to_thread(
            self._analyze_sync,
            resume_bytes,
            job_description,
            include_shap,
            include_lime,
            compare_methods
        )
    
    def _analyze_sync(
        self,
        resume_bytes: bytes,
        job_description: str,
        include_shap: bool,
        include_lime: bool,
        compare_methods: bool
    ) -> EnhancedAnalysisResponse:
        """Synchronous analysis with XAI explanations"""
        import tempfile
        from pathlib import Path
        
        # Save resume to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(resume_bytes)
            tmp_path = Path(tmp.name)
        
        try:
            # Run standard ATS analysis
            results = self.pipeline.analyze(
                str(tmp_path),
                job_description,
                verbose=False,
                analyze_format=False
            )
        finally:
            tmp_path.unlink(missing_ok=True)
        
        # Extract results
        resume_keywords = results.get("resume_analysis", {}).get("keywords", {})
        keyword_list = (
            resume_keywords.get("technical_skills")
            or resume_keywords.get("all_keywords")
            or []
        )
        skills_text = " ".join(keyword_list[:120])
        
        similarity_scores = results.get("similarity_scores", {}) or {}
        detailed_scores = similarity_scores.get("detailed_scores", {}) or {}
        matched_skills = similarity_scores.get("matched_skills") or []
        missing_skills = similarity_scores.get("missing_skills") or []
        
        # Job prediction with LSTM
        job_prediction = None
        if skills_text:
            prediction = self.job_predictor.predict_job_role(skills_text, top_n=5)
            job_prediction = {
                'predicted_role': prediction.get('predicted_role'),
                'confidence': prediction.get('confidence'),
                'top_predictions': [
                    {'role': role, 'probability': prob}
                    for role, prob in prediction.get('top_predictions', [])
                ]
            }
        
        # XAI Explanations
        xai_explanation = None
        if self.xai_available and self.xai_explainer and skills_text:
            try:
                xai_explanation = self._generate_xai_explanation(
                    skills_text,
                    include_shap,
                    include_lime,
                    compare_methods
                )
            except Exception as e:
                print(f"⚠️ XAI explanation failed: {e}")
        
        # Build response
        response = EnhancedAnalysisResponse(
            success=bool(results.get('success', True)),
            overall_match=similarity_scores.get('overall_percentage'),
            match_level=similarity_scores.get('match_level'),
            skills_match_rate=detailed_scores.get('skills_match_rate'),
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            job_prediction=job_prediction,
            xai_explanation=xai_explanation,
            raw=results
        )
        
        return response
    
    def _generate_xai_explanation(
        self,
        skills_text: str,
        include_shap: bool,
        include_lime: bool,
        compare_methods: bool
    ) -> XAIPredictionExplanation:
        """Generate XAI explanation using SHAP and/or LIME"""
        
        if not self.xai_explainer.xgb_model:
            # XGBoost model not trained yet
            return None
        
        # Get prediction from XGBoost
        prediction = self.xai_explainer.predict_with_xgboost(skills_text)
        
        shap_features = None
        lime_features = None
        comparison = None
        
        # Generate SHAP explanation
        if include_shap:
            try:
                shap_result = self.xai_explainer.explain_with_shap(skills_text)
                shap_features = [
                    XAIFeatureExplanation(
                        feature=feat['feature'],
                        value=feat.get('value', 0),
                        shap_value=feat['shap_value'],
                        impact=feat['impact'],
                        rationale=f"SHAP value of {feat['shap_value']:.4f} indicates {'strong' if abs(feat['shap_value']) > 0.1 else 'moderate'} {'positive' if feat['impact'] == 'positive' else 'negative'} contribution"
                    )
                    for feat in shap_result['top_features'][:15]
                ]
            except Exception as e:
                print(f"⚠️ SHAP generation failed: {e}")
        
        # Generate LIME explanation
        if include_lime:
            try:
                lime_result = self.xai_explainer.explain_with_lime(skills_text)
                lime_features = [
                    XAIFeatureExplanation(
                        feature=feat['feature'],
                        value=0,
                        lime_weight=feat['weight'],
                        impact=feat['impact'],
                        rationale=f"LIME weight of {feat['weight']:.4f} indicates {'strong' if abs(feat['weight']) > 0.1 else 'moderate'} {'positive' if feat['impact'] == 'positive' else 'negative'} influence"
                    )
                    for feat in lime_result['features'][:15]
                ]
            except Exception as e:
                print(f"⚠️ LIME generation failed: {e}")
        
        # Compare methods if requested
        if compare_methods and shap_features and lime_features:
            shap_feature_names = set(f.feature for f in shap_features)
            lime_feature_names = set(f.feature.split()[0] for f in lime_features)
            overlap = shap_feature_names & lime_feature_names
            comparison = f"SHAP and LIME agree on {len(overlap)} features. " + (
                "High agreement indicates robust explanation." if len(overlap) > 5
                else "Different perspectives provide complementary insights."
            )
        
        # Build summary
        method = "BOTH" if (shap_features and lime_features) else ("SHAP" if shap_features else "LIME")
        summary_parts = [f"Predicted role: {prediction['predicted_role']} (confidence: {prediction['confidence']:.2%})"]
        
        if shap_features:
            top_shap = shap_features[0]
            summary_parts.append(f"Top SHAP feature: '{top_shap.feature}' ({top_shap.impact})")
        
        if lime_features:
            top_lime = lime_features[0]
            summary_parts.append(f"Top LIME feature: '{top_lime.feature}' ({top_lime.impact})")
        
        summary = ". ".join(summary_parts)
        
        return XAIPredictionExplanation(
            method=method,
            predicted_class=prediction['predicted_role'],
            confidence=prediction['confidence'],
            shap_features=shap_features,
            lime_features=lime_features,
            summary=summary,
            comparison=comparison
        )


async def main():
    """Test the enhanced ATS service"""
    print("="*80)
    print("ENHANCED ATS SERVICE WITH XAI - TEST")
    print("="*80)
    
    from pathlib import Path
    
    # Initialize service
    ats_path = Path(__file__).parent
    service = EnhancedAtsService(ats_path)
    
    print("\n⚠️ To test, provide resume PDF bytes and job description")
    print("   Example usage is shown in the FastAPI integration")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
