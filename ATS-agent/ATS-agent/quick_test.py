"""
Quick test to see actual predictions and SHAP values
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from xai_explainer import XAIExplainer

# Initialize
explainer = XAIExplainer()

# Test with a clear example
test_cases = [
    "data scientist analytics python sql statistics machine learning pandas",
    "software engineer java python javascript react node backend frontend",
    "devops engineer docker kubernetes aws cloud infrastructure automation",
]

for skills in test_cases:
    print("\n" + "="*80)
    print(f"Testing: {skills}")
    print("="*80)
    
    # Get prediction
    pred = explainer.predict_with_xgboost(skills, top_n=5)
    print(f"\nTop 5 Predictions:")
    for i, p in enumerate(pred['top_predictions'], 1):
        print(f"  {i}. {p['role']:35s} {p['probability']:6.2%}")
    
    # Get SHAP explanation
    try:
        shap_exp = explainer.explain_with_shap(skills)
        print(f"\nSHAP Top 5 Features:")
        for feat in shap_exp['top_features'][:5]:
            print(f"  {feat['feature']:25s} {feat['shap_value']:+8.6f} ({feat['impact']})")
    except Exception as e:
        print(f"SHAP error: {e}")
