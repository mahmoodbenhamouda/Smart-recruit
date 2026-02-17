"""
Demo script for Explainable AI features
Shows how to use SHAP and LIME with example data
"""

import sys
from pathlib import Path

# Add ATS-agent to path
ats_path = Path(__file__).parent / "ATS-agent" / "ATS-agent"
if str(ats_path) not in sys.path:
    sys.path.insert(0, str(ats_path))


def demo_basic_prediction():
    """Demo 1: Basic job prediction with explanations"""
    print("\n" + "="*80)
    print("DEMO 1: BASIC JOB PREDICTION WITH XAI")
    print("="*80)
    
    try:
        from xai_explainer import XAIExplainer
    except ImportError as e:
        print(f"❌ Failed to import XAI module: {e}")
        print("   Make sure you're in the correct directory and have installed dependencies")
        return
    
    # Initialize explainer
    print("\n📚 Initializing XAI Explainer...")
    explainer = XAIExplainer()
    
    if explainer.xgb_model is None:
        print("\n⚠️ XGBoost model not trained yet!")
        print("   Run the following command first:")
        print("   python ATS-agent/ATS-agent/train_xai_model.py")
        return
    
    # Test cases
    test_cases = [
        {
            'name': 'Machine Learning Engineer',
            'skills': 'Python TensorFlow Keras Deep Learning Neural Networks PyTorch Machine Learning'
        },
        {
            'name': 'DevOps Engineer',
            'skills': 'Docker Kubernetes AWS Jenkins CI/CD Terraform Ansible Linux DevOps'
        },
        {
            'name': 'Data Scientist',
            'skills': 'Python R SQL Pandas NumPy Statistics Machine Learning Data Analysis Visualization'
        },
        {
            'name': 'Software Engineer',
            'skills': 'Java Spring Boot Microservices REST API MySQL Git Agile'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}: {test_case['name']}")
        print('='*80)
        print(f"Skills: {test_case['skills']}")
        
        # Make prediction
        prediction = explainer.predict_with_xgboost(test_case['skills'])
        
        print(f"\n🎯 Prediction:")
        print(f"   Role:       {prediction['predicted_role']}")
        print(f"   Confidence: {prediction['confidence']:.1%}")
        
        print(f"\n📊 Top 3 Predictions:")
        for j, pred in enumerate(prediction['top_predictions'], 1):
            print(f"   {j}. {pred['role']:30s} {pred['probability']:.1%}")


def demo_shap_explanation():
    """Demo 2: SHAP explanations"""
    print("\n" + "="*80)
    print("DEMO 2: SHAP FEATURE IMPORTANCE")
    print("="*80)
    
    try:
        from xai_explainer import XAIExplainer
    except ImportError:
        print("❌ XAI module not available")
        return
    
    explainer = XAIExplainer()
    
    if explainer.xgb_model is None:
        print("⚠️ XGBoost model not trained")
        return
    
    # Example skills
    skills = "Python TensorFlow Deep Learning Neural Networks Computer Vision PyTorch"
    
    print(f"\n📝 Input Skills:")
    print(f"   {skills}")
    
    try:
        # Get SHAP explanation
        shap_exp = explainer.explain_with_shap(skills)
        
        print(f"\n🔍 SHAP Analysis:")
        print(f"   Predicted Class: {shap_exp['predicted_class']}")
        print(f"   Confidence: {shap_exp['confidence']:.1%}")
        print(f"\n   Summary: {shap_exp['summary']}")
        
        print(f"\n📊 Top 10 SHAP Features:")
        print(f"   {'Feature':<25} {'SHAP Value':>12} {'Impact':<10}")
        print("   " + "-"*50)
        
        for feat in shap_exp['top_features'][:10]:
            impact_icon = '✅' if feat['impact'] == 'positive' else '❌'
            print(f"   {feat['feature']:<25} {feat['shap_value']:>12.4f} {impact_icon} {feat['impact']:<10}")
        
    except Exception as e:
        print(f"❌ SHAP explanation failed: {e}")


def demo_lime_explanation():
    """Demo 3: LIME explanations"""
    print("\n" + "="*80)
    print("DEMO 3: LIME LOCAL EXPLANATIONS")
    print("="*80)
    
    try:
        from xai_explainer import XAIExplainer
    except ImportError:
        print("❌ XAI module not available")
        return
    
    explainer = XAIExplainer()
    
    if explainer.xgb_model is None:
        print("⚠️ XGBoost model not trained")
        return
    
    # Example skills
    skills = "Docker Kubernetes AWS DevOps Jenkins CI/CD Terraform Ansible"
    
    print(f"\n📝 Input Skills:")
    print(f"   {skills}")
    
    try:
        # Get LIME explanation
        lime_exp = explainer.explain_with_lime(skills, num_features=10)
        
        print(f"\n🔍 LIME Analysis:")
        print(f"   Predicted Class: {lime_exp['predicted_class']}")
        print(f"   Confidence: {lime_exp['confidence']:.1%}")
        print(f"\n   Summary: {lime_exp['summary']}")
        
        print(f"\n📊 Top 10 LIME Features:")
        print(f"   {'Feature':<30} {'Weight':>12} {'Impact':<10}")
        print("   " + "-"*55)
        
        for feat in lime_exp['features'][:10]:
            impact_icon = '✅' if feat['impact'] == 'positive' else '❌'
            print(f"   {feat['feature']:<30} {feat['weight']:>12.4f} {impact_icon} {feat['impact']:<10}")
        
    except Exception as e:
        print(f"❌ LIME explanation failed: {e}")


def demo_comparison():
    """Demo 4: Compare SHAP and LIME"""
    print("\n" + "="*80)
    print("DEMO 4: SHAP vs LIME COMPARISON")
    print("="*80)
    
    try:
        from xai_explainer import XAIExplainer
    except ImportError:
        print("❌ XAI module not available")
        return
    
    explainer = XAIExplainer()
    
    if explainer.xgb_model is None:
        print("⚠️ XGBoost model not trained")
        return
    
    # Example skills
    skills = "Java Spring Boot REST API Microservices MySQL PostgreSQL"
    
    print(f"\n📝 Input Skills:")
    print(f"   {skills}")
    
    try:
        # Get both explanations
        comparison = explainer.compare_explanations(skills)
        
        print(f"\n🎯 Prediction:")
        print(f"   Role:       {comparison['prediction']['predicted_role']}")
        print(f"   Confidence: {comparison['prediction']['confidence']:.1%}")
        
        print(f"\n📊 SHAP Top 5:")
        for feat in comparison['shap']['top_features'][:5]:
            print(f"   {feat['feature']:<25} {feat['shap_value']:>8.4f}")
        
        print(f"\n📊 LIME Top 5:")
        for feat in comparison['lime']['features'][:5]:
            print(f"   {feat['feature']:<25} {feat['weight']:>8.4f}")
        
        print(f"\n⚖️ Comparison:")
        print(f"   {comparison['comparison_summary']}")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")


def demo_actionable_insights():
    """Demo 5: Generate actionable insights"""
    print("\n" + "="*80)
    print("DEMO 5: ACTIONABLE INSIGHTS FOR CANDIDATES")
    print("="*80)
    
    try:
        from xai_explainer import XAIExplainer
    except ImportError:
        print("❌ XAI module not available")
        return
    
    explainer = XAIExplainer()
    
    if explainer.xgb_model is None:
        print("⚠️ XGBoost model not trained")
        return
    
    # Candidate skills (moderate profile)
    skills = "Python SQL pandas data analysis basic machine learning"
    
    print(f"\n👤 Candidate Profile:")
    print(f"   Skills: {skills}")
    
    try:
        # Get explanation
        shap_exp = explainer.explain_with_shap(skills)
        
        print(f"\n🎯 Assessment:")
        print(f"   Best Match: {shap_exp['predicted_class']}")
        print(f"   Confidence: {shap_exp['confidence']:.1%}")
        
        # Extract strengths and weaknesses
        positive_features = [f for f in shap_exp['top_features'] if f['impact'] == 'positive']
        negative_features = [f for f in shap_exp['top_features'] if f['impact'] == 'negative']
        
        print(f"\n✅ Your Strengths:")
        for feat in positive_features[:5]:
            print(f"   • {feat['feature']}: Strong indicator for the role")
        
        print(f"\n📈 Areas to Develop:")
        for feat in negative_features[:5]:
            print(f"   • {feat['feature']}: Consider adding this to your skillset")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if shap_exp['confidence'] < 0.5:
            print("   • Low confidence suggests skill gaps")
            print("   • Focus on developing core competencies")
            print("   • Consider online courses or certifications")
        elif shap_exp['confidence'] < 0.75:
            print("   • You're on the right track!")
            print("   • Strengthen your top skills")
            print("   • Add 2-3 key missing technologies")
        else:
            print("   • Strong profile for this role!")
            print("   • Focus on practical experience")
            print("   • Consider advanced specializations")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")


def main():
    """Run all demos"""
    print("="*80)
    print("EXPLAINABLE AI (XAI) DEMO SUITE")
    print("="*80)
    print("\nThis demo showcases SHAP and LIME explanations for job role predictions.")
    print("Make sure you've trained the XGBoost model first:")
    print("  python ATS-agent/ATS-agent/train_xai_model.py")
    
    demos = [
        ("Basic Predictions", demo_basic_prediction),
        ("SHAP Explanations", demo_shap_explanation),
        ("LIME Explanations", demo_lime_explanation),
        ("SHAP vs LIME", demo_comparison),
        ("Actionable Insights", demo_actionable_insights),
    ]
    
    print("\n" + "="*80)
    print("AVAILABLE DEMOS:")
    print("="*80)
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    print("0. Run All Demos")
    
    try:
        choice = input("\nSelect demo (0-5): ").strip()
        
        if choice == '0':
            for name, demo_func in demos:
                demo_func()
        elif choice in ['1', '2', '3', '4', '5']:
            idx = int(choice) - 1
            demos[idx][1]()
        else:
            print("Invalid choice")
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nFor more information, see XAI_INTEGRATION_GUIDE.md")


if __name__ == "__main__":
    main()
