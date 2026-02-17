"""
Training script for XGBoost model with explainability features
Uses the resume screening dataset to train an explainable job role classifier
"""

import sys
import pandas as pd
from pathlib import Path
from xai_explainer import XAIExplainer


def load_training_data(dataset_path: str = None) -> tuple:
    """
    Load training data from CSV dataset.
    
    Args:
        dataset_path: Path to CSV file with resume data
        
    Returns:
        Tuple of (texts, labels)
    """
    # Try multiple possible dataset locations
    # PRIORITY: Use skills-based dataset if available (better for prediction)
    possible_paths = [
        dataset_path,
        Path("resume_skills_train.csv"),  # Skills-focused dataset (preferred)
        Path(__file__).parent.parent.parent / "deep_Learning_Project" / "resume_screening_dataset_train.csv",
        Path("resume_screening_dataset_train.csv"),
        Path("../deep_Learning_Project/resume_screening_dataset_train.csv"),
    ]
    
    df = None
    for path in possible_paths:
        if path and Path(path).exists():
            print(f"📂 Loading dataset from: {path}")
            df = pd.read_csv(path)
            break
    
    if df is None:
        raise FileNotFoundError(
            "Could not find resume_screening_dataset_train.csv. "
            "Please provide the path to your training dataset."
        )
    
    # Normalize column names
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Find text and label columns
    text_col = None
    label_col = None
    
    # Priority-based column selection for ATS dataset
    # 1. Try to find 'skills' column for text (highest priority for skills-based dataset)
    for col in df.columns:
        if col == 'skills':
            text_col = col
            break
    
    # 2. Try to find 'resume' column for text
    if text_col is None:
        for col in df.columns:
            if col == 'resume':
                text_col = col
                break
    
    # If not found, try other text patterns
    if text_col is None:
        for col in df.columns:
            if 'skill' in col or 'resume' in col or 'text' in col:
                text_col = col
                break
    
    # 2. Try to find 'role' column for label (NOT job_description!)
    for col in df.columns:
        if col == 'role' or col == 'category':
            label_col = col
            break
    
    # If not found, try other label patterns (but exclude 'job_description')
    if label_col is None:
        for col in df.columns:
            if ('role' in col or 'category' in col or 'title' in col or 'label' in col) and 'job' not in col and 'description' not in col:
                label_col = col
                break
    
    if text_col is None or label_col is None:
        print(f"Available columns: {df.columns.tolist()}")
        print(f"Detected: text_col='{text_col}', label_col='{label_col}'")
        raise ValueError("Could not automatically identify text and label columns.")
    
    print(f"✅ Using columns: text='{text_col}', label='{label_col}'")
    print(f"   Dataset size: {len(df)} samples")
    print(f"   Unique roles: {df[label_col].nunique()}")
    
    # Prepare data
    df = df.dropna(subset=[text_col, label_col])
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(str).tolist()
    
    # Show class distribution
    print("\n📊 Class distribution:")
    class_counts = df[label_col].value_counts()
    for role, count in class_counts.head(10).items():
        print(f"   {role:30s}: {count:4d} samples")
    if len(class_counts) > 10:
        print(f"   ... and {len(class_counts) - 10} more roles")
    
    return texts, labels


def train_model(dataset_path: str = None, model_dir: str = "JobPrediction_Model"):
    """
    Train XGBoost model with XAI capabilities.
    
    Args:
        dataset_path: Path to training dataset CSV
        model_dir: Directory to save trained models
    """
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODEL WITH EXPLAINABLE AI")
    print("="*80)
    
    # Load training data
    print("\n📚 Loading training data...")
    texts, labels = load_training_data(dataset_path)
    
    # Initialize explainer
    print(f"\n🔧 Initializing XAI explainer (model dir: {model_dir})...")
    explainer = XAIExplainer(model_path=model_dir)
    
    # Train model
    print("\n🚀 Starting training...")
    metrics = explainer.train_xgboost_model(
        texts=texts,
        labels=labels,
        max_features=5000,
        test_size=0.2
    )
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
    print(f"Test Accuracy:  {metrics['test_accuracy']:.2%}")
    print(f"Train F1 Score: {metrics['train_f1']:.2%}")
    print(f"Test F1 Score:  {metrics['test_f1']:.2%}")
    print(f"Total Samples:  {metrics['n_samples']}")
    print(f"Features:       {metrics['n_features']}")
    
    # Test prediction with explanation
    print("\n" + "="*80)
    print("TEST PREDICTION WITH EXPLANATIONS")
    print("="*80)
    
    test_cases = [
        "Python TensorFlow Keras Machine Learning Deep Learning Neural Networks PyTorch",
        "Docker Kubernetes AWS Jenkins CI/CD Terraform Ansible DevOps",
        "Java Spring Boot Microservices REST API MySQL",
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"Test Case {i}: {test_text[:60]}...")
        print('='*60)
        
        # Make prediction
        prediction = explainer.predict_with_xgboost(test_text)
        print(f"\n🎯 Prediction: {prediction['predicted_role']}")
        print(f"   Confidence: {prediction['confidence']:.2%}")
        
        # SHAP explanation
        try:
            shap_exp = explainer.explain_with_shap(test_text)
            print(f"\n📊 SHAP Explanation:")
            print(f"   {shap_exp['summary']}")
            print("\n   Top 5 Features:")
            for feat in shap_exp['top_features'][:5]:
                impact_symbol = '✅' if feat['impact'] == 'positive' else '❌'
                print(f"   {impact_symbol} {feat['feature']:20s} {feat['shap_value']:+.4f}")
        except Exception as e:
            print(f"   ⚠️ SHAP explanation error: {e}")
        
        # LIME explanation
        try:
            lime_exp = explainer.explain_with_lime(test_text, num_features=10)
            print(f"\n📊 LIME Explanation:")
            print(f"   {lime_exp['summary']}")
            print("\n   Top 5 Features:")
            for feat in lime_exp['features'][:5]:
                impact_symbol = '✅' if feat['impact'] == 'positive' else '❌'
                print(f"   {impact_symbol} {feat['feature']:25s} {feat['weight']:+.4f}")
        except Exception as e:
            print(f"   ⚠️ LIME explanation error: {e}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"Models saved to: {model_dir}")
    print("You can now use xai_explainer.py for predictions with explanations.")


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train XGBoost model with SHAP and LIME explainability"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to training dataset CSV'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='JobPrediction_Model',
        help='Directory to save trained models'
    )
    
    args = parser.parse_args()
    
    try:
        train_model(dataset_path=args.dataset, model_dir=args.model_dir)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
