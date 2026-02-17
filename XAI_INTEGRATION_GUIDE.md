# Explainable AI (XAI) Integration Guide

## Overview

This project now includes comprehensive **Explainable AI (XAI)** capabilities using SHAP, LIME, and XGBoost to provide transparent, interpretable explanations for job role predictions and resume screening decisions.

## 🎯 What's New

### Core Features

1. **SHAP (SHapley Additive exPlanations)**
   - Tree-based feature importance for XGBoost models
   - Shows exact contribution of each skill/feature to predictions
   - Provides mathematically rigorous explanations

2. **LIME (Local Interpretable Model-agnostic Explanations)**
   - Local linear approximations of model behavior
   - Highlights which words/phrases influenced the prediction
   - Provides intuitive, human-readable explanations

3. **XGBoost Classifier**
   - Fast, accurate gradient boosting model
   - Built-in feature importance
   - Works seamlessly with SHAP

## 📁 New Files

### ATS-agent/ATS-agent/
- `xai_explainer.py` - Core XAI module with SHAP/LIME integration
- `train_xai_model.py` - Training script for XGBoost model
- `enhanced_ats_service.py` - Enhanced ATS service with XAI

### ats_service/app/
- `xai_routes.py` - FastAPI endpoints for XAI features

### smart_resume_suite/services/
- `xai_visualization.py` - Streamlit visualization components

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Install XAI dependencies for ATS-agent
cd ATS-agent\ATS-agent
pip install -r requirements.txt

# Install XAI dependencies for ats_service
cd ..\..\ats_service
pip install -r requirements.txt

# Install visualization dependencies for smart_resume_suite
cd ..\smart_resume_suite
pip install streamlit plotly matplotlib
```

### 2. Train XGBoost Model

The XGBoost model needs to be trained before you can use XAI features:

```powershell
cd ATS-agent\ATS-agent

# Option 1: Auto-detect dataset
python train_xai_model.py

# Option 2: Specify dataset path
python train_xai_model.py --dataset "..\..\deep_Learning_Project\resume_screening_dataset_train.csv"

# Option 3: Custom model directory
python train_xai_model.py --dataset path\to\data.csv --model-dir custom_models
```

This will:
- Load training data from the resume screening dataset
- Train an XGBoost classifier
- Generate SHAP and LIME explainers
- Save models to `JobPrediction_Model/` directory
- Run test predictions with explanations

### 3. Test XAI Module

```powershell
# Test the XAI explainer directly
cd ATS-agent\ATS-agent
python xai_explainer.py
```

### 4. Start FastAPI Service with XAI

```powershell
cd ats_service
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The service will now include XAI endpoints at:
- `http://localhost:8000/docs` - Interactive API documentation
- `http://localhost:8000/api/v1/xai/*` - XAI endpoints

## 📡 API Endpoints

### 1. Analyze Resume with XAI

**POST** `/api/v1/xai/analyze`

Upload a resume and get detailed XAI explanations.

**Form Data:**
- `resume` (file): PDF resume
- `job_description` (string): Job description text
- `include_shap` (bool, default=true): Include SHAP explanations
- `include_lime` (bool, default=true): Include LIME explanations
- `compare_methods` (bool, default=false): Compare SHAP and LIME

**Response:**
```json
{
  "success": true,
  "overall_match": 75.5,
  "match_level": "Good Match",
  "predicted_role": "Machine Learning Engineer",
  "confidence": 0.85,
  "xai_explanation": {
    "method": "BOTH",
    "predicted_class": "Machine Learning Engineer",
    "confidence": 0.85,
    "shap_features": [
      {
        "feature": "python",
        "value": 0.8,
        "shap_value": 0.25,
        "impact": "positive",
        "rationale": "SHAP value of 0.2500 indicates strong positive contribution"
      }
    ],
    "lime_features": [
      {
        "feature": "tensorflow",
        "lime_weight": 0.22,
        "impact": "positive"
      }
    ],
    "summary": "Predicted role: Machine Learning Engineer. Top SHAP feature: 'python' (positive)",
    "comparison": "SHAP and LIME agree on 8 features. High agreement indicates robust explanation."
  }
}
```

### 2. Explain Prediction (Skills Only)

**POST** `/api/v1/xai/explain-prediction`

Get explanation for skills text without uploading a full resume.

**Form Data:**
- `skills_text` (string): Skills text (comma or space separated)
- `method` (string): "shap", "lime", or "both"

**Example:**
```powershell
curl -X POST "http://localhost:8000/api/v1/xai/explain-prediction" \
  -F "skills_text=Python TensorFlow Machine Learning Deep Learning PyTorch" \
  -F "method=both"
```

### 3. Model Information

**GET** `/api/v1/xai/model-info`

Get information about XAI models and their status.

**Response:**
```json
{
  "success": true,
  "xai_status": {
    "xai_available": true,
    "xgboost_trained": true,
    "job_roles": ["Machine Learning Engineer", "Data Science", "DevOps", ...],
    "explainability_methods": ["SHAP", "LIME"]
  }
}
```

## 🎨 Streamlit Visualization

Use the visualization components in your Streamlit apps:

```python
import streamlit as st
from services.xai_visualization import display_xai_explanation, create_xai_dashboard

# Display a single explanation
explanation = {
    'method': 'BOTH',
    'predicted_class': 'Machine Learning Engineer',
    'confidence': 0.85,
    'shap_features': [...],
    'lime_features': [...]
}

display_xai_explanation(explanation)

# Or create a complete dashboard
analysis_result = {
    'xai_explanation': explanation,
    'job_prediction': {...},
    'matched_skills': [...],
    'missing_skills': [...]
}

create_xai_dashboard(analysis_result)
```

## 🔍 Understanding the Explanations

### SHAP Values

- **Positive SHAP Value** (+): Feature increases prediction probability
- **Negative SHAP Value** (−): Feature decreases prediction probability
- **Magnitude**: Larger absolute value = stronger influence

**Example:**
```
python: +0.25  → Strong positive contributor
java: -0.05    → Slight negative contributor
```

### LIME Weights

- **Positive Weight** (+): Feature supports the prediction
- **Negative Weight** (−): Feature contradicts the prediction
- **Magnitude**: Indicates strength of local influence

### When to Use Each

- **SHAP**: More accurate, globally consistent, best for model debugging
- **LIME**: Faster, easier to interpret, best for end-user explanations
- **Both**: Maximum transparency and validation

## 📊 Example Use Cases

### 1. Resume Screening

```python
from xai_explainer import XAIExplainer

explainer = XAIExplainer()

# Analyze candidate skills
skills = "Python TensorFlow Deep Learning Neural Networks"
explanation = explainer.explain_with_shap(skills)

print(f"Predicted Role: {explanation['predicted_class']}")
print(f"Top Features:")
for feat in explanation['top_features'][:5]:
    print(f"  {feat['feature']}: {feat['shap_value']:+.4f}")
```

### 2. Job Role Prediction

```python
# Compare SHAP and LIME
comparison = explainer.compare_explanations(
    "Docker Kubernetes AWS DevOps Jenkins"
)

print(f"SHAP says: {comparison['shap']['summary']}")
print(f"LIME says: {comparison['lime']['summary']}")
print(f"Agreement: {comparison['comparison_summary']}")
```

### 3. Candidate Feedback

```python
# Generate actionable feedback
prediction = explainer.predict_with_xgboost(resume_text)
explanation = explainer.explain_with_lime(resume_text)

positive_features = [f for f in explanation['features'] if f['impact'] == 'positive']
negative_features = [f for f in explanation['features'] if f['impact'] == 'negative']

print("Your Strengths:")
for feat in positive_features[:5]:
    print(f"  ✅ {feat['feature']}")

print("\nAreas to Improve:")
for feat in negative_features[:5]:
    print(f"  📈 {feat['feature']}")
```

## 🛠️ Configuration

### Model Parameters

Edit `xai_explainer.py` to adjust XGBoost parameters:

```python
self.xgb_model = xgb.XGBClassifier(
    n_estimators=200,        # Number of trees
    max_depth=6,             # Tree depth
    learning_rate=0.1,       # Learning rate
    random_state=42
)
```

### TF-IDF Settings

Adjust feature extraction:

```python
self.vectorizer = TfidfVectorizer(
    max_features=5000,       # Maximum vocabulary size
    ngram_range=(1, 3),      # Use 1-3 word phrases
    min_df=2,                # Minimum document frequency
    max_df=0.8              # Maximum document frequency
)
```

### LIME Parameters

Customize LIME explanations:

```python
explainer.explain_with_lime(
    text,
    num_features=20,         # Features to show
    num_samples=1000         # Perturbation samples
)
```

## 🧪 Testing

### Unit Tests

```powershell
# Test XAI module
python -m pytest tests/test_xai_explainer.py

# Test API endpoints
python -m pytest tests/test_xai_routes.py
```

### Manual Testing

```powershell
# Test training
python train_xai_model.py --dataset test_data.csv

# Test predictions
python xai_explainer.py

# Test API
curl -X POST http://localhost:8000/api/v1/xai/model-info
```

## 📈 Performance

- **XGBoost Training**: ~1-5 minutes (depends on dataset size)
- **SHAP Explanation**: ~0.1-0.5 seconds per prediction
- **LIME Explanation**: ~1-3 seconds per prediction (adjustable via num_samples)

## 🔧 Troubleshooting

### Model Not Trained

**Error**: `XGBoost model not loaded`

**Solution**: Train the model first:
```powershell
python train_xai_model.py
```

### Dataset Not Found

**Error**: `Dataset not found at...`

**Solution**: Provide explicit path:
```powershell
python train_xai_model.py --dataset "path\to\your\dataset.csv"
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'shap'`

**Solution**: Install dependencies:
```powershell
pip install shap lime xgboost matplotlib plotly
```

### Slow LIME Explanations

**Solution**: Reduce num_samples:
```python
explainer.explain_with_lime(text, num_samples=500)  # Default is 1000
```

## 📚 Additional Resources

- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Paper](https://arxiv.org/abs/1602.04938)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)

## 🤝 Contributing

To extend XAI features:

1. Add new explainability methods in `xai_explainer.py`
2. Create corresponding API endpoints in `xai_routes.py`
3. Add visualization components in `xai_visualization.py`
4. Update this documentation

## 📄 License

Same as the main project.

---

**Note**: XAI features require training data. Use the existing `resume_screening_dataset_train.csv` or provide your own labeled dataset with columns for text/skills and job roles.
