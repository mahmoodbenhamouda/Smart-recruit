# 🚀 Quick Start Guide - Testing XAI with Streamlit

## Prerequisites

Make sure you have the XGBoost model trained:
```powershell
cd ATS-agent\ATS-agent
python train_xai_model.py
```

## Installation

Install required packages:
```powershell
pip install streamlit plotly
```

Or update your requirements.txt and install:
```powershell
pip install -r requirements.txt
```

## Running the Streamlit App

### Option 1: From ATS-agent directory

```powershell
cd ATS-agent\ATS-agent
streamlit run streamlit_xai_test.py
```

### Option 2: From project root

```powershell
streamlit run ATS-agent\ATS-agent\streamlit_xai_test.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Features

### 🎯 Tab 1: Predict & Explain
- Enter skills and get job role predictions
- Generate SHAP explanations (feature importance)
- Generate LIME explanations (word-level influence)
- View top predictions with confidence scores
- Interactive visualizations

**Example Usage:**
1. Click "ML Engineer" in sidebar for pre-filled example
2. Click "🎯 Predict Role" to see predictions
3. Click "📊 SHAP Explanation" for detailed feature analysis
4. Click "📊 LIME Explanation" for word-level explanations

### 📊 Tab 2: Compare Methods
- Compare SHAP and LIME side-by-side
- See which features both methods agree on
- Understand different explanation perspectives

**Example Usage:**
1. Enter skills: "Docker Kubernetes AWS DevOps"
2. Click "🔍 Compare Methods"
3. View SHAP vs LIME analysis

### 🔍 Tab 3: Explore Job Roles
- Browse all 45 available job roles
- Search for specific roles
- See complete role list

### 📈 Tab 4: Batch Analysis
- Analyze multiple skill sets at once
- Export results to CSV
- Quick bulk predictions

**Example Usage:**
1. Enter multiple skill sets (one per line):
   ```
   Python TensorFlow Machine Learning
   Docker Kubernetes AWS
   Java Spring Boot Microservices
   ```
2. Click "🚀 Analyze Batch"
3. Download results as CSV

## Sidebar Controls

### ⚙️ Settings
- **Top Features to Show**: Adjust how many features to display (5-20)
- **LIME Samples**: Control LIME accuracy vs speed (100-2000)
- **Show All Predictions**: View probabilities for all 45 job roles

### 📝 Quick Examples
Click any button to auto-fill skills:
- **ML Engineer**: ML/AI focused skills
- **DevOps Engineer**: Infrastructure/cloud skills
- **Data Scientist**: Analytics focused skills
- **Software Engineer**: General development skills
- **Data Engineer**: Data pipeline skills

## Tips & Tricks

### 1. Understanding SHAP Values
- **Positive value (+)**: Feature increases prediction confidence
- **Negative value (-)**: Feature decreases prediction confidence
- **Larger magnitude**: Stronger influence

### 2. Understanding LIME Weights
- **Positive weight (+)**: Word supports the prediction
- **Negative weight (-)**: Word contradicts the prediction
- Similar to SHAP but focuses on word-level

### 3. Performance Tips
- Start with fewer LIME samples (500) for faster results
- SHAP is generally faster than LIME
- Use batch analysis for multiple predictions

### 4. Best Results
- Enter 5-10 relevant skills
- Use specific technical terms
- Mix hard and soft skills
- Be consistent with terminology

## Troubleshooting

### Model Not Loaded
**Error**: "❌ Model Not Loaded"

**Solution**:
```powershell
cd ATS-agent\ATS-agent
python train_xai_model.py
```

### Import Errors
**Error**: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**:
```powershell
pip install streamlit plotly pandas
```

### Slow Performance
**Issue**: LIME taking too long

**Solution**: Reduce LIME samples in sidebar (500 instead of 1000)

### Port Already in Use
**Error**: "Address already in use"

**Solution**:
```powershell
streamlit run streamlit_xai_test.py --server.port 8502
```

## Screenshots & Examples

### Example 1: ML Engineer
**Input**: `Python TensorFlow Keras PyTorch Deep Learning Neural Networks`

**Expected Output**:
- Prediction: Machine Learning Engineer / Data Scientist
- Confidence: 70-90%
- Top SHAP features: tensorflow, pytorch, deep, learning

### Example 2: DevOps
**Input**: `Docker Kubernetes AWS Jenkins CI/CD Terraform`

**Expected Output**:
- Prediction: DevOps Engineer / Cloud Engineer
- Confidence: 70-85%
- Top SHAP features: docker, kubernetes, aws

### Example 3: Full Stack
**Input**: `React Node.js Python Django REST API MongoDB`

**Expected Output**:
- Prediction: Software Engineer / Full Stack Developer
- Confidence: 60-80%
- Top SHAP features: react, node, api

## Advanced Usage

### Custom Port
```powershell
streamlit run streamlit_xai_test.py --server.port 8502
```

### Different Host
```powershell
streamlit run streamlit_xai_test.py --server.address 0.0.0.0
```

### Auto-reload on Changes
```powershell
streamlit run streamlit_xai_test.py --server.runOnSave true
```

### Dark Theme
Add to `.streamlit/config.toml`:
```toml
[theme]
base="dark"
```

## Integration with Your App

You can integrate the visualization components into your own Streamlit app:

```python
import sys
from pathlib import Path

# Add ATS-agent to path
sys.path.insert(0, 'path/to/ATS-agent/ATS-agent')

from xai_explainer import XAIExplainer

# Initialize
explainer = XAIExplainer()

# Use in your app
skills = st.text_input("Enter skills")
if st.button("Analyze"):
    result = explainer.predict_with_xgboost(skills)
    st.write(f"Predicted: {result['predicted_role']}")
    
    # Generate explanation
    shap_exp = explainer.explain_with_shap(skills)
    # Display using custom visualization...
```

## What You Can Test

✅ **Predictions**
- Job role classification
- Confidence scores
- Top-N predictions

✅ **SHAP Explanations**
- Feature importance
- Positive/negative contributions
- Base values

✅ **LIME Explanations**
- Word-level influence
- Local approximations
- Feature weights

✅ **Comparisons**
- SHAP vs LIME agreement
- Multiple explanation perspectives
- Robustness validation

✅ **Batch Processing**
- Multiple predictions
- CSV export
- Bulk analysis

## Next Steps

1. ✅ Test with your own skill combinations
2. ✅ Experiment with different job roles
3. ✅ Compare SHAP and LIME explanations
4. ✅ Use batch analysis for multiple candidates
5. ✅ Integrate visualizations into your main app

## Support

- Check `XAI_INTEGRATION_GUIDE.md` for detailed documentation
- Run `python demo_xai.py` for command-line examples
- See `XAI_SUMMARY.md` for quick reference

---

**Ready to test? Run:**
```powershell
streamlit run streamlit_xai_test.py
```

🎉 Enjoy exploring XAI explanations!
