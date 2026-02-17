# 🎯 XAI Integration with SmartRecruiter Chatbot

## Overview
Successfully integrated **Explainable AI (SHAP/LIME)** with **XGBoost** into the SmartRecruiter feedback chatbot that uses Groq LLM.

## What Was Added

### 1. **XAI-Powered Job Role Prediction**
- XGBoost classifier predicts the best-fit job role for candidate's CV
- 96.27% test accuracy on 45 job roles
- Confidence scores for top 5 predicted roles

### 2. **SHAP Analysis**
- Shows which skills/features contribute most to the prediction
- Positive/negative impact visualization
- Percentage point contributions for interpretability

### 3. **LIME Explanations**
- Local interpretability - explains individual predictions
- Shows how each word/phrase influences the decision
- Alternative "what-if" scenario analysis

### 4. **Enhanced LangGraph Pipeline**
The chatbot now follows this enhanced workflow:
```
User uploads CV + asks question
          ↓
    RAG Retrieval (find similar cases)
          ↓
    XAI Prediction (XGBoost + SHAP + LIME)
          ↓
  ╔═══════════════╦════════════════╦═══════════════╗
  ↓               ↓                ↓               
Feedback     Coaching         Alternative Roles
(why rejected) (how to improve) (better fits)
  ↓               ↓                ↓
  ╚═══════════════╩════════════════╩═══════════════╝
          ↓
    Synthesize (Groq LLM combines all insights)
          ↓
    Final Answer with XAI insights
```

## Files Modified/Added

### Modified:
1. **`app.py`** - Main Streamlit application
   - Added XAI explainer initialization
   - Added `node_xai_predict()` to LangGraph
   - Enhanced feedback/coaching/matching nodes with XAI insights
   - Added new "🔍 XAI Analysis" tab
   - Updated prompts to include SHAP/LIME insights

### Added:
2. **`xai_explainer.py`** - Core XAI module (copied from ATS-agent)
3. **`resume_skills_train.csv`** - Skills-focused training dataset
4. **`JobPrediction_Model/`** - Trained XGBoost model + vectorizer + encoder

## Features

### Tab 1: 💬 Chat
- Upload CV (PDF)
- Ask questions (e.g., "Why was I rejected?", "What should I improve?")
- Get AI-powered feedback with XAI insights embedded
- Real-time conversation with context from:
  - RAG retrieval (similar cases from dataset)
  - XAI predictions (your best-fit roles)
  - SHAP analysis (key skills driving predictions)

### Tab 2: 🔍 XAI Analysis
- **Job Role Prediction**
  - Predicted best-fit role with confidence
  - Top 5 role matches with probabilities
  
- **SHAP Visualization**
  - Interactive bar chart showing feature importance
  - Color-coded: Green (positive) / Red (negative)
  - Percentage point contributions (e.g., +0.25pp)
  
- **LIME Visualization**
  - Alternative explanation method
  - Word-level importance scores
  - Shows which phrases most influenced the prediction

### Tab 3: 📊 Analytics
- Dataset statistics
- Rejection reasons distribution
- Common roles
- Original analytics dashboard

## How XAI Enhances the Chatbot

### Before (Without XAI):
```
User: "Why was I rejected for Data Scientist role?"
Bot: "Based on similar cases, you may lack experience in 
      machine learning and statistics..."
```
❌ Generic feedback based only on RAG retrieval

### After (With XAI):
```
User: "Why was I rejected for Data Scientist role?"
Bot: 
**AI-Powered Analysis:**
- Your CV best matches: Machine Learning Engineer (18.4%)
- Target role (Data Scientist) is your 2nd best fit (11.2%)

**Key Skills Identified (SHAP):**
✅ machine learning: +0.141pp (strong)
✅ python: +0.082pp (good)
❌ statistics: -0.045pp (missing!)
❌ data analysis: -0.032pp (weak)

**Feedback:** While you have strong ML skills, your CV lacks 
emphasis on statistical analysis and data visualization...

**Coaching:** 
1. Complete "Statistics for Data Science" on Coursera
2. Add data analysis projects (pandas, matplotlib)
3. Highlight statistical modeling experience...

**Alternative Roles:** Your profile strongly matches:
- Machine Learning Engineer (18.4% confidence)
- AI Engineer (8.7% confidence)
```
✅ **Specific, data-driven, actionable feedback!**

## Technical Details

### XAI Integration Points:

1. **State Extension** (`SRState` TypedDict):
```python
xai_prediction: Dict[str, Any]  # XGBoost prediction with confidence
xai_shap: Dict[str, Any]        # SHAP explanation
xai_lime: Dict[str, Any]        # LIME explanation
```

2. **New LangGraph Node**:
```python
def node_xai_predict(state: SRState) -> Dict[str, Any]:
    """Predicts job role with SHAP/LIME explainability"""
    # Extract skills from CV
    # Run XGBoost prediction
    # Generate SHAP explanation
    # Generate LIME explanation
    return xai results
```

3. **Enhanced Prompts**:
All agent prompts now include:
- Predicted role vs target role comparison
- Top 3 alternative roles
- SHAP feature importance (top 5 skills)
- Skill gap analysis

4. **Visualization**:
- Plotly interactive charts for SHAP/LIME
- Color-coded features (positive/negative impact)
- Sortable tables with contribution values

## How to Run

### 1. Install Dependencies
```bash
cd deep_Learning_Project
pip install -r requirements.txt
```

### 2. Set Groq API Key
Create a `.env` file:
```
GROQ_API_KEY=gsk_your_actual_api_key_here
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Use the Chatbot
1. Upload your CV (PDF)
2. Ask questions in the Chat tab
3. View XAI analysis in the XAI Analysis tab
4. Check dataset analytics in Analytics tab

## Example Questions to Ask

- "Why was I rejected for this position?"
- "What skills am I missing?"
- "What should I learn next?"
- "What roles am I better suited for?"
- "How can I improve my chances?"
- "What are my strongest skills?"

## Benefits of XAI Integration

### For Candidates:
✅ **Transparency**: Understand exactly why you were/weren't selected
✅ **Actionable Feedback**: Know which specific skills to improve
✅ **Career Guidance**: Discover alternative roles you're better suited for
✅ **Confidence**: See quantified skill assessments

### For Recruiters:
✅ **Data-Driven**: Decisions backed by ML model with 96% accuracy
✅ **Explainability**: Can justify decisions with SHAP/LIME analysis
✅ **Fairness**: Objective skill-based evaluation
✅ **Efficiency**: Automated initial screening with explanations

## Model Performance

- **Training Dataset**: 10,174 resumes across 45 job roles
- **Training Accuracy**: 100%
- **Test Accuracy**: 96.27%
- **F1 Score**: 96.18%
- **Model**: XGBoost with TF-IDF features (5000 max features)

## Troubleshooting

### "XAI features not available"
- Ensure `xai_explainer.py` is in the same directory as `app.py`
- Ensure `JobPrediction_Model/` directory exists with trained models
- Run: `pip install xgboost shap lime matplotlib plotly`

### "Groq API key is not set"
- Create `.env` file with `GROQ_API_KEY=your_key`
- Or set environment variable: `$env:GROQ_API_KEY="your_key"`

### "Dataset not found"
- Ensure `resume_screening_dataset_train.csv` is in the same directory
- Check the file path in the error message

## Future Enhancements

- [ ] Real-time SHAP waterfall charts
- [ ] LIME interactive text highlighting
- [ ] Model comparison (XGBoost vs LSTM)
- [ ] Export XAI report as PDF
- [ ] Batch analysis for multiple candidates
- [ ] Custom model training interface

## Credits

- **XGBoost**: Gradient boosting for classification
- **SHAP**: SHapley Additive exPlanations
- **LIME**: Local Interpretable Model-agnostic Explanations
- **Groq**: Fast LLM inference
- **LangGraph**: Multi-agent orchestration
- **Streamlit**: Web app framework

---

**Status**: ✅ Fully integrated and tested
**Date**: November 10, 2025
**Impact**: Enhanced chatbot with explainable AI insights for better candidate feedback
