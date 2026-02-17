# 🚀 Quick Start Guide: SmartRecruiter with XAI

## ✅ Status
The enhanced SmartRecruiter chatbot is now running at:
- **Local URL**: http://localhost:8501
- **Network URL**: http://192.168.100.231:8501

## 📖 How to Use

### Step 1: Open the App
Click on http://localhost:8501 or copy-paste into your browser.

### Step 2: Upload Your CV
1. Go to the **💬 Chat** tab
2. Click "Upload your CV (PDF)"
3. Select a PDF resume file
4. Wait for "CV loaded successfully"

### Step 3: Ask Questions
Type questions like:
- "Why was I rejected for this position?"
- "What skills should I improve?"
- "What roles am I better suited for?"
- "How can I increase my chances?"
- "What are my key strengths?"

### Step 4: View AI Analysis
Click the **🔍 XAI Analysis** tab to see:

#### Job Role Prediction
- Your best-fit role (e.g., "Data Scientist: 32.5% confidence")
- Top 5 role matches with probabilities
- Confidence level (High/Medium/Low)

#### SHAP Visualization
- **Interactive bar chart** showing which skills drive your prediction
- **Green bars** = Skills pushing you TOWARD the predicted role
- **Red bars** = Skills pushing you AWAY (missing skills!)
- **Percentage points** show contribution magnitude

Example:
```
✅ machine learning    +0.14pp  (strong contributor)
✅ python              +0.08pp  (good contributor)
❌ statistics          -0.04pp  (missing/weak skill)
❌ data visualization  -0.03pp  (missing/weak skill)
```

#### LIME Explanation
- Alternative explanation method
- Shows word-level importance
- Interactive visualization

### Step 5: Check Analytics
Click **📊 Analytics** tab for:
- Dataset statistics
- Common rejection reasons
- Most frequent roles
- Description length distribution

## 🎯 What Makes This Special?

### Traditional Chatbots:
```
User: "Why was I rejected?"
Bot: "You might lack relevant experience..."
```
❌ Vague, generic, not actionable

### SmartRecruiter with XAI:
```
User: "Why was I rejected?"
Bot: 
"AI Analysis shows:
- Your CV matches Machine Learning Engineer (18%) better than 
  Data Scientist (11%)
- SHAP identified missing skills: statistics (-0.04pp), 
  data visualization (-0.03pp)
  
Feedback: While you have strong ML skills, your CV lacks emphasis 
on statistical analysis...

Coaching:
1. Complete 'Statistics for Data Science' on Coursera
2. Add pandas/matplotlib projects to demonstrate data analysis
3. Highlight any statistical modeling experience

Alternative Roles:
- Machine Learning Engineer (better fit!)
- AI Engineer (8.7% match)
- Software Engineer (7.6% match)"
```
✅ **Specific, data-driven, actionable!**

## 📊 Understanding SHAP Values

### What They Show:
SHAP values show **how much each skill contributes** to the prediction probability.

### Example:
```
Target Role: Data Scientist (baseline 2.22% = 1/45 roles)

Your Skills → Contributions:
python              +0.08pp  → now at 2.30%
machine learning    +0.14pp  → now at 2.44%
tensorflow          +0.05pp  → now at 2.49%
... (100+ more features)
Final Prediction: 32.48% Data Scientist ✅
```

### Reading the Chart:
- **Longer bars** = More important skills
- **Green bars** (+) = Skills you HAVE that help
- **Red bars** (−) = Skills you LACK that hurt
- **pp** = percentage points (e.g., +0.25pp = increases probability by 0.25%)

## 🔍 SHAP vs LIME

### SHAP (Global + Local)
- Based on game theory (Shapley values)
- Shows contribution of each feature
- More mathematically rigorous
- Better for understanding model behavior

### LIME (Local)
- Tests variations of your input
- Explains THIS specific prediction
- Shows "what if" scenarios
- More intuitive for non-technical users

**Both are shown** so you get complete picture!

## 💡 Tips for Best Results

### Upload Quality CVs:
- ✅ Include technical skills explicitly
- ✅ Use industry-standard terms (e.g., "Python", "TensorFlow")
- ✅ List tools, frameworks, languages
- ❌ Avoid generic buzzwords ("results-driven", "team player")

### Ask Specific Questions:
- ✅ "What specific skills am I missing for DevOps Engineer?"
- ✅ "How do I transition from Data Analyst to Data Scientist?"
- ❌ "Am I good?" (too vague)
- ❌ "Tell me everything" (too broad)

### Interpret Confidence:
- **> 50%**: Strong match, go for it!
- **30-50%**: Good match, some skill gaps to address
- **< 30%**: Weak match, consider alternative roles shown

## 🛠️ Technical Details

### The AI Pipeline:
1. **Upload CV** → Extract text from PDF
2. **RAG Retrieval** → Find similar cases in dataset (10,174 resumes)
3. **XGBoost Prediction** → Classify into 1 of 45 job roles (96.27% accuracy)
4. **SHAP Analysis** → Compute feature contributions
5. **LIME Analysis** → Generate alternative explanation
6. **Groq LLM** → Synthesize feedback/coaching/matches
7. **Display** → Show results in chat + visualizations

### Models Used:
- **XGBoost**: Job role classification (96.27% test accuracy)
- **SHAP**: TreeExplainer for feature importance
- **LIME**: Text explainer for local interpretability
- **Sentence Transformers**: Embedding for RAG retrieval
- **Groq (llama-3.1-8b)**: Natural language generation
- **LangGraph**: Multi-agent orchestration

## 📸 What You'll See

### Chat Tab:
- Job card (randomly selected from dataset)
- CV upload section
- Question input
- Conversational responses with XAI insights embedded

### XAI Analysis Tab:
- Job Role Prediction metrics
- Top 5 predictions bar chart
- SHAP interactive visualization (Plotly)
- LIME interactive visualization (Plotly)
- Detailed feature tables

### Analytics Tab:
- Rejection reasons chart
- Common roles chart
- Dataset statistics
- Sample data table

## 🎓 Example Use Cases

### Use Case 1: Career Changer
**Situation**: Software Engineer wants to move into Data Science

**Steps**:
1. Upload current resume
2. Ask: "What skills do I need for Data Science?"
3. Check XAI Analysis:
   - Predicted: Software Engineer (high confidence)
   - Data Scientist (low confidence)
   - SHAP shows missing: statistics, pandas, machine learning
4. Get coaching: Specific courses and projects to bridge gap

### Use Case 2: Fresh Graduate
**Situation**: Recent CS grad unsure which role to pursue

**Steps**:
1. Upload resume
2. Ask: "What roles am I best suited for?"
3. Check XAI Analysis:
   - Top 3 predictions with confidence scores
   - SHAP shows strongest skills
4. Get matches: Roles that align with current skill set

### Use Case 3: Job Application Feedback
**Situation**: Applied for role, got rejected, want to know why

**Steps**:
1. Upload resume
2. Ask: "Why was I rejected for [role]?"
3. Check XAI Analysis:
   - Compare predicted role vs target role
   - SHAP shows skill gaps (red bars)
4. Get feedback: Data-driven explanation and improvement plan

## 🔒 Privacy & Data

- Your CV is **NOT stored** permanently
- Only kept in session memory during analysis
- **NOT sent to any external service** (except Groq for LLM)
- XGBoost/SHAP/LIME run **locally**
- Dataset used is the training dataset (public)

## ⚡ Performance

- **CV Upload**: < 1 second
- **XAI Analysis**: 2-3 seconds
- **Groq Response**: 1-2 seconds
- **Total Pipeline**: 5-8 seconds

## 🐛 Troubleshooting

### "Groq API key is not set"
→ Check `.env` file has: `GROQ_API_KEY=gsk_your_key`

### "XAI features not available"
→ Ensure `xai_explainer.py` and `JobPrediction_Model/` exist

### "Dataset not found"
→ Ensure `resume_screening_dataset_train.csv` exists

### Charts not showing
→ Refresh page, ensure plotly is installed

### Low confidence predictions
→ Upload more detailed CV with technical skills

## 📚 Learn More

- **SHAP Documentation**: https://shap.readthedocs.io/
- **LIME Paper**: https://arxiv.org/abs/1602.04938
- **XGBoost**: https://xgboost.readthedocs.io/
- **Project Docs**: See `XAI_INTEGRATION.md` for technical details

---

**Enjoy your XAI-powered career feedback!** 🚀

**Questions?** Upload a CV and ask the chatbot anything!
