# 🚀 TalentBridge - Quick Reference Card

## ⚡ 1-Minute Setup

```bash
# Install
cd server && npm install && cd ../client && npm install && cd ..
python -m venv venv && venv\Scripts\activate && pip install -r ats_api_requirements.txt

# Configure
# 1. Copy server/env.example → server/.env
# 2. Set MONGODB_URI and JWT_SECRET
# 3. Start MongoDB: net start MongoDB

# Run
.\start-all.ps1  # Windows
# OR
./start-all.sh   # Mac/Linux
```

---

## 🌐 Service URLs

| Service | URL | Status |
|---------|-----|--------|
| **Frontend** | http://localhost:5173 | Main UI |
| **Backend API** | http://localhost:3000 | REST API |
| **Python ATS** | http://localhost:8000 | XAI Engine |
| **MongoDB** | mongodb://localhost:27017 | Database |

---

## 🔑 Demo Accounts

**Recruiter:**
- Email: `recruiter@demo.com`
- Password: `password123`

**Candidate:**
- Email: `candidate@demo.com`  
- Password: `password123`

---

## 📊 Key Features

### For Recruiters 👔
- Post jobs with descriptions
- View dashboard with statistics
- See top candidates ranked by AI
- Review ATS scores (0-100%)
- View SHAP feature importance
- Accept/reject applications

### For Candidates 💼
- Browse and apply to jobs
- Upload PDF resume
- Get instant ATS score
- See job role predictions
- View missing skills analysis
- Get improvement recommendations

---

## 🧠 AI Features

| Feature | Technology | Output |
|---------|------------|--------|
| **Job Prediction** | XGBoost (96.27%) | Predicted role + top 5 alternatives |
| **SHAP** | Feature Importance | Which skills drive prediction |
| **LIME** | Word-Level | Which words matter most |
| **Missing Skills** | Counterfactual | Top 10 skills to add (+impact %) |

---

## 🛠️ Troubleshooting

**MongoDB not starting?**
```bash
net start MongoDB  # Windows
```

**Port already in use?**
```bash
netstat -ano | findstr :3000  # Find PID
taskkill /PID <PID> /F       # Kill process
```

**Python API error?**
```bash
# Check Python version (need 3.9+)
python --version

# Reinstall dependencies
pip install -r ats_api_requirements.txt --force-reinstall
```

**Frontend not loading?**
```bash
cd client
rm -rf node_modules
npm install
npm run dev
```

---

## 📁 Important Files

| File | Purpose |
|------|---------|
| `ats_api_service.py` | Python AI service |
| `server/src/routes/dashboard.js` | Dashboard APIs |
| `client/src/components/ATSResults.jsx` | ATS visualization |
| `client/src/components/UI.jsx` | UI components |
| `DEPLOYMENT_README.md` | Full documentation |

---

## 🔗 API Endpoints (Quick Reference)

```
Auth:
POST   /api/auth/register
POST   /api/auth/login

Jobs:
GET    /api/jobs
POST   /api/jobs
GET    /api/jobs/:id

Applications:
POST   /api/applications/:jobId
GET    /api/applications/me

Dashboard:
GET    /api/dashboard/recruiter/stats
GET    /api/dashboard/candidate/insights

Python ATS:
POST   http://localhost:8000/analyze
GET    http://localhost:8000/health
```

---

## ✅ Success Checklist

Before demo, verify:
- [ ] All 4 services running (MongoDB, Python, Node, React)
- [ ] Can register new user
- [ ] Can login
- [ ] Recruiter can post job
- [ ] Candidate can upload PDF and apply
- [ ] ATS analysis runs (check Python terminal for logs)
- [ ] Dashboard shows statistics
- [ ] SHAP/LIME/Missing Skills display correctly

---

## 🎯 Demo Script (2 Minutes)

1. **Login** as recruiter (recruiter@demo.com)
2. **Show dashboard** - statistics, top candidates
3. **Post new job** with description
4. **Login** as candidate (candidate@demo.com)
5. **Apply** with sample resume PDF
6. **Show ATS analysis**:
   - ATS Score (e.g., 78%)
   - Job Prediction (e.g., "Data Scientist")
   - SHAP features (top skills impacting prediction)
   - LIME words (most influential keywords)
   - Missing Skills (top 10 to add with +2.3% impact)
7. **Back to recruiter** - show updated dashboard with new application

---

## 💡 Key Talking Points

- **Innovation**: First ATS to quantify skill gap impact with counterfactual analysis
- **Explainability**: Dual XAI methods (SHAP + LIME) for transparency
- **Actionability**: Beyond explanation - quantified recommendations
- **Accuracy**: 96.27% on 45 job roles, 10K+ training samples
- **Trust**: Users understand WHY, not just WHAT the AI predicts

---

## 📚 Documentation Files

1. **DEPLOYMENT_README.md** - Full setup guide
2. **IMPLEMENTATION_SUMMARY.md** - Technical details
3. **This file** - Quick reference

---

**Need help? Check DEPLOYMENT_README.md for detailed troubleshooting!**
