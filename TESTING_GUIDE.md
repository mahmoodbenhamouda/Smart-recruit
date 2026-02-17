# 🧪 ATS System Testing Guide

## ✅ Current Status

**Python API**: ✅ **RUNNING** on port 8000
- SimilarityCalculator loaded
- XGBoost model with 45 job roles loaded
- SHAP/LIME explainers initialized

**Node.js Backend**: Should be running on port 5000
**React Frontend**: Should be running on port 5173

---

## 🎯 Why You See "ATS analysis not available"

Applications submitted **before** the Python API was running don't have ATS data in the database. You have **2 options**:

### **Option 1: Submit NEW Application (Recommended)**

This is the fastest way to test the complete flow.

#### **Steps:**

1. **Verify Python API is running**:
   - Check terminal where you ran: `python ats_api_service.py`
   - You should see: `INFO: Uvicorn running on http://0.0.0.0:8000`

2. **Open browser as Candidate**:
   ```
   http://localhost:5173
   ```

3. **Register/Login as candidate** (if not already logged in)

4. **Apply to a job**:
   - Click "Apply Now" on any job card
   - Upload a PDF resume (drag & drop or click to upload)
   - Click "Submit Application"

5. **Check Node.js logs** (in server terminal):
   ```
   📄 Processing resume: [filename]
   🚀 Calling ATS service at http://localhost:8000/analyze
   ✅ ATS analysis completed successfully
   📊 Results preview: { overallMatch: 75.5, predictedRole: 'Data Scientist' }
   💾 Saving application with ATS data
   ```

6. **View results**:
   - Go to "My Applications"
   - You should see:
     - ATS Score > 0% (e.g., 75.5%)
     - Predicted Role (e.g., "Data Scientist")
   - Click "View Full Analysis" to see:
     - SHAP feature importance chart
     - LIME explanation chart
     - Matched skills badges
     - Missing skills badges
     - Personalized feedback section
   - Click "Get Career Guidance" to see:
     - 30-day action plan
     - Learning resources
     - Alternative career paths

7. **Check Recruiter View**:
   - Logout and login as recruiter (the one who posted the job)
   - Go to "My Jobs"
   - Click on the job you applied to
   - View "Applications" tab
   - You should see:
     - Ranked candidates (🥇🥈🥉)
     - ATS scores with progress bars
     - Matched/missing skills
     - Full XAI analysis modal

---

### **Option 2: Re-analyze OLD Applications**

If you want to fix existing applications instead of creating new ones.

#### **Steps:**

1. **Verify Python API is running** (port 8000)

2. **Login as Recruiter**

3. **Go to Migration Page**:
   ```
   http://localhost:5173/admin/migrate
   ```

4. **Click "Start Re-analysis"**

5. **Wait for completion** (may take a few minutes)
   - Total applications found
   - Number processed successfully
   - Number failed (with error details)

6. **Check updated applications**:
   - Go to "My Jobs" → Click job → View applications
   - ATS scores should now be visible

#### **What the Migration Does:**

- Finds all applications with:
  - No `atsReport` field
  - `atsReport.overallMatch = 0`
  - Missing ATS data
- Re-uploads resume to Python API
- Gets fresh ATS analysis
- Saves `parsedCV` and `atsReport` to MongoDB
- Updates application in database

---

## 🐛 Troubleshooting

### **Issue 1: Python API Not Running**

**Symptom**: Node.js logs show "Error calling ATS service: connect ECONNREFUSED"

**Solution**:
```powershell
cd C:\Users\Mahmoud\Desktop\integ
.\venv\Scripts\Activate.ps1
python ats_api_service.py
```

**Verify**:
```
✅ Loaded SimilarityCalculator from ATS-agent
✅ Loaded XGBoost model and vectorizer (45 classes)
✅ Explainers initialized successfully
INFO: Uvicorn running on http://0.0.0.0:8000
```

---

### **Issue 2: Port 8000 Already in Use**

**Symptom**: `ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000)`

**Solution**:
```powershell
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace XXXX with actual PID)
taskkill /F /PID XXXX

# Restart Python API
.\venv\Scripts\Activate.ps1
python ats_api_service.py
```

---

### **Issue 3: Still Showing 0% After New Application**

**Check 1 - Node.js Logs**:
Look for these messages in server terminal:
```
📄 Processing resume: [filename]
🚀 Calling ATS service...
✅ ATS analysis completed successfully
```

**If missing**: Backend might not be calling Python API
- Check if `http://localhost:8000` is accessible
- Run: `curl http://localhost:8000/health` (should return `{"status":"ok"}`)

**Check 2 - Python API Logs**:
Look for incoming requests:
```
INFO: 127.0.0.1:XXXXX - "POST /analyze HTTP/1.1" 200 OK
```

**If missing**: Node.js isn't calling Python API
- Check `server/src/services/atsService.js`
- Verify API URL is `http://localhost:8000/analyze`

**Check 3 - MongoDB**:
```javascript
// In MongoDB Compass or shell
db.applications.findOne({}, { atsReport: 1, parsedCV: 1 })
```

Should show:
```javascript
{
  atsReport: {
    overallMatch: 75.5,
    matchLevel: "Excellent",
    matchedSkills: [...],
    missingSkills: [...],
    jobPrediction: { predictedRole: "...", confidence: 0.85, ... }
  },
  parsedCV: {
    rawText: "...",
    skills: [...],
    ...
  }
}
```

---

### **Issue 4: Migration Fails**

**Symptom**: Migration page shows errors

**Common Causes**:
1. Python API not running
2. Resume file not found in `server/uploads/`
3. Job deleted but application still exists

**Solution**:
- Check Python API status (port 8000)
- Verify files exist: `ls C:\Users\Mahmoud\Desktop\integ\server\uploads\`
- Check error details in migration results
- Re-run migration after fixing issues

---

### **Issue 5: Frontend Shows "undefined" or null**

**Symptom**: UI displays "undefined" or empty fields

**Check**:
1. Open browser DevTools (F12)
2. Go to Console tab
3. Look for errors
4. Check Network tab for API calls

**Common Issue**: Old applications in state/cache
- Hard refresh: `Ctrl + Shift + R`
- Clear browser cache
- Logout and login again

---

## 📊 Expected Results

### **For Candidates:**

**My Applications Page**:
- ✅ Application card shows:
  - Job title and company
  - Application status
  - **ATS Score** (e.g., 75.5%) with colored badge
  - **Predicted Role** (e.g., "Data Scientist")
  - "View Full Analysis" button
  - "Get Career Guidance" button

**Full Analysis Modal**:
- ✅ ATS Score with circular progress
- ✅ Match Level (Excellent/Good/Medium/Low)
- ✅ Predicted Job Role with confidence
- ✅ SHAP Feature Importance chart
- ✅ LIME Explanation chart
- ✅ Matched Skills (green badges)
- ✅ Missing Skills (orange badges)
- ✅ Personalized Feedback section:
  - Score Assessment
  - Recommended Actions (Skills, Resume, Interview)
  - Alternative Career Paths
  - Learning Resources

**Career Guidance Page**:
- ✅ Match Score display
- ✅ Your Strengths (categorized by type)
- ✅ Skill Development Plan (categorized missing skills)
- ✅ Career Path Recommendations (top 6)
- ✅ 30-Day Action Plan (weekly breakdown)
- ✅ Learning Platforms (6 cards)

### **For Recruiters:**

**Job Applications Page**:
- ✅ Candidates ranked by ATS score
- ✅ Rank badges (🥇🥈🥉)
- ✅ ATS score with progress bar
- ✅ Match percentage display
- ✅ Matched Skills count
- ✅ Missing Skills count
- ✅ "View Full Analysis" button
- ✅ Accept/Reject actions

**XAI Analysis Modal**:
- ✅ All fields from candidate view
- ✅ Parsed CV sections:
  - Personal Info
  - Summary
  - Experience
  - Education
  - Skills
  - Certifications
  - Projects
  - Languages

---

## 🔍 Test Checklist

### **Before Testing**:
- [ ] Python API running (port 8000)
- [ ] Node.js backend running (port 5000)
- [ ] React frontend running (port 5173)
- [ ] MongoDB Atlas connected
- [ ] At least 1 job posted by recruiter
- [ ] Sample PDF resume ready

### **Test 1: New Application Flow**:
- [ ] Login as candidate
- [ ] Navigate to job detail page
- [ ] Upload PDF resume
- [ ] Submit application
- [ ] Check Node.js logs for "✅ ATS analysis completed"
- [ ] Go to "My Applications"
- [ ] Verify ATS score > 0%
- [ ] Click "View Full Analysis"
- [ ] Verify all XAI charts display
- [ ] Click "Get Career Guidance"
- [ ] Verify 30-day plan displays

### **Test 2: Recruiter View**:
- [ ] Logout, login as recruiter
- [ ] Go to "My Jobs"
- [ ] Click on job with applications
- [ ] Verify candidates ranked by score
- [ ] Verify matched/missing skills shown
- [ ] Click "View Full Analysis"
- [ ] Verify XAI modal displays
- [ ] Verify parsed CV sections display

### **Test 3: Migration (Optional)**:
- [ ] Login as recruiter
- [ ] Navigate to `/admin/migrate`
- [ ] Click "Start Re-analysis"
- [ ] Wait for completion
- [ ] Check results (successful/failed)
- [ ] Go to applications page
- [ ] Verify old applications now have scores

---

## 📝 Sample Test Resume

If you don't have a test resume, create a simple PDF with:

```
John Doe
Email: john.doe@email.com
Phone: +1-234-567-8900

SUMMARY
Experienced Data Scientist with 5 years in machine learning and AI.

EXPERIENCE
Data Scientist | TechCorp | 2020-2024
- Built predictive models using Python and scikit-learn
- Deployed ML models to production with 95% accuracy
- Worked with large datasets (10M+ records)

EDUCATION
Master of Science in Data Science | University XYZ | 2020
Bachelor of Science in Computer Science | University ABC | 2018

SKILLS
Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, scikit-learn,
Pandas, NumPy, SQL, MongoDB, Docker, AWS, Git, Data Visualization, NLP,
Computer Vision, Statistics, A/B Testing

CERTIFICATIONS
- AWS Certified Machine Learning Specialist
- Google Data Analytics Professional Certificate

PROJECTS
1. Customer Churn Prediction System
   Technologies: Python, XGBoost, Flask, Docker
   
2. Image Classification API
   Technologies: TensorFlow, Keras, FastAPI, AWS
```

---

## 🎯 Success Criteria

You'll know everything is working when:

1. ✅ **New applications** show ATS score > 0%
2. ✅ **Matched/missing skills** arrays are populated (not empty)
3. ✅ **Job prediction** shows predicted role with confidence
4. ✅ **SHAP chart** displays feature importance bars
5. ✅ **LIME chart** displays explanation values
6. ✅ **Personalized feedback** shows actionable recommendations
7. ✅ **Career guidance** shows 30-day action plan
8. ✅ **Recruiter view** ranks candidates by score
9. ✅ **CV sections** are parsed and displayed

---

## 🚀 Quick Start (Copy-Paste)

**Terminal 1 - Python API**:
```powershell
cd C:\Users\Mahmoud\Desktop\integ
.\venv\Scripts\Activate.ps1
python ats_api_service.py
```

**Terminal 2 - Node.js Backend**:
```powershell
cd C:\Users\Mahmoud\Desktop\integ\server
npm run dev
```

**Terminal 3 - React Frontend**:
```powershell
cd C:\Users\Mahmoud\Desktop\integ\client
npm run dev
```

**Test URLs**:
- Frontend: http://localhost:5173
- Backend API: http://localhost:5000/health
- Python API: http://localhost:8000/health
- Migration Page: http://localhost:5173/admin/migrate

---

## 📞 Need Help?

If you're still seeing issues after following this guide:

1. **Check all 3 services are running** (Python, Node, React)
2. **Look at terminal logs** for error messages
3. **Check browser DevTools** (F12) for errors
4. **Verify MongoDB connection** in Node.js logs
5. **Try the migration tool** to re-analyze old applications
6. **Submit a new application** to test fresh flow

Remember: **Old applications won't have scores** until you either:
- Re-analyze them using the migration tool, OR
- Submit new applications (recommended)
