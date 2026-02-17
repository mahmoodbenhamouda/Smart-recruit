# ✅ TalentBridge Installation Checklist

Use this checklist to ensure everything is set up correctly before running the application.

---

## 📋 Pre-Installation

- [ ] **Node.js installed** (v18+)
  ```bash
  node --version  # Should show v18.0.0 or higher
  ```

- [ ] **Python installed** (3.9+)
  ```bash
  python --version  # Should show 3.9.0 or higher
  ```

- [ ] **MongoDB installed**
  ```bash
  mongosh  # Should connect successfully
  ```

- [ ] **Git installed** (optional, for cloning)
  ```bash
  git --version
  ```

---

## 📦 Installation Steps

### Backend (Node.js)

- [ ] Navigate to server folder
  ```bash
  cd server
  ```

- [ ] Install dependencies
  ```bash
  npm install
  ```

- [ ] Create .env file
  ```bash
  # Copy env.example to .env
  cp env.example .env  # Mac/Linux
  copy env.example .env  # Windows
  ```

- [ ] Configure .env file
  - [ ] Set `MONGODB_URI`
  - [ ] Set `JWT_SECRET` (use a long random string)
  - [ ] Verify `CLIENT_ORIGIN=http://localhost:5173`
  - [ ] Verify `ATS_SERVICE_URL=http://localhost:8000`

### Frontend (React)

- [ ] Navigate to client folder
  ```bash
  cd ../client
  ```

- [ ] Install dependencies
  ```bash
  npm install
  ```

- [ ] Create .env file
  ```bash
  # Copy env.example to .env
  cp env.example .env  # Mac/Linux
  copy env.example .env  # Windows
  ```

- [ ] Verify .env file
  - [ ] `VITE_API_URL=http://localhost:3000/api`

### Python AI Service

- [ ] Navigate to project root
  ```bash
  cd ..
  ```

- [ ] Create virtual environment
  ```bash
  python -m venv venv
  ```

- [ ] Activate virtual environment
  ```bash
  # Windows:
  venv\Scripts\activate
  
  # Mac/Linux:
  source venv/bin/activate
  ```

- [ ] Install Python dependencies
  ```bash
  pip install -r ats_api_requirements.txt
  ```

- [ ] Verify XGBoost model exists
  - [ ] Check: `deep_Learning_Project/JobPrediction_Model/xgboost_model.pkl`
  - [ ] Check: `deep_Learning_Project/JobPrediction_Model/xgb_tfidf_vectorizer.pkl`
  - [ ] Check: `deep_Learning_Project/JobPrediction_Model/xgb_label_encoder.pkl`

---

## 🚀 First Run

- [ ] **Start MongoDB** (if not running as service)
  ```bash
  # Windows:
  net start MongoDB
  
  # Mac:
  brew services start mongodb-community
  
  # Linux:
  sudo systemctl start mongod
  ```

- [ ] **Verify MongoDB is running**
  ```bash
  mongosh
  # Should connect without errors
  # Type: exit
  ```

- [ ] **Start Python ATS API** (Terminal 1)
  ```bash
  # Ensure venv is activated
  python ats_api_service.py
  ```
  - [ ] Should see: "Starting ATS XAI API Service"
  - [ ] Should see: "XGBoost Loaded: True"
  - [ ] Should see: "Uvicorn running on http://0.0.0.0:8000"

- [ ] **Test Python API** (separate terminal)
  ```bash
  curl http://localhost:8000/health
  # Should return: {"status":"ok","service":"ATS XAI API",...}
  ```

- [ ] **Start Node.js Backend** (Terminal 2)
  ```bash
  cd server
  npm run dev
  ```
  - [ ] Should see: "Server running on port 3000"
  - [ ] Should see: "MongoDB connected"

- [ ] **Test Node.js API** (separate terminal)
  ```bash
  curl http://localhost:3000/health
  # Should return: {"status":"ok","timestamp":...}
  ```

- [ ] **Start React Frontend** (Terminal 3)
  ```bash
  cd client
  npm run dev
  ```
  - [ ] Should see: "Local: http://localhost:5173/"
  - [ ] Should see: "ready in [X] ms"

- [ ] **Open browser to http://localhost:5173**
  - [ ] Page should load without errors
  - [ ] Should see "TalentBridge" logo and navigation

---

## ✅ Functional Testing

### Test Authentication

- [ ] **Register new user**
  - [ ] Click "Sign Up"
  - [ ] Fill form (name, email, password, role)
  - [ ] Click "Sign Up" button
  - [ ] Should redirect to dashboard (recruiter) or home (candidate)

- [ ] **Login**
  - [ ] Click "Login"
  - [ ] Enter credentials
  - [ ] Should redirect successfully

- [ ] **Logout**
  - [ ] Click "Logout" button
  - [ ] Should redirect to login page

### Test Recruiter Features

- [ ] **Post a job**
  - [ ] Login as recruiter
  - [ ] Click "+ Post Job"
  - [ ] Fill job details (title, description, location, etc.)
  - [ ] Click "Post Job"
  - [ ] Job should appear in dashboard

- [ ] **View dashboard**
  - [ ] Navigate to "Dashboard"
  - [ ] Should see statistics cards
  - [ ] Should see posted jobs list

### Test Candidate Features

- [ ] **Apply to job**
  - [ ] Login as candidate
  - [ ] Browse jobs on home page
  - [ ] Click on a job
  - [ ] Upload PDF resume
  - [ ] Click "Apply"
  - [ ] Should see success message

- [ ] **View applications**
  - [ ] Navigate to "My Applications"
  - [ ] Should see applied jobs
  - [ ] Should see ATS scores

- [ ] **View ATS analysis**
  - [ ] Click "View Full Analysis" on an application
  - [ ] Should see:
    - [ ] ATS Score (percentage)
    - [ ] Job Prediction
    - [ ] Matched Skills
    - [ ] Missing Skills
    - [ ] SHAP explanation
    - [ ] LIME explanation
    - [ ] Missing Skills Analysis

### Test ATS Pipeline (Most Important!)

- [ ] **Upload sample resume**
  - [ ] Apply to a job with a test PDF resume
  - [ ] Check Python terminal for logs:
    - [ ] Should see "Analyzing [X] features for missing skills..."
    - [ ] Should see "Found [X] missing skills with positive impact"
    - [ ] Should NOT see error messages

- [ ] **Verify ATS results**
  - [ ] Go to recruiter dashboard
  - [ ] Click on job with application
  - [ ] Click on application
  - [ ] Verify all XAI data displays:
    - [ ] ATS score is NOT 0.0%
    - [ ] Job prediction shows role name
    - [ ] SHAP features display
    - [ ] LIME words display
    - [ ] Missing skills show with impact percentages

---

## 🐛 Common Issues & Fixes

### Issue: "MongoDB connection error"
**Fix:**
```bash
# Check MongoDB is running
mongosh

# If not running, start it:
net start MongoDB  # Windows
brew services start mongodb-community  # Mac
sudo systemctl start mongod  # Linux
```

### Issue: "Port 3000 already in use"
**Fix:**
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:3000 | xargs kill -9
```

### Issue: "Python module not found"
**Fix:**
```bash
# Ensure venv is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Reinstall dependencies
pip install -r ats_api_requirements.txt --force-reinstall
```

### Issue: "XGBoost model not found"
**Fix:**
- Verify model files exist in `deep_Learning_Project/JobPrediction_Model/`
- If missing, you need to train the model first (see original project README)

### Issue: "Tailwind styles not working"
**Fix:**
```bash
cd client
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init
npm run dev
```

### Issue: "ATS score always 0.0%"
**Fix:**
- Check Python API is running on port 8000
- Check `ATS_SERVICE_URL=http://localhost:8000` in server/.env
- Check Python terminal for error messages
- Test Python API directly: `curl http://localhost:8000/health`

---

## ✅ Final Verification

All checks passed? You're ready to demo! 🎉

- [ ] All 4 services running (MongoDB, Python, Node, React)
- [ ] Can register and login
- [ ] Recruiter can post jobs
- [ ] Candidate can apply with PDF
- [ ] ATS analysis runs successfully
- [ ] Dashboard shows statistics
- [ ] SHAP, LIME, Missing Skills all display

---

## 📚 Next Steps

1. Read `DEPLOYMENT_README.md` for detailed documentation
2. Read `QUICK_REFERENCE.md` for demo script
3. Read `IMPLEMENTATION_SUMMARY.md` for technical details

---

**Happy demoing! 🚀**
