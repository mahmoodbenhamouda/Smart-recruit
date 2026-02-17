# TalentBridge - AI-Powered ATS Platform

## 🚀 Full-Stack Recruitment Platform with Explainable AI

TalentBridge is a comprehensive recruitment platform featuring:
- **Role-Based Authentication** (Recruiter & Candidate)
- **AI-Powered ATS Scoring** with SHAP & LIME explainability
- **Job Prediction** using XGBoost (96.27% accuracy, 45 job roles)
- **Missing Skills Analysis** with quantified improvement recommendations
- **Modern UI/UX** with Tailwind CSS and responsive design
- **Real-time Dashboards** for both recruiters and candidates

---

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [API Documentation](#api-documentation)
7. [User Workflow](#user-workflow)
8. [Technology Stack](#technology-stack)

---

## 🏗️ System Architecture

```
┌─────────────────┐
│  React Client   │ (Port 5173)
│  (Vite + Tailwind)│
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│  Node.js API    │ (Port 3000)
│  (Express + JWT)│
└────────┬────────┘
         │
         ├─→ MongoDB (Stores users, jobs, applications)
         │
         └─→ ┌─────────────────┐
             │ Python ATS API  │ (Port 8000)
             │ (FastAPI + XAI) │
             └─────────────────┘
                     │
                     ├─→ XGBoost Model (Job Prediction)
                     ├─→ SHAP (Feature Importance)
                     ├─→ LIME (Word-Level Explanation)
                     └─→ Missing Skills Analysis
```

---

## ✅ Prerequisites

### Required Software

1. **Node.js** (v18 or higher)
   - Download: https://nodejs.org/

2. **Python** (3.9 or higher)
   - Download: https://www.python.org/downloads/

3. **MongoDB** (v5 or higher)
   - Download: https://www.mongodb.com/try/download/community
   - Or use MongoDB Atlas (cloud): https://www.mongodb.com/cloud/atlas

4. **Git**
   - Download: https://git-scm.com/downloads

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd integ
```

### 2. Install Backend Dependencies (Node.js)

```bash
cd server
npm install
cd ..
```

### 3. Install Frontend Dependencies (React)

```bash
cd client
npm install
cd ..
```

### 4. Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r ats_api_requirements.txt
```

---

## ⚙️ Configuration

### 1. MongoDB Setup

**Option A: Local MongoDB**
- Start MongoDB service:
  ```bash
  # Windows (if installed as service):
  net start MongoDB
  
  # Mac:
  brew services start mongodb-community
  
  # Linux:
  sudo systemctl start mongod
  ```

**Option B: MongoDB Atlas (Cloud)**
1. Create account at https://www.mongodb.com/cloud/atlas
2. Create a free cluster
3. Get connection string (e.g., `mongodb+srv://username:password@cluster.mongodb.net/talentbridge`)

### 2. Backend Configuration (.env)

Create `server/.env`:

```env
# Server
PORT=3000
NODE_ENV=development

# MongoDB
MONGODB_URI=mongodb://localhost:27017/talentbridge
# Or for Atlas:
# MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/talentbridge

# JWT
JWT_SECRET=your-super-secret-jwt-key-change-this-in-production
JWT_EXPIRES_IN=7d

# Client
CLIENT_ORIGIN=http://localhost:5173

# ATS Service
ATS_SERVICE_URL=http://localhost:8000
ATS_SERVICE_TIMEOUT=180000
```

### 3. Frontend Configuration (.env)

Create `client/.env`:

```env
VITE_API_URL=http://localhost:3000/api
```

---

## 🚀 Running the Application

### Method 1: Run All Services Separately (Recommended for Development)

#### Terminal 1: MongoDB
```bash
# Already running as service or:
mongod --dbpath /path/to/data
```

#### Terminal 2: Python ATS API
```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux

# Start FastAPI service
python ats_api_service.py
```
Server will start on http://localhost:8000

#### Terminal 3: Node.js Backend
```bash
cd server
npm run dev
```
Server will start on http://localhost:3000

#### Terminal 4: React Frontend
```bash
cd client
npm run dev
```
Client will start on http://localhost:5173

### Method 2: Production Build

#### Backend:
```bash
cd server
npm start
```

#### Frontend:
```bash
cd client
npm run build
npm run preview
```

---

## 👥 User Workflow

### For Recruiters:

1. **Register** as recruiter at http://localhost:5173/register
2. **Post Jobs** with descriptions and requirements
3. **View Dashboard** to see:
   - Total jobs posted
   - Applications received
   - Average ATS scores
   - Top candidates ranked by AI
4. **Review Applications** with:
   - ATS score and match level
   - SHAP feature importance
   - LIME word-level explanations
   - Job role predictions
5. **Manage Candidates** (accept/reject)

### For Candidates:

1. **Register** as candidate at http://localhost:5173/register
2. **Browse Jobs** posted by recruiters
3. **Apply with Resume** (PDF upload)
4. **View Dashboard** to see:
   - Application status
   - Average ATS scores
   - Top matched job role
   - Common missing skills
5. **Get Personalized Feedback**:
   - ATS score breakdown
   - Skills matched vs missing
   - Job prediction alternatives
   - Missing skills analysis with quantified impact
   - Recommendations for improvement

---

## 📊 API Documentation

### Authentication

```
POST /api/auth/register
POST /api/auth/login
```

### Jobs

```
GET    /api/jobs              # List all jobs
GET    /api/jobs/:id          # Get job details
POST   /api/jobs              # Create job (recruiter only)
PUT    /api/jobs/:id          # Update job (recruiter only)
DELETE /api/jobs/:id          # Delete job (recruiter only)
GET    /api/jobs/recruiter/me # Get recruiter's jobs
```

### Applications

```
POST   /api/applications/:jobId              # Apply to job (candidate)
GET    /api/applications/me                  # Get candidate's applications
GET    /api/applications/job/:jobId          # Get applications for job (recruiter)
PATCH  /api/applications/:id/status          # Update status (recruiter)
GET    /api/applications/:id/resume          # Download resume
```

### Dashboard

```
GET /api/dashboard/recruiter/stats   # Recruiter statistics
GET /api/dashboard/recruiter/jobs    # Jobs with application counts
GET /api/dashboard/candidate/insights # Candidate insights
```

### Python ATS API

```
POST /analyze
  - Body: multipart/form-data
    - resume: PDF file
    - job_description: string
    - target_role: string (optional)
  
  - Returns:
    - ATS score and matching
    - Job prediction with confidence
    - SHAP feature importance
    - LIME word explanations
    - Missing skills analysis

GET /health       # Health check
GET /roles        # List available job roles
```

---

## 🛠️ Technology Stack

### Frontend
- **React 18** - UI library
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **React Router** - Routing
- **Recharts** - Data visualization
- **Axios** - HTTP client

### Backend
- **Node.js** - Runtime
- **Express** - Web framework
- **MongoDB** - Database
- **Mongoose** - ODM
- **JWT** - Authentication
- **Multer** - File uploads
- **bcryptjs** - Password hashing

### AI/ML (Python)
- **FastAPI** - API framework
- **XGBoost 2.0+** - Job prediction (96.27% accuracy)
- **SHAP** - Feature importance explainability
- **LIME** - Local interpretable explanations
- **scikit-learn** - TF-IDF vectorization
- **pandas/numpy** - Data processing
- **PyPDF2** - PDF text extraction

---

## 🎯 Key Features

### 1. Explainable AI (XAI)

#### SHAP (SHapley Additive exPlanations)
- Quantifies each feature's contribution to prediction
- Shows positive/negative impact
- Based on game theory principles

#### LIME (Local Interpretable Model-agnostic Explanations)
- Word-level explanations
- Tests 5000 perturbed variations
- Human-readable output

#### Missing Skills Analysis (Innovation ⭐)
- Tests all 5000 features counterfactually
- Quantifies improvement if skill added
- Prioritizes by impact (High/Medium/Low)
- Provides actionable recommendations

### 2. Role-Based Access Control

- **Recruiters**: Post jobs, review applications, see analytics
- **Candidates**: Apply to jobs, track applications, get feedback

### 3. Real-Time Analytics

- Dashboard statistics
- ATS score trends
- Top candidates ranking
- Common missing skills across applications

### 4. Modern UI/UX

- Responsive design (mobile-friendly)
- Tailwind CSS components
- Interactive charts
- Modal dialogs for detailed views
- Progress bars and badges

---

## 🐛 Troubleshooting

### MongoDB Connection Issues
```bash
# Check MongoDB is running
mongosh  # Should connect successfully

# If using Atlas, verify:
# - IP whitelist includes your IP
# - Credentials are correct
# - Connection string format is correct
```

### Python API Not Starting
```bash
# Check Python version
python --version  # Should be 3.9+

# Reinstall dependencies
pip install -r ats_api_requirements.txt --force-reinstall

# Check model files exist
ls deep_Learning_Project/JobPrediction_Model/
# Should see: xgboost_model.pkl, xgb_tfidf_vectorizer.pkl, xgb_label_encoder.pkl
```

### Frontend Build Errors
```bash
# Clear node_modules and reinstall
cd client
rm -rf node_modules package-lock.json
npm install

# If Tailwind not working
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init
```

### Port Already in Use
```bash
# Windows - Kill process on port 3000:
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Mac/Linux:
lsof -ti:3000 | xargs kill -9
```

---

## 📝 Demo Accounts

For quick testing, you can create demo accounts or use these credentials if seeded:

**Recruiter:**
- Email: recruiter@demo.com
- Password: password123

**Candidate:**
- Email: candidate@demo.com
- Password: password123

---

## 🔒 Security Notes

**For Production:**

1. Change `JWT_SECRET` to a strong random string
2. Use environment variables for all secrets
3. Enable HTTPS
4. Set up proper CORS origins
5. Add rate limiting
6. Implement input sanitization
7. Use MongoDB Atlas with proper access controls
8. Enable MongoDB authentication

---

## 📄 License

This project is for educational/demonstration purposes.

---

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs in terminal windows
3. Verify all services are running
4. Check MongoDB connection

---

## 🎉 Success Criteria

Application is working correctly when:
✅ All 4 services running (MongoDB, Python API, Node API, React)
✅ Can register and login
✅ Recruiters can post jobs
✅ Candidates can apply with PDF resume
✅ ATS analysis runs successfully (check Python API logs)
✅ Dashboard shows statistics
✅ XAI explanations display (SHAP, LIME, Missing Skills)

---

**Built with ❤️ using React, Node.js, Python, and MongoDB**
