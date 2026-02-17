# Talent Bridge – Recruiter & Candidate Portal

This workspace now includes a full-stack web application that lets recruiters publish jobs and review AI-assisted applications, while candidates can discover openings, upload resumes, and receive ATS feedback.

## Project Layout

- `server/` – Node.js + Express API (authentication, job postings, applications, ATS integration)
- `client/` – React SPA (Vite) for recruiter & candidate experiences
- `ats_service/` – Legacy FastAPI microservice that hosted the ATS pipeline (now consumed by `resume_service`)
- `resume_service/` – FastAPI microservice wrapping OCR + LayoutLM section detection + personalised feedback, **now also exposing the ATS analysis endpoints with explainability payloads**
- `combined_service/` – Unified FastAPI gateway that exposes both microservices on one Swagger page (optional; resume_service now provides a single Swagger UI by default)
- `ATS-agent/`, `pi5eme1.../`, `deep_Learning_Project/` – existing analysis modules leveraged by the new app

## Prerequisites

- **Node.js** 18+
- **Python** 3.10+ (same environment used for the existing ATS pipeline)
- **MongoDB Atlas** cluster (connection string provided by you)
- Recommended: `pip install -r smart_resume_suite/requirements.txt` (or the consolidated root `requirements.txt`) so the ATS modules load correctly.

## Backend Setup (`server/`)

1. Copy `server/env.example` to `server/.env` and update values:
   ```bash
   MONGODB_URI=mongodb+srv://benhammoudamahmoud00_db_user:kqKH5p2kG9UvU7EZ@cluster0.ohiordc.mongodb.net/?appName=Cluster0
   JWT_SECRET=<generate a strong secret>
   JWT_EXPIRES_IN=2h
   CLIENT_ORIGIN=http://localhost:5173
   # PYTHON_PATH=C:\Users\Mahmoud\miniconda3\python.exe (optional if python is not on PATH)
   ```

2. Install dependencies:
   ```bash
   cd server
   npm install
   ```

3. Start the API (uses port 5000 by default):
   ```bash
   npm run dev   # with Nodemon
   # or
   npm start
   ```

### API Highlights

- `POST /api/auth/register` / `POST /api/auth/login` – JWT-based auth for candidates & recruiters
- `POST /api/jobs` (recruiter) – publish jobs with descriptions & required skills
- `POST /api/applications/:jobId` (candidate) – upload resume (PDF). If `ATS_SERVICE_URL` is set the backend calls the FastAPI microservice; otherwise it falls back to `server/scripts/run_ats_pipeline.py`, which reuses:
  - `ATS-agent/ATS-agent/ats_pipeline.py` for similarity scoring
  - `JobPrediction_Model` for AI job projections
- `GET /api/applications/job/:jobId` (recruiter) – view submissions, ATS match scores, predicted roles, and matched/missing skills
- `GET /api/applications/me` (candidate) – track personal applications & ATS feedback
- `GET /api/applications/:applicationId/resume` – secure resume download (recruiter owner or candidate applicant only)

Uploaded resumes live under `server/uploads/` (git-ignored). Each application stores ATS summaries plus the full raw response for auditability.

## ATS Service (`ats_service/`)

> **Note:** The dedicated FastAPI wrapper is now optional because its endpoints
> are exposed via `resume_service`. Keep this package if you need the legacy
> microservice or for unit testing the pipeline in isolation.

1. Create and activate a Python 3.10+ environment.

   ```bash
   cd ats_service
   python -m venv .venv
   . .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   uvicorn app.main:get_fastapi_app --host 0.0.0.0 --port 8001 --reload
   ```

2. (Optional) Container build:
   ```bash
   docker build -t ats-service .
   docker run -p 8001:8001 ats-service
   ```

3. Point the Node backend at the service by setting:
   ```
   ATS_SERVICE_URL=http://localhost:8001
   ```
   in `server/.env` (also see `ATS_SERVICE_TIMEOUT` for request timeouts).

The FastAPI service loads the ATS models once at startup and returns structured JSON (`overall_match`, `job_prediction`, etc.) consumed by the Express API.

## Resume Intelligence Service (`resume_service/`)

1. Create and activate a Python 3.10+ environment:

   ```powershell
   cd resume_service
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   uvicorn app.main:get_fastapi_app --host 0.0.0.0 --port 8002 --reload
   ```

2. Configure environment variables (PowerShell example):

   ```powershell
   $Env:POPPLER_PATH="C:\path\to\poppler\Library\bin"
   $Env:TESSERACT_CMD="C:\Program Files\Tesseract-OCR\tesseract.exe"
   $Env:GROQ_API_KEY="sk-..."  # optional, enables AI wording
   ```

   The feedback module also needs `deep_Learning_Project/resume_screening_dataset_train.csv`
   to stay in place. Without the Groq key, the endpoint returns deterministic fallback text.

3. Verify the service at `http://localhost:8002/docs`. The Node backend (or any consumer)
   can call:

   - `POST /parse-resume` with `multipart/form-data` (`resume` field expecting a PDF)
   - `POST /feedback` with JSON body `{ "resume_text": "...", "job_description": "...", "user_question": "..." }`
   - `POST /ats/analyze` with the same payload format as the legacy ATS service (`resume` file + `job_description` form field)
   - `GET /ats/health` for ATS pipeline readiness
   - `GET /health` to check an aggregated status for both subsystems
   - All endpoints now return explainability artefacts (section confidence, skill-gap traces, ATS metric breakdowns) to support validation demos.

### Explainability Enhancements

- `parse-resume` now returns per-section `confidence` scores, keyword evidence, and
  `ocr_diagnostics` describing how text was recovered.
- `feedback` surfaces `skill_gap_insights`, `feedback_breakdown`, and the intermediate
  feedback/coaching sections so reviewers can trace each recommendation back to data.
- `ats/analyze` augments its response with an `explanations` object that details
  matched vs missing skill contributions, similarity metric breakdowns, and the keywords
  driving job-role predictions.

Set `RESUME_SERVICE_URL=http://localhost:8002` (or similar) in consuming applications to
route requests to this microservice.

## Unified Gateway (`combined_service/`)

1. Create and activate a Python 3.10+ environment:

   ```powershell
   cd combined_service
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
   ```

2. Browse to `http://localhost:8003/docs`. You will see:

   - `ATS Service` endpoints surfaced under the `/ats` prefix.
   - `Resume Intelligence` endpoints surfaced under the `/resume` prefix.
   - Gateway utilities such as `/health` (aggregated status) and `/`.

Use this gateway when you want a single Swagger UI to exercise both services.

## Frontend Setup (`client/`)

1. (Optional) create `client/.env` from `client/env.example` if you need a custom API base.

2. Install dependencies:
   ```bash
   cd client
   npm install
   ```

3. Run the React dev server (port 5173):
   ```bash
   npm run dev
   ```

   The Vite dev server proxies `/api/*` calls to `http://localhost:5000`, so the client and server cooperate during development.

### SPA Overview

- Candidates: browse jobs, open details, apply with a PDF resume, and monitor ATS feedback in “My Applications”.
- Recruiters: dashboard listing their postings, ability to publish new jobs, and per-job application review (match score, predicted role, matched/missing skills, resume download).
- Authentication state persists via local storage; JWT is attached to API calls automatically.

## Running the ATS Pipeline Manually

The legacy script is still available if you need to debug outside the FastAPI service:

```bash
python server/scripts/run_ats_pipeline.py <path-to-resume.pdf> <path-to-job-description.txt>
```

Ensure the Python environment has all dependencies listed under `requirements.txt` / `smart_resume_suite/requirements.txt`, including spaCy, TensorFlow, transformers, etc.

## Deployment Notes

- For production builds:
  - `cd client && npm run build` generates static assets under `client/dist`.
  - Serve the SPA via a CDN or static host, pointing API calls to the deployed Express server.
  - Run the Express server with `npm start` (ensure `.env` values match production secrets).
- Consider storing resumes in cloud object storage (S3, GCS, etc.) if you scale beyond single-server setups.
- Monitor the Python pipeline: it is compute-intensive the first time models load. Warm the process or cache results for large volumes.

## Next Steps / Extensions

- Add email notifications for new applications or status updates.
- Allow recruiters to add interview feedback and change application status via the UI (`PATCH /api/applications/:id/status` endpoint already exists).
- Surface richer analytics using `atsReport.raw`.

Let me know if you want to tailor the UX further or automate environment provisioning! 

