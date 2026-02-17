# Talent Bridge AI Gateway

Single FastAPI application that combines the ATS analysis service and the resume
intelligence service into one Swagger UI. Endpoints are namespaced so you can
test everything from a single `/docs` page.

## Run locally

```powershell
cd combined_service
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8003 --reload
```

Then open `http://localhost:8003/docs` and you will see:

- `ATS Service` group (`/ats/health`, `/ats/analyze`)
- `Resume Intelligence` group (`/resume/parse-resume`, `/resume/feedback`, `/resume/health`)
- `Gateway` probes (`/health`, `/`)

## Environment

The gateway relies on the same dependencies and assets as the underlying
services:

- ATS models referenced in `ats_service/app/pipeline.py`
- LayoutLM + spaCy models and the dataset under `pi5eme1 - Copie (2)` and `deep_Learning_Project`
- Optional environment variables (`POPPLER_PATH`, `TESSERACT_CMD`, `GROQ_API_KEY`, etc.)

Make sure they are configured before launching the gateway so both service
sections report healthy status.


