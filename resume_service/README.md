# Resume Intelligence FastAPI Service

FastAPI microservice that exposes the resume OCR pipeline (PDF text extraction +
LayoutLM section detection + spaCy skill entities) and the personalised
feedback engine that reuses the SmartRecruiter dataset and Groq prompts.

## Features

- `POST /parse-resume` – accepts a resume PDF (`multipart/form-data`) and returns
  raw text, detected sections, skill entities, OCR flag, and warnings.
- `POST /feedback` – takes `resume_text`, optional `job_description`, and an
  optional `user_question` to generate personalised feedback.
- `POST /ats/analyze` – submit a resume PDF plus job description text for ATS
  similarity scoring and job-role prediction.
- `GET /ats/health` – quick status of the ATS pipeline.
- `GET /health` – aggregated readiness information for both subsystems.

## Explainability Outputs

- **Resume parsing** now returns `confidence` and `evidence` (keywords + snippet)
  for each detected section, along with `ocr_diagnostics` that clarify how text
  was extracted.
- **Feedback generation** surfaces `skill_gap_insights`, `feedback_breakdown`,
  and `intermediate_sections` to map every coaching point back to missing skills
  and retrieved dataset rows.
- **ATS analysis** augments the classic response with an `explanations` payload
  covering matched/missing skill rationales, metric contributions, and the
  keywords that steered the job-role prediction.

## Local Development

```powershell
cd resume_service
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:get_fastapi_app --host 0.0.0.0 --port 8002 --reload
```

Then browse to `http://localhost:8002/docs` for the interactive Swagger UI.
ATS endpoints are namespaced under `/ats` so you can exercise both workflows
from the same page.

### Example Responses

- `POST /parse-resume` highlights the reasoning for each section:
  ```json
  {
    "section_name": "Experience",
    "confidence": 0.86,
    "evidence": {
      "rationale": "Detected 3 domain keywords for experience. Bullet-style formatting reinforces the section structure.",
      "keywords": ["managed", "team", "delivery"],
      "snippet": "Software Engineer – led cross-functional teams delivering..."
    }
  }
  ```
- `POST /feedback` links every feedback bullet to supporting skills and dataset indices.
- `POST /ats/analyze` exposes an `explanations` object detailing skill weights,
  metric contributions, and job prediction keywords.

## Environment Variables

Set these before running the service (either in PowerShell via `$Env:NAME` or in
a `.env` file in the repository root):

- `POPPLER_PATH` – required if OCR fallback should work on scanned PDFs.
- `TESSERACT_CMD` – full path to the Tesseract executable.
- `GROQ_API_KEY` – enables the Groq-powered feedback synthesis (fallback text is
  returned when absent).
- `LAYOUTLM_MODEL_PATH` and `SPACY_SKILL_MODEL_PATH` – override model folders if
  they live outside the default `pi5eme1 - Copie (2)` tree.

The feedback engine also expects the dataset
`deep_Learning_Project/resume_screening_dataset_train.csv`.

## Production Notes

- Mount a persistent volume for the `.tmp/` directory if you want to retain
  uploaded resumes longer than the request lifecycle.
- Point other services (e.g. the Node backend) at `http://<host>:8002` and set
  appropriate timeouts (section detection can take a few seconds on first use).
- Use the `feedback_ready` flag from `/health` to determine whether Groq / the
  dataset loaded successfully.


