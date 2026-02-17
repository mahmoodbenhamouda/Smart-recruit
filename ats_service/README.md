## ATS Pipeline FastAPI Service

This service wraps the legacy ATS resume analysis and job prediction pipeline in a FastAPI application, making deployment and horizontal scaling easier. It loads the heavy TensorFlow / transformers models once at startup and exposes a simple REST API for other services.

### Endpoints

- `GET /health` – Basic health probe.
- `POST /analyze` – Accepts a resume PDF (`multipart/form-data`) and a job description string. Returns match scores, matched/missing skills, and job role predictions.

### Local Development

```bash
cd ats_service
python -m venv .venv
. .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:get_fastapi_app --host 0.0.0.0 --port 8001 --reload
```

The service assumes the existing ATS modules live in `../ATS-agent/ATS-agent`. Override via the `ATS_AGENT_PATH` environment variable if needed. You can also override `JOB_PREDICTION_MODEL_PATH` to point at a different `JobPrediction_Model` folder.

### Docker Build

```bash
cd ats_service
docker build -t ats-service .
docker run -p 8001:8001 ats-service
```

### Environment Variables

- `ATS_SERVICE_HOST` / `ATS_SERVICE_PORT` – override default bind address and port.
- `ATS_AGENT_PATH` – path to the ATS-agent package (default relative to repo).
- `JOB_PREDICTION_MODEL_PATH` – explicit path to the job prediction model directory.

### Integration

The Node backend reads `ATS_SERVICE_URL` from its `.env`. When set (e.g. `http://localhost:8001` or the deployed URL), it calls this FastAPI service via HTTP. If the variable is omitted, the backend falls back to the previous behaviour of spawning the Python script locally.

