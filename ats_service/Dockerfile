# syntax=docker/dockerfile:1.6

ARG PYTHON_VERSION=3.10

FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.8.3

WORKDIR /service

# System deps for spaCy / transformers (poppler/tesseract optional if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_md

COPY app ./app
COPY ../ATS-agent ./ATS-agent

ENV ATS_AGENT_PATH=/service/ATS-agent/ATS-agent

EXPOSE 8001

CMD ["uvicorn", "app.main:get_fastapi_app", "--host", "0.0.0.0", "--port", "8001"]

