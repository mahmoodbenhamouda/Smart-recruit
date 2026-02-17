"""
Centralised configuration helpers for the Smart Resume Suite.

The configuration values are primarily driven by environment variables so the
application can be customised without touching the source code.  Sensible
fallbacks are provided where possible so that the multi-page Streamlit app can
run in a minimal environment, while still supporting advanced setups that rely
on Poppler, Tesseract, or hosted LLM providers.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# -----------------------------
# Paths
# -----------------------------

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from the project root .env, falling back to the
# default resolver so system/environment variables still take precedence.
load_dotenv(WORKSPACE_ROOT / ".env")
load_dotenv()
DATA_DIR = WORKSPACE_ROOT / "data"
MODELS_DIR = WORKSPACE_ROOT / "models"
TEMP_DIR = WORKSPACE_ROOT / ".tmp"

TEMP_DIR.mkdir(exist_ok=True)

# -----------------------------
# External tool configuration
# -----------------------------

POPPLER_BIN: Optional[str] = os.getenv("POPPLER_PATH")
TESSERACT_CMD: Optional[str] = os.getenv("TESSERACT_CMD")

# -----------------------------
# API Keys (optional features)
# -----------------------------

GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")

# -----------------------------
# Model paths
# -----------------------------

LAYOUTLM_MODEL_PATH = os.getenv(
    "LAYOUTLM_MODEL_PATH", str(MODELS_DIR / "layoutlmv3-resume")
)
SPACY_SKILL_MODEL_PATH = os.getenv(
    "SPACY_SKILL_MODEL_PATH", str(MODELS_DIR / "ner_model")
)

# -----------------------------
# Utility helpers
# -----------------------------


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_poppler_path() -> Optional[Path]:
    """Return the configured Poppler binary folder if available."""
    if not POPPLER_BIN:
        return None
    poppler_path = Path(POPPLER_BIN)
    return poppler_path if poppler_path.exists() else None


def resolve_tesseract_cmd() -> Optional[Path]:
    """Return the configured Tesseract binary if available."""
    if not TESSERACT_CMD:
        return None
    cmd_path = Path(TESSERACT_CMD)
    return cmd_path if cmd_path.exists() else None

