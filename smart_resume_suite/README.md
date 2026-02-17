# Smart Resume Suite

Unified Streamlit multi-page experience that chains together the three legacy
projects that were previously separate:

1. **CV Parser** (`pi5eme1`) – OCR, LayoutLMv3 section detection, spaCy skills.
2. **ATS Analyzer** (`ATS-agent`) – keyword extraction, similarity scoring, RAG.
3. **SmartRecruiter Feedback** (`deep_Learning_Project`) – dataset-driven advice and Groq LLM prompts.

## Getting Started

1. Create (or update) a Python environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
   ```

2. (Optional) Configure external tools and API keys in a `.env` file at the
   repository root:

   ```env
   POPPLER_PATH=C:\path\to\poppler\Library\bin
   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   GROQ_API_KEY=...
   GOOGLE_API_KEY=...  # Enables Gemini in the ATS module if desired
   ```

3. Launch the multi-page Streamlit app:

   ```bash
   streamlit run smart_resume_suite/main.py
   ```

4. Use the sidebar navigation to follow the three-step workflow.

## Directory Layout

- `smart_resume_suite/` – New integration package and Streamlit pages.
- `ATS-agent/`, `pi5eme1.../`, `deep_Learning_Project/` – Legacy codebases kept intact (referenced by the suite).
- `requirements.txt` – Consolidated dependency list for the unified project.

## Notes

- OCR is triggered automatically when PyPDF2 extraction returns very little text.
- LayoutLMv3 models load from `models/layoutlmv3-resume` by default; override via environment variable if needed.
- Groq-based feedback is optional; when the API key is missing the app falls back to deterministic messaging.

