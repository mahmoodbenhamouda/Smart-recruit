import json
import os
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ATS_DIR = Path(__file__).resolve().parents[2] / "ATS-agent" / "ATS-agent"

sys.path.insert(0, str(ATS_DIR))

try:
    from ats_pipeline import ATSPipeline
    from job_role_predictor import JobRolePredictor
except ImportError as exc:
    print(json.dumps({"success": False, "error": f"Failed to import ATS modules: {exc}"}))
    sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print(json.dumps({"success": False, "error": "Usage: run_ats_pipeline.py <resume.pdf> <job_description.txt>"}))
        sys.exit(1)

    resume_path = Path(sys.argv[1]).resolve()
    job_desc_path = Path(sys.argv[2]).resolve()

    if not resume_path.exists():
        print(json.dumps({"success": False, "error": f"Resume not found: {resume_path}"}))
        sys.exit(1)

    if not job_desc_path.exists():
        print(json.dumps({"success": False, "error": f"Job description not found: {job_desc_path}"}))
        sys.exit(1)

    job_description = job_desc_path.read_text(encoding="utf-8")

    pipeline = ATSPipeline(use_spacy=True)
    results = pipeline.analyze(str(resume_path), job_description, verbose=False, analyze_format=False)

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    silence_buffer = StringIO()
    with redirect_stdout(silence_buffer):
        job_predictor = JobRolePredictor(model_path=str(ATS_DIR / "JobPrediction_Model"))
    resume_keywords = results.get("resume_analysis", {}).get("keywords", {})
    keyword_list = resume_keywords.get("technical_skills") or resume_keywords.get("all_keywords") or []
    skills_text = " ".join(keyword_list[:120])

    if skills_text:
        with redirect_stdout(StringIO()):
            job_prediction = job_predictor.predict_job_role(skills_text, top_n=5)
        results["job_prediction"] = job_prediction
    else:
        results["job_prediction"] = {"predicted_role": None, "top_predictions": []}

    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()

