from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = Field("0.0.0.0", env="ATS_SERVICE_HOST")
    port: int = Field(8001, env="ATS_SERVICE_PORT")
    reload: bool = Field(False, env="ATS_SERVICE_RELOAD")
    log_level: str = Field("info", env="ATS_SERVICE_LOG_LEVEL")
    ats_agent_path: Path = Field(
        Path(__file__).resolve().parents[2] / "ATS-agent" / "ATS-agent",
        env="ATS_AGENT_PATH"
    )
    job_prediction_model_path: Optional[Path] = Field(
        None,
        env="JOB_PREDICTION_MODEL_PATH",
        description="Override path to JobPrediction_Model directory"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

