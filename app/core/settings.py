from pathlib import Path
import os

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Project base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # Environment variable loading configuration
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # API Settings
    API_TITLE: str = "Emotion Recognition API"
    API_DESCRIPTION: str = "API for predicting emotions from audio samples"
    API_VERSION: str = "1.0.0"
    OPENAPI_URL: str = "/openapi.json"
    DOCS_URL: str = "/docs"
    API_V1_STR: str = "/api/v1"

    # Environment Settings
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "True").lower() in ("true", "1", "t"))
    ENVIRONMENT: str = Field(default="development")

    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = Field(default_factory=lambda: [
        "http://localhost:3000",
        "http://localhost:8000"
    ])

    # Path Settings
    MODEL_DIR: Path = Field(
        default_factory=lambda: Path(os.getenv("MODEL_DIR", BASE_DIR / "model_outputs"))
    )
    PRETRAINED_MODEL_PATH: Path = Field(
        default_factory=lambda: Path(os.getenv("PRETRAINED_MODEL_PATH",
                                              BASE_DIR / "model_outputs" / "best_model.pt"))
    )

    # Audio Processing Settings
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024   # 50 MB
    DEFAULT_SAMPLE_RATE: int = 22050          # Hz
    MAX_AUDIO_LENGTH: float = 3.0             # seconds
    ALLOWED_AUDIO_EXTENSIONS: set[str] = Field(
        default_factory=lambda: {"mp3", "wav", "ogg", "flac", "m4a", "webm"}
    )

    # Model Settings
    CLASS_NAMES: list[str] = Field(
        default_factory=lambda: ["anger", "fear", "happiness", "neutral", "sadness", "surprised"]
    )

    # MEL Spectrogram Settings
    HOP_LENGTH: int = 512
    N_FFT: int = 2048
    N_MELS: int = 128


    # GPU Settings
    USE_CUDA: bool = Field(default_factory=lambda: os.getenv("USE_CUDA", "True").lower() in ("true", "1", "t"))

# Singleton configuration instance
settings = Settings()
