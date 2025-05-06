from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Audio Emotion Detection"
    DEBUG: bool = False
    
    # CORS settings
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:8000"]
    
    # Model settings
    MODELS_DIR: Path = Path("app/models/saved_models")
    DEFAULT_SAMPLE_RATE: int = 16000
    MAX_AUDIO_LENGTH_SECONDS: float = 10.0
    EMOTION_LABELS: list[str] = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    
    # Audio feature extraction settings
    FEATURE_TYPES: list[str] = ["melspectrogram", "mfcc", "chroma"]
    N_MELS: int = 128
    N_MFCC: int = 40
    N_CHROMA: int = 12
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    
    # Ensemble model weights
    MODEL_WEIGHTS: dict[str, float] = {
        "melspectrogram": 0.4,
        "mfcc": 0.4,
        "chroma": 0.2,
    }
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB in bytes
    ALLOWED_AUDIO_EXTENSIONS: set[str] = {"mp3", "wav", "ogg", "flac", "m4a", "webm"}
    
    # Fast API settings
    WORKERS: int = 2
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 