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
    MODELS_DIR: Path = Path(__file__).parent.parent / "models" / "saved_models"
    DEFAULT_MODEL_PATH: Path = MODELS_DIR / "melspectrogram_model.pt"
    DEFAULT_STATS_PATH: Path = MODELS_DIR / "melspectrogram_stats.json"
    DEFAULT_FEATURE_TYPE: str = "melspectrogram"
    DEFAULT_SAMPLE_RATE: int = 24000
    MAX_AUDIO_LENGTH_SECONDS: float = 10.0
    EMOTION_LABELS: list[str] = ["anger", "fear", "happiness", "neutral", "sadness", "surprised"]
    
    # Simplified feature types for melspectrogram-only model
    FEATURE_TYPES: list[str] = ["melspectrogram"]
    
    # Audio feature extraction setting
    TARGET_LENGTH = 141
    DEFAULT_SAMPLE_RATE = 24000
    N_MELS: int = 128
    N_MFCC: int = 40
    N_CHROMA: int = 12
    N_FFT: int = 2048
    HOP_LENGTH: int = 512
    
    # Ensemble model weights
    MODEL_WEIGHTS: dict[str, float] = {
        "chroma": 0.13857384163072003,
        "melspectrogram": 0.31651310994496806,
        "mfcc": 0.544913048424312
    }
    
    # File upload settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB in bytes
    ALLOWED_AUDIO_EXTENSIONS: set[str] = {"mp3", "wav", "ogg", "flac", "m4a", "webm"}
    
    # Fast API settings
    WORKERS: int = 2
    
    # Dodatkowe ustawienia dla feature extraction
    TARGET_LENGTH: int = 141  # Długość docelowa dla cech
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 