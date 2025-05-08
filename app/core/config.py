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
    EMOTION_LABELS: list[str] = ["anger", "fear", "happiness", "neutral", "sadness", "surprised"]
    
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
    
    # Model calibration parametrs
    MODEL_CALIBRATION: dict[str, dict[str, float]] = {
        "melspectrogram": {"alpha": 1.2, "beta": 0.0},  # Zmniejszenie pewności modelu melspectrogram
        "mfcc": {"alpha": 1.0, "beta": 0.0},            # Bez zmian dla mfcc
        "chroma": {"alpha": 0.8, "beta": 0.0}           # Zwiększenie pewności modelu chroma
    }
    
    # Korekta bias dla modeli
    BIAS_CORRECTION: dict[str, list[float]] = {
        # Korekta bias dla klas [anger, fear, happiness, neutral, sadness, surprised]
        "ensemble": [-0.1, 0.02, 0.02, 0.03, 0.02, 0.01]  # Zmniejszenie bias dla klasy "anger"
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