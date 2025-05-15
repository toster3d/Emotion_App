from pathlib import Path
import os

class Settings:
    # API Settings
    API_TITLE = "Emotion Recognition API"
    API_DESCRIPTION = "API for predicting emotions from audio samples"
    API_VERSION = "1.0.0"
    OPENAPI_URL = "/openapi.json"
    DOCS_URL = "/docs"

    # Environment Settings
    DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

    # Path Settings
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODEL_DIR = os.getenv("MODEL_DIR", BASE_DIR / "model_outputs")
    PRETRAINED_MODEL_PATH = os.getenv("PRETRAINED_MODEL_PATH", MODEL_DIR / "best_model.pt")

    # Audio Processing Settings
    MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
    DEFAULT_SAMPLE_RATE = 22050  # Hz
    MAX_AUDIO_LENGTH = 3.0  # seconds
    ALLOWED_AUDIO_EXTENSIONS = {"mp3", "wav", "ogg", "flac", "m4a", "webm"}

    # Model Settings
    CLASS_NAMES = ["anger", "fear", "happiness", "neutral", "sadness", "surprised"]

    # MEL Spectrogram Settings
    HOP_LENGTH = 512
    N_FFT = 2048
    N_MELS = 128
    N_MFCC = 40
    N_CHROMA = 12

    # GPU Settings
    USE_CUDA = os.getenv("USE_CUDA", "True").lower() in ("true", "1", "t")

settings = Settings()