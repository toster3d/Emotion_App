import tempfile
import uuid
import logging
from pathlib import Path

import librosa  # type: ignore
import numpy as np

from numpy.typing import NDArray
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form

from app.core.settings import settings
from app.core.model_manager import model_manager
from app.api.schemas import EmotionPrediction, HealthCheck, ErrorResponse
from app.models.feature_extraction import AudioFeatures, prepare_audio_features
from app.core.model_manager import PredictionResult

router = APIRouter()
logger = logging.getLogger(__name__)


def validate_audio_file(file: UploadFile) -> bool:
    if not file.filename:
        return False
    ext = file.filename.rsplit(".", 1)[-1].lower()
    return ext in settings.ALLOWED_AUDIO_EXTENSIONS


@router.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check() -> HealthCheck:
    if not model_manager.is_loaded:
        return HealthCheck(
            status="error",
            models_loaded=False,
            available_models=[],
            device="N/A"
        )
    return HealthCheck(
        status="ok",
        models_loaded=model_manager.is_loaded,
        available_models=["melspectrogram"],
        device=str(model_manager.device)
    )


@router.post(
    "/predict",
    response_model=EmotionPrediction,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    },
    tags=["prediction"]
)
async def predict_emotion(
    file: UploadFile = File(..., description="Audio file to analyze"),
    sample_rate: int | None = Query(
        None,
        description="Sample rate override (8000–48000 Hz)",
        ge=8000,
        le=48000
    )
) -> EmotionPrediction:
    if not model_manager.is_loaded:
        raise HTTPException(503, "Model service not initialized.")
    # if not validate_audio_file(file):
    #     raise HTTPException(400, f"Unsupported audio format. Allowed: {settings.ALLOWED_AUDIO_EXTENSIONS}")
    
    filename = file.filename
    
    if not filename:
        raise HTTPException(400, "Missing filename in upload")
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported audio format: .{ext}. Allowed: {settings.ALLOWED_AUDIO_EXTENSIONS}"
        )
    
    # Wczytaj plik do pamięci
    content = await file.read()
    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(400, f"File too large ({settings.MAX_UPLOAD_SIZE/1e6}MB max)")

    # Zapisz tymczasowo na dysku, by librosa mogło odczytać format
    suffix = f".{ext}"
    tmp_path: str
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        audio_array: NDArray[np.float32]
        detected_sr: int
        audio_array, detected_sr = librosa.load( # type: ignore
            tmp_path,
            sr=sample_rate or settings.DEFAULT_SAMPLE_RATE,
            mono=True
        )

        # Wyciągnij cechy
        features: AudioFeatures = prepare_audio_features(audio_array, detected_sr)

        # Predykcja
        prediction_dict: PredictionResult = model_manager.predict(
            features["melspectrogram"]
        )

        # Konwersja do Pydantic
        prediction = EmotionPrediction(**prediction_dict)
        return prediction

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing /predict: {e}")
        raise HTTPException(500, f"Cannot process audio file: {e}")
    finally:
        try:
            Path(tmp_path).unlink()
        except Exception:
            logger.warning(f"Could not delete temp file {tmp_path}")


@router.post(
    "/record",
    response_model=EmotionPrediction,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["prediction"]
)
async def record_and_predict(
    audio_data: bytes = File(..., description="Raw audio bytes"),
    sample_rate: int = Form(settings.DEFAULT_SAMPLE_RATE, description="Sample rate of recording")
) -> EmotionPrediction:
    if not model_manager.is_loaded:
        raise HTTPException(503, "Model service not initialized.")
    if len(audio_data) < 1000:
        raise HTTPException(400, "Recorded audio too short.")

    temp_dir = Path("temp_audio")
    temp_dir.mkdir(exist_ok=True)
    tmp_file = temp_dir / f"rec_{uuid.uuid4()}.wav"
    tmp_file.write_bytes(audio_data)

    try:
        audio_array: NDArray[np.float32]
        detected_sr: int
        audio_array, detected_sr = librosa.load(tmp_file, sr=sample_rate, mono=True) # type: ignore

        features: AudioFeatures = prepare_audio_features(audio_array, detected_sr)
        prediction_dict: PredictionResult = model_manager.predict(
            features["melspectrogram"]
        )
        prediction = EmotionPrediction(**prediction_dict)
        return prediction

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing /record: {e}")
        raise HTTPException(500, f"Cannot process recorded audio: {e}")

    finally:
        try:
            tmp_file.unlink()
            logger.info(f"Removed temp file {tmp_file}")
        except Exception:
            logger.warning(f"Failed to remove temp file {tmp_file}")
