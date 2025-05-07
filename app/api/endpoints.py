import tempfile
import io
import logging
import soundfile as sf
import librosa
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form
from fastapi.responses import JSONResponse
from typing import Optional

from app.core import settings, model_manager
from app.api.schemas import EmotionPrediction, HealthCheck, ErrorResponse
from app.models.feature_extraction import prepare_audio_features

router = APIRouter()
logger = logging.getLogger(__name__)

def validate_audio_file(file: UploadFile) -> bool:
    """Validate that the uploaded file is a valid audio file."""
    # Check file extension
    if file.filename is None:
        return False
        
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in settings.ALLOWED_AUDIO_EXTENSIONS:
        return False
    
    return True

@router.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """Check if the API is running and models are loaded."""
    is_loaded = model_manager.is_loaded
    
    # Determine available models
    available_models = []
    if is_loaded and model_manager.ensemble_model:
        available_models = list(model_manager.ensemble_model.models.keys())
    
    # Get device information
    device = str(model_manager.device)
    
    return HealthCheck(
        status="ok", 
        models_loaded=is_loaded,
        available_models=available_models,
        device=device
    )

@router.post(
    "/predict", 
    response_model=EmotionPrediction,
    responses={400: {"model": ErrorResponse}},
    tags=["prediction"]
)
async def predict_emotion(
    file: UploadFile = File(...),
    sample_rate: Optional[int] = Query(None, description="Sample rate to use for processing. If not provided, will be detected or default to 16000Hz.")
):
    """
    Predict emotion from an uploaded audio file.
    
    - **file**: Audio file to analyze (mp3, wav, ogg, flac, m4a, webm)
    - **sample_rate**: Optional sample rate override
    """
    # Validate the file
    if not validate_audio_file(file):
        raise HTTPException(
            status_code=400, 
            detail="Invalid audio file. Supported formats: mp3, wav, ogg, flac, m4a, webm"
        )
    
    try:
        # Read the uploaded file content
        audio_content = await file.read()
        
        # Convert to in-memory file for reading with librosa
        with io.BytesIO(audio_content) as audio_buffer:
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(suffix=f".{file.filename.split('.')[-1]}", delete=False) as temp_file:
                temp_file.write(audio_content)
                temp_file_path = temp_file.name
            
            try:
                # Load audio using librosa for better compatibility
                logger.info(f"Loading audio file {file.filename}")
                audio_array, detected_sr = librosa.load(
                    temp_file_path, 
                    sr=sample_rate or settings.DEFAULT_SAMPLE_RATE,
                    mono=True
                )
                
                # Prepare features for the model
                logger.info(f"Extracting features from audio file")
                features = prepare_audio_features(
                    audio_array, 
                    detected_sr, 
                    required_features=settings.FEATURE_TYPES
                )
                
                # Get prediction from the model
                logger.info(f"Running prediction on extracted features")
                prediction = await model_manager.predict(features)
                
                return prediction
                
            except Exception as e:
                logger.error(f"Szczegółowy błąd ładowania audio: {e}")
                logger.error(f"Typ pliku: {file.filename}")
                # Zapisz nieprawidłowy plik do analizy
                import shutil
                debug_path = Path("debug_audio")
                debug_path.mkdir(exist_ok=True)
                shutil.copy(temp_file_path, debug_path / f"problematic_{file.filename}")
                raise HTTPException(status_code=500, detail=f"Nie można przetworzyć pliku audio: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

@router.post(
    "/record", 
    response_model=EmotionPrediction,
    responses={400: {"model": ErrorResponse}},
    tags=["prediction"]
)
async def record_and_predict(
    audio_data: bytes = File(..., description="Raw audio data as bytes"),
    sample_rate: int = Form(16000, description="Sample rate of the recorded audio")
):
    """
    Predict emotion from audio data recorded in the browser.
    
    - **audio_data**: Raw audio data as bytes
    - **sample_rate**: Sample rate of the recorded audio
    """
    try:
        # Convert the audio data to a numpy array
        with io.BytesIO(audio_data) as audio_buffer:
            # Create a temporary file to store the audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
                
            try:
                # Load audio using librosa for better compatibility
                logger.info("Loading recorded audio data")
                audio_array, detected_sr = librosa.load(
                    temp_file_path, 
                    sr=sample_rate,
                    mono=True
                )
                
                # Prepare features for the model
                logger.info("Extracting features from recorded audio")
                features = prepare_audio_features(
                    audio_array, 
                    detected_sr, 
                    required_features=settings.FEATURE_TYPES
                )
                
                # Get prediction from the model
                logger.info("Running prediction on extracted features")
                prediction = await model_manager.predict(features)
                
                return prediction
                
            except Exception as e:
                logger.error(f"Szczegółowy błąd ładowania audio: {e}")
                logger.error(f"Typ pliku: {file.filename}")
                # Zapisz nieprawidłowy plik do analizy
                import shutil
                debug_path = Path("debug_audio")
                debug_path.mkdir(exist_ok=True)
                shutil.copy(temp_file_path, debug_path / f"problematic_{file.filename}")
                raise HTTPException(status_code=500, detail=f"Nie można przetworzyć pliku audio: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error processing recorded audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing recorded audio: {str(e)}"
        ) 