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
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["prediction"]
)
async def predict_emotion(
    file: UploadFile = File(..., description="Audio file to analyze"),
    sample_rate: Optional[int] = Query(
        None, 
        description="Sample rate to use for processing. If not provided, will be detected or default to 24000Hz.",
        ge=8000,
        le=48000
    )
):
    """
    Predict emotion from an uploaded audio file.
    
    - **file**: Audio file to analyze (mp3, wav, ogg, flac, m4a, webm)
    - **sample_rate**: Optional sample rate override (8000-48000 Hz)
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model service is not ready. Please try again in a few moments."
        )
    
    # Validate the file
    if not validate_audio_file(file):
        raise HTTPException(
            status_code=400, 
            detail="Invalid audio file. Supported formats: mp3, wav, ogg, flac, m4a, webm"
        )
    
    try:
        # Read the uploaded file content
        audio_content = await file.read()
        
        if len(audio_content) > settings.MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_SIZE/1024/1024}MB"
            )
        
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
                
                logger.info(f"Audio loaded successfully: shape={audio_array.shape}, sr={detected_sr}")
                
                # Prepare features for the model
                logger.info(f"Extracting features from audio file")
                features = prepare_audio_features(
                    audio_array, 
                    detected_sr, 
                    required_features=settings.FEATURE_TYPES
                )
                
                # Log tensor shapes for debugging
                for feature_type, tensor in features.items():
                    logger.info(f"Feature {feature_type} shape: {tensor.shape}")
                
                # Get prediction from the model
                logger.info(f"Running prediction on extracted features")
                prediction = await model_manager.predict(features)
                
                # Log prediction results
                logger.info(f"Prediction result: {prediction['emotion']} with confidence {prediction['confidence']}")
                
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
    sample_rate: int = Form(24000, description="Sample rate of the recorded audio")
):
    """
    Predict emotion from audio data recorded in the browser.
    
    - **audio_data**: Raw audio data as bytes
    - **sample_rate**: Sample rate of the recorded audio
    """
    try:
        logger.info(f"Processing recorded audio: sample_rate={sample_rate}, data size={len(audio_data)} bytes")
        
        if len(audio_data) < 1000:
            logger.warning(f"Otrzymano bardzo mały plik audio ({len(audio_data)} bajtów), może być nieprawidłowy")
            raise HTTPException(
                status_code=400,
                detail="Plik audio jest zbyt mały, nagraj dłuższą próbkę"
            )
        
        # Utwórz folder tymczasowy, jeśli nie istnieje
        temp_folder = Path("temp_audio")
        temp_folder.mkdir(exist_ok=True)
        
        # Generuj unikalną nazwę pliku
        import uuid
        temp_filename = f"recording_{uuid.uuid4()}.wav"
        temp_file_path = temp_folder / temp_filename
        
        try:
            # Zapisz plik audio
            with open(temp_file_path, "wb") as f:
                f.write(audio_data)
            
            logger.info(f"Recorded audio saved to temporary file: {temp_file_path}")
            
            try:
                # Sprawdź format nagrania
                try:
                    import wave
                    with wave.open(str(temp_file_path), "rb") as wave_file:
                        wave_params = wave_file.getparams()
                        logger.info(f"Wave file parameters: channels={wave_params.nchannels}, "
                                   f"sampwidth={wave_params.sampwidth}, framerate={wave_params.framerate}")
                except Exception as wave_err:
                    logger.warning(f"Nie można odczytać jako plik WAV. Próba użycia librosa: {wave_err}")
                
                # Load audio using librosa for better compatibility
                logger.info("Loading recorded audio data")
                audio_array, detected_sr = librosa.load(
                    temp_file_path, 
                    sr=sample_rate,
                    mono=True
                )
                
                logger.info(f"Audio loaded successfully: shape={audio_array.shape}, sr={detected_sr}")
                
                # Check if audio is empty or too short
                if len(audio_array) < 100:
                    raise ValueError("Nagranie jest zbyt krótkie lub puste")
                
                # Prepare features for the model
                logger.info("Extracting features from recorded audio")
                features = prepare_audio_features(
                    audio_array, 
                    detected_sr, 
                    required_features=settings.FEATURE_TYPES
                )
                
                # Log tensor shapes for debugging
                for feature_type, tensor in features.items():
                    logger.info(f"Feature {feature_type} shape: {tensor.shape}")
                
                # Get prediction from the model
                logger.info("Running prediction on extracted features")
                prediction = await model_manager.predict(features)
                
                # Log prediction results
                logger.info(f"Prediction result: {prediction['emotion']} with confidence {prediction['confidence']}")
                
                # Usuń plik tymczasowy po zakończeniu analizy
                try:
                    if temp_file_path.exists():
                        temp_file_path.unlink()
                        logger.info(f"Temporary file removed: {temp_file_path}")
                except Exception as e:
                    logger.warning(f"Nie można usunąć pliku tymczasowego: {e}")
                
                return prediction
                
            except Exception as e:
                logger.error(f"Szczegółowy błąd ładowania audio: {e}")
                logger.error("Typ pliku: nagranie z przeglądarki")
                # Zapisz nieprawidłowy plik do analizy
                import shutil
                debug_path = Path("debug_audio")
                debug_path.mkdir(exist_ok=True)
                if temp_file_path.exists():
                    shutil.copy(temp_file_path, debug_path / f"problematic_browser_{temp_filename}")
                    logger.info(f"Zapisano problematyczny plik do: {debug_path / f'problematic_browser_{temp_filename}'}")
                
                raise HTTPException(status_code=500, detail=f"Nie można przetworzyć pliku audio: {str(e)}")
        finally:
            # Upewnij się, że plik tymczasowy zostanie usunięty
            try:
                if temp_file_path.exists():
                    temp_file_path.unlink()
            except Exception as e:
                logger.warning(f"Nie można usunąć pliku tymczasowego w bloku finally: {e}")
    
    except HTTPException as he:
        # Re-throw HTTP exceptions
        raise he
    except Exception as e:
        logger.error(f"Error processing recorded audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing recorded audio: {str(e)}"
        ) 