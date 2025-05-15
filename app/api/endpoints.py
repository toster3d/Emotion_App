import tempfile
import logging
import librosa
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form

from app.core.settings import settings
from app.core.model_manager import ModelManager
from app.api.schemas import EmotionPrediction, HealthCheck, ErrorResponse
from app.models.feature_extraction import prepare_audio_features

# Initialize model manager
try:
    model_manager = ModelManager()
except Exception as e:
    logging.error(f"Failed to initialize model manager: {e}")
    model_manager = None

router = APIRouter()
logger = logging.getLogger(__name__)

def validate_audio_file(file: UploadFile) -> bool:
    """Validate that the uploaded file is a valid audio file."""
    if file.filename is None:
        return False
        
    file_ext = file.filename.split('.')[-1].lower()
    return file_ext in settings.ALLOWED_AUDIO_EXTENSIONS

@router.get("/health", response_model=HealthCheck, tags=["health"])
async def health_check():
    """Check if the API is running and models are loaded."""
    if model_manager is None:
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
        400: {"model": ErrorResponse, "description": "Invalid input"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    tags=["prediction"]
)
async def predict_emotion(
    file: UploadFile = File(..., description="Audio file to analyze"),
    sample_rate: int | None = Query(
        None, 
        description="Sample rate to use for processing. If not provided, will use default.",
        ge=8000,
        le=48000
    )
):
    """
    Predict emotion from an uploaded audio file.
    
    - **file**: Audio file to analyze (mp3, wav, ogg, flac, m4a, webm)
    - **sample_rate**: Optional sample rate override (8000-48000 Hz)
    """
    if model_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model service is not initialized. Please contact support."
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
                detected_sr
            )
            
            # Get prediction from the model
            logger.info(f"Running prediction on extracted features")
            prediction = model_manager.predict(features['melspectrogram'])
            
            # Log prediction results
            logger.info(f"Prediction result: {prediction['emotion']} with confidence {prediction['confidence']}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            raise HTTPException(status_code=500, detail=f"Cannot process audio file: {str(e)}")
        
        finally:
            # Clean up temporary file
            try:
                Path(temp_file_path).unlink()
            except Exception as cleanup_error:
                logger.warning(f"Could not remove temporary file: {cleanup_error}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error processing audio: {str(e)}"
        )

@router.post(
    "/record", 
    response_model=EmotionPrediction,
    responses={400: {"model": ErrorResponse}},
    tags=["prediction"]
)
async def record_and_predict(
    audio_data: bytes = File(..., description="Raw audio data as bytes"),
    sample_rate: int = Form(settings.DEFAULT_SAMPLE_RATE, description="Sample rate of the recorded audio")
):
    """
    Predict emotion from audio data recorded in the browser.
    
    - **audio_data**: Raw audio data as bytes
    - **sample_rate**: Sample rate of the recorded audio
    """
    if model_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Model service is not initialized. Please contact support."
        )
    
    try:
        logger.info(f"Processing recorded audio: sample_rate={sample_rate}, data size={len(audio_data)} bytes")
        
        if len(audio_data) < 1000:
            logger.warning(f"Received very small audio file ({len(audio_data)} bytes)")
            raise HTTPException(
                status_code=400,
                detail="Audio file is too small, record a longer sample"
            )
        
        # Create temporary folder
        temp_folder = Path("temp_audio")
        temp_folder.mkdir(exist_ok=True)
        
        # Generate unique filename
        import uuid
        temp_filename = f"recording_{uuid.uuid4()}.wav"
        temp_file_path = temp_folder / temp_filename
        
        try:
            # Save audio file
            with open(temp_file_path, "wb") as f:
                f.write(audio_data)
            
            logger.info(f"Recorded audio saved to temporary file: {temp_file_path}")
            
            # Load audio using librosa
            logger.info("Loading recorded audio data")
            audio_array, detected_sr = librosa.load(
                temp_file_path, 
                sr=sample_rate,
                mono=True
            )
            
            logger.info(f"Audio loaded successfully: shape={audio_array.shape}, sr={detected_sr}")
            
            # Check if audio is empty or too short
            if len(audio_array) < 100:
                raise ValueError("Recording is too short or empty")
            
            # Prepare features for the model
            logger.info("Extracting features from recorded audio")
            features = prepare_audio_features(
                audio_array, 
                detected_sr
            )
            
            # Get prediction from the model
            logger.info("Running prediction on extracted features")
            prediction = model_manager.predict(features['melspectrogram'])
            
            # Log prediction results
            logger.info(f"Prediction result: {prediction['emotion']} with confidence {prediction['confidence']}")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error processing recorded audio: {e}")
            raise HTTPException(status_code=500, detail=f"Cannot process audio file: {str(e)}")
        
        finally:
            # Ensure temporary file is removed
            try:
                if temp_file_path.exists():
                    temp_file_path.unlink()
                    logger.info(f"Temporary file removed: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"Could not remove temporary file: {cleanup_error}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing recorded audio: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error processing recorded audio: {str(e)}"
        ) 