from contextlib import asynccontextmanager
import logging
import numpy as np
import torch
from numpy.typing import NDArray
from fastapi import FastAPI
from pathlib import Path
from typing import AsyncIterator, TypedDict
from app.core.settings import settings
from app.models.audio_resnet import AudioResNet

logger = logging.getLogger(__name__)

class PredictionResult(TypedDict):
    emotion: str
    confidence: float
    probabilities: dict[str, float]

class ModelManager:
    """
    Class responsible for loading and managing ML models for emotion recognition
    """
    
    def __init__(self, model_path: Path | str | None = None) -> None:
        """
        Initialize the model manager with specified model path or default path

        Args:
            model_path: Path to the trained model file (.pt)
        """
        self.model_path = Path(model_path) if model_path else Path(settings.PRETRAINED_MODEL_PATH)
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.USE_CUDA else "cpu")
        self.model: AudioResNet | None = None
        self.is_loaded: bool = False
        
        logger.info(f"Initializing ModelManager with model path: {self.model_path}, device: {self.device}")
        
        # Load the model
        try:
            self.load_model()
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def load_model(self) -> None:
        """
        Load the PyTorch model from the specified path
        """
        if not self.model_path.exists():
            logger.error(f"Model file not found at {self.model_path}")
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Initialize the model
            self.model = AudioResNet(num_classes=len(settings.CLASS_NAMES), dropout_rate=0.5)
            
            # Load state dictionary
            state_dict = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully and moved to {self.device}")
            self.is_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            raise
    
    def predict(self, features: NDArray[np.float32]) -> PredictionResult:
        """
        Make a prediction using the loaded model

        Args:
            features: Audio features in the format expected by the model 
                      (preprocessed melspectrogram)

        Returns:
            Dictionary containing predicted emotion, confidence, and probabilities
        """
        if not self.is_loaded or self.model is None:
            logger.error("Model is not loaded. Cannot make prediction.")
            raise RuntimeError("Model is not loaded. Cannot make prediction.")
        
        try:
            # Ensure features have correct shape (batch, channel, height, width)
            if len(features.shape) == 3:  # (channel, height, width)
                features = np.expand_dims(features, axis=0)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(features).to(self.device)
            
            # Get model prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities: NDArray[np.float32] = torch.nn.functional.softmax(outputs, dim=1).to("cpu").numpy()[0] # type: ignore
            
            # Get predicted class index and confidence
            predicted_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_idx])
            
            # Map index to emotion name
            predicted_emotion = settings.CLASS_NAMES[predicted_idx]
            
            # Create dictionary of all probabilities
            all_probabilities: dict[str, float] = {
                emotion: float(prob) 
                for emotion, prob in zip(settings.CLASS_NAMES, probabilities)
            }
            
            return {
                "emotion": predicted_emotion,
                "confidence": confidence,
                "probabilities": all_probabilities
            }
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise

model_manager = ModelManager()

@asynccontextmanager
async def lifespan_model_loading(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context manager"""
    try:
        model_manager.load_model()
        yield
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    finally:
        logger.info("Shutting down model service")
