import torch
import logging
from typing import Any
import json
from pathlib import Path
from contextlib import asynccontextmanager

from app.core.config import settings
from app.models.resnet_model import AudioResNet
from app.models.feature_extraction import AudioFeatureExtractor

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(
        self, 
        model_path: str | Path = settings.DEFAULT_MODEL_PATH, 
        feature_type: str = settings.DEFAULT_FEATURE_TYPE, 
        num_classes: int = len(settings.EMOTION_LABELS)
    ):
        """
        Initialize model manager with flexible configuration
        
        Args:
            model_path: Path to the trained model
            feature_type: Type of audio feature
            num_classes: Number of emotion classes
        """
        # Convert to Path if string
        model_path = Path(model_path)
        
        # Validate model path
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = AudioFeatureExtractor(feature_type)
        
        # Model definition and loading
        self.model = AudioResNet(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # Evaluation mode
        
        # Emotion class names
        self.class_names = settings.EMOTION_LABELS
        self.is_loaded = True
    
    def load_models(self):
        """
        Load models (for compatibility with lifespan context)
        
        This method is a no-op since models are loaded in the constructor,
        but it's kept for backward compatibility.
        """
        if not self.is_loaded:
            logger.warning("Model not loaded. Attempting to reload.")
            # Reinitialize the model manager if not loaded
            self.__init__()
        return self
    
    def predict(self, audio_data: Any) -> dict[str, Any]:
        """
        Predict emotion from an audio sample
        
        Args:
            audio_data: Audio sample (path or array)
        
        Returns:
            Prediction results
        """
        # Feature extraction
        feature = self.feature_extractor.extract_feature(audio_data)
        feature = feature.to(self.device)
        
        # Prediction
        with torch.no_grad():
            outputs = self.model(feature)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1)
        
        return {
            'emotion': self.class_names[predicted_class.item()],
            'probabilities': {
                name: prob.item() for name, prob in 
                zip(self.class_names, probabilities[0])
            },
            'confidence': probabilities[0][predicted_class].item()
        }
    
    def batch_predict(self, audio_files: list[Any]) -> list[dict[str, Any]]:
        """
        Predict emotions for multiple audio samples
        
        Args:
            audio_files: List of audio samples
        
        Returns:
            List of prediction results
        """
        return [self.predict(audio) for audio in audio_files]

# Global model manager instance
model_manager = ModelManager()

@asynccontextmanager
async def lifespan_model_loading(app):
    """FastAPI lifespan context manager"""
    try:
        model_manager.load_models()
        yield
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    finally:
        logger.info("Shutting down model service")