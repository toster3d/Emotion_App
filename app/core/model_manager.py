import os
import torch
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from app.core.config import settings
from app.models.resnet_model import AudioResNet
from app.models.ensemble_model import WeightedEnsembleModel

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Manager for handling PyTorch model loading and inference.
    Implements a singleton pattern to ensure models are loaded only once.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.ensemble_model = None
        self._initialized = True
        self.is_loaded = False
        
    def _load_resnet_model(self, model_path, feature_type, num_classes=6):
        """Load a single ResNet model."""
        try:
            model = AudioResNet(num_classes=num_classes)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            model.eval()
            
            # Enable JIT optimization for inference
            with torch.jit.optimized_execution(True):
                # Create example input for tracing
                example_input = torch.randn(1, 1, 128, 128, device=self.device)
                traced_model = torch.jit.trace(model, example_input)
                
            # Freeze the model for further optimization
            traced_model = torch.jit.freeze(traced_model)
            
            return traced_model
        except Exception as e:
            logger.error(f"Error loading model {model_path}: {str(e)}")
            raise RuntimeError(f"Failed to load model {model_path}: {str(e)}")
    
    def load_models(self):
        """Load all models needed for the ensemble."""
        if self.is_loaded:
            return
            
        logger.info(f"Loading models on device: {self.device}")
        models_dir = settings.MODELS_DIR
        
        try:
            # Load individual feature models
            model_files = {
                "melspectrogram": "melspectrogram_model.pt",
                "mfcc": "mfcc_model.pt",
                "chroma": "chroma_model.pt"
            }
            
            models_dict = {}
            for feature_type, model_file in model_files.items():
                model_path = models_dir / model_file
                if not model_path.exists():
                    logger.warning(f"Model file {model_path} not found, skipping...")
                    continue
                
                logger.info(f"Loading {feature_type} model from {model_path}")
                model = self._load_resnet_model(model_path, feature_type)
                models_dict[feature_type] = model
            
            if not models_dict:
                raise RuntimeError("No models could be loaded")
            
            # Load ensemble model if available, or create a new one
            ensemble_path = models_dir / "ensemble_model.pt"
            if ensemble_path.exists():
                logger.info(f"Loading ensemble model from {ensemble_path}")
                self.ensemble_model = WeightedEnsembleModel.load(
                    ensemble_path, 
                    models_dict,
                    device=self.device
                )
            else:
                logger.info("Creating new ensemble model with default weights")
                self.ensemble_model = WeightedEnsembleModel(
                    models_dict,
                    weights=settings.MODEL_WEIGHTS
                )
            
            self.ensemble_model.to(self.device)
            self.ensemble_model.eval()
            
            self.is_loaded = True
            logger.info("All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ensemble model: {str(e)}")
            raise RuntimeError(f"Failed to load models: {str(e)}")
    
    async def predict(self, features):
        """
        Run inference on provided features.
        
        Args:
            features: Dict with feature tensors in format {feature_type: tensor}
            
        Returns:
            Dict with emotion predictions including probabilities and labels
        """
        if not self.is_loaded:
            self.load_models()
        
        # Move tensors to the correct device
        device_features = {k: v.to(self.device) for k, v in features.items()}
        
        # Run inference
        with torch.inference_mode():
            output = self.ensemble_model(device_features)
            
            # Get probabilities and predicted class
            probabilities = output[0].cpu().numpy()
            pred_idx = torch.argmax(output, dim=1)[0].item()
            
            # Map to emotion labels
            emotion_labels = settings.EMOTION_LABELS
            pred_label = emotion_labels[pred_idx] if pred_idx < len(emotion_labels) else "unknown"
            
            # Format the result
            result = {
                "emotion": pred_label,
                "confidence": float(probabilities[pred_idx]),
                "probabilities": {
                    emotion: float(prob) 
                    for emotion, prob in zip(emotion_labels, probabilities)
                }
            }
            
            return result

# Create a singleton model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan_model_loading(app):
    """Context manager for FastAPI to load models on startup."""
    try:
        # Load models on startup
        model_manager.load_models()
        yield
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise
    finally:
        # Clean up resources if needed
        logger.info("Shutting down model service") 