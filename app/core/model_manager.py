import torch
import logging
import numpy as np
import json
from pathlib import Path
from contextlib import asynccontextmanager

from app.core.config import settings
from app.models.resnet_model import AudioResNet
from app.models.helpers.ensemble_model import WeightedEnsembleModel

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
        self.normalization_stats = {}
        self._initialized = True
        self.is_loaded = False

    def _load_resnet_model(self, model_path, feature_type):
        """Load a single ResNet model from path."""
        try:
            model = AudioResNet(feature_type=feature_type)
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            # Usunięto kompilację aby uniknąć problemów z kompilatorem na Windows
            
            return model
        except Exception as e:
            logger.error(f"Error loading {feature_type} model: {str(e)}")
            raise

    def _load_normalization_stats(self, models_dir):
        """Load normalization statistics from JSON file"""
        stats_path = models_dir / "normalization_stats.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Normalization stats file not found: {stats_path}")
        
        with open(stats_path) as f:
            raw_stats = json.load(f)
        
        # Convert to tensors and move to device
        for feature_type in raw_stats:
            self.normalization_stats[feature_type] = {
                'mean': torch.tensor(raw_stats[feature_type]['mean'], device=self.device),
                'std': torch.tensor(raw_stats[feature_type]['std'], device=self.device)
            }
        logger.info("Successfully loaded normalization statistics")

    def load_models(self):
        """Load all models and normalization statistics"""
        if self.is_loaded:
            return

        logger.info(f"Loading models on device: {self.device}")
        models_dir = settings.MODELS_DIR

        try:
            # Load normalization statistics
            self._load_normalization_stats(models_dir)

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

            # Load ensemble model
            ensemble_path = models_dir / "ensemble_model.pt"
            if ensemble_path.exists():
                logger.info(f"Loading ensemble model from {ensemble_path}")
                try:
                    # Próba załadowania ensemble modelu
                    self.ensemble_model, class_names = WeightedEnsembleModel.load(
                        str(ensemble_path),
                        models_dict
                    )
                    # Jeśli załadowanie się powiodło, przenosimy model na odpowiednie urządzenie
                    self.ensemble_model.to(self.device)
                    logger.info(f"Ensemble model loaded successfully: weights={self.ensemble_model.get_weights()}")
                except Exception as e:
                    logger.error(f"Error loading ensemble model: {str(e)}")
                    logger.info("Creating new ensemble with default weights")
                    self.ensemble_model = WeightedEnsembleModel(
                        models_dict,
                        weights=settings.MODEL_WEIGHTS
                    )
            else:
                logger.info("Creating new ensemble with default weights")
                self.ensemble_model = WeightedEnsembleModel(
                    models_dict,
                    weights=settings.MODEL_WEIGHTS
                )

            # Finalize model setup
            self.ensemble_model.to(self.device)
            self.ensemble_model.eval()
            
            # Usunięto cały blok kompilacji modelu ensemble
            # aby uniknąć błędów związanych z brakiem kompilatora cl na Windows

            # Zachowujemy oryginalny model jako referencję
            self._original_ensemble_model = self.ensemble_model

            self.is_loaded = True
            logger.info("All models loaded successfully")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise RuntimeError(f"Failed to load models: {str(e)}")

    async def predict(self, features):
        """Run inference with feature normalization"""
        if not self.is_loaded:
            self.load_models()

        # Normalize features
        normalized_features = {}
        for feature_type, tensor in features.items():
            if feature_type in self.normalization_stats:
                stats = self.normalization_stats[feature_type]
                norm_tensor = (tensor.to(self.device) - stats['mean']) / (stats['std'] + 1e-8)
                normalized_features[feature_type] = norm_tensor
            else:
                logger.warning(f"No normalization stats for {feature_type}, using raw features")
                normalized_features[feature_type] = tensor.to(self.device)

        # Inference
        with torch.inference_mode():
            # Individual model predictions
            individual_predictions = {}
            for feature_type, model in self.ensemble_model.models.items():
                if feature_type in normalized_features:
                    with torch.no_grad():
                        output = model(normalized_features[feature_type])
                        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
                        pred_idx = torch.argmax(output, dim=1)[0].item()
                        individual_predictions[feature_type] = {
                            "prediction": settings.EMOTION_LABELS[pred_idx],
                            "confidence": float(probs[pred_idx]),
                            "probabilities": probs
                        }

            # Ensemble prediction
            ensemble_output = self.ensemble_model(normalized_features)
            probabilities = torch.softmax(ensemble_output, dim=1)[0].cpu().numpy()

            # Final result
            pred_idx = np.argmax(probabilities)
            return {
                "emotion": settings.EMOTION_LABELS[pred_idx],
                "confidence": float(probabilities[pred_idx]),
                "probabilities": {e: float(p) for e, p in zip(settings.EMOTION_LABELS, probabilities)},
                "individual_predictions": individual_predictions
            }


# Singleton instance
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
