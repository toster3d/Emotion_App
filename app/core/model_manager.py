import os
import torch
import logging
import numpy as np
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
        self._initialized = True
        self.is_loaded = False
        
    def _load_resnet_model(self, model_path, feature_type):
        """Load a single ResNet model from path."""
        try:
            model = AudioResNet(feature_type=feature_type)
            # Używanie weights_only=False przy ładowaniu state_dict dla modeli PyTorch
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()  # Ustawienie trybu ewaluacji
            return model
        except Exception as e:
            logger.error(f"Error loading {feature_type} model: {str(e)}")
            raise
    
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
            
            # Create ensemble model with default weights
            logger.info("Creating new ensemble model with model weights")
            self.ensemble_model = WeightedEnsembleModel(
                models_dict,
                weights=settings.MODEL_WEIGHTS
            )
            
            # Zastosowanie kalibracji z pliku konfiguracyjnego
            if hasattr(settings, 'MODEL_CALIBRATION'):
                logger.info("Applying model calibration from settings")
                for model_type, calib_params in settings.MODEL_CALIBRATION.items():
                    if model_type in models_dict:
                        alpha = calib_params.get("alpha", 1.0)
                        beta = calib_params.get("beta", 0.0)
                        logger.info(f"Calibrating {model_type} model: alpha={alpha}, beta={beta}")
                        self.ensemble_model.set_calibration(model_type, alpha=alpha, beta=beta)
            
            self.ensemble_model.to(self.device)
            self.ensemble_model.eval()  # Zapewnienie trybu ewaluacji
            
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
        
        # Zapisz kształty cech wejściowych do debugowania
        logger.info(f"Kształty cech wejściowych: {[(k, v.shape) for k, v in device_features.items()]}")
        
        # Run inference
        with torch.inference_mode():  # Używanie inference_mode zamiast no_grad dla lepszej wydajności
            logger.info(f"Wykonywanie predykcji na modelach: {list(device_features.keys())}")
            
            # Ocena poszczególnych modeli przed ensemble
            individual_predictions = {}
            for feature_type, model in self.ensemble_model.models.items():
                if feature_type in device_features:
                    with torch.no_grad():
                        model_output = model(device_features[feature_type])
                        model_probs = torch.softmax(model_output, dim=1)[0].cpu().numpy()
                        model_pred_idx = torch.argmax(model_output, dim=1)[0].item()
                        model_pred_label = settings.EMOTION_LABELS[model_pred_idx]
                        individual_predictions[feature_type] = {
                            "prediction": model_pred_label,
                            "confidence": float(model_probs[model_pred_idx]),
                            "probabilities": model_probs
                        }
                        logger.info(f"Model {feature_type}: predykcja={model_pred_label}, pewność={model_probs[model_pred_idx]:.4f}")
            
            # Główna predykcja z modelu ensemble
            output = self.ensemble_model(device_features)
            
            # Dodaj logi
            logger.info(f"Surowy wynik modelu ensemble: {output}")
            
            # Zastosuj softmax do surowego wyniku - zapewnia prawidłowe prawdopodobieństwa sumujące się do 1
            probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
            logger.info(f"Prawdopodobieństwa po softmax: {probabilities}")
            
            # Zastosowanie korekcji bias z konfiguracji
            if hasattr(settings, 'BIAS_CORRECTION') and 'ensemble' in settings.BIAS_CORRECTION:
                bias_correction = settings.BIAS_CORRECTION['ensemble']
                if len(bias_correction) == len(probabilities):
                    original_probs = probabilities.copy()
                    # Zastosowanie korekcji - dodanie bias
                    probabilities = probabilities + np.array(bias_correction)
                    # Upewnienie się, że wartości są dodatnie
                    probabilities = np.maximum(probabilities, 0.0)
                    # Renormalizacja aby suma była 1
                    probabilities = probabilities / np.sum(probabilities)
                    logger.info(f"Zastosowano korekcję bias: {bias_correction}")
                    logger.info(f"Prawdopodobieństwa po korekcji: {probabilities}")
                    
                    # Porównanie predykcji przed i po korekcji
                    old_pred = np.argmax(original_probs)
                    new_pred = np.argmax(probabilities)
                    if old_pred != new_pred:
                        logger.info(f"Korekcja bias zmieniła predykcję z {settings.EMOTION_LABELS[old_pred]} na {settings.EMOTION_LABELS[new_pred]}")
            
            # Sprawdzenie rozkładu prawdopodobieństw - czy nie jest zbyt pewny jednej klasy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = -np.log2(1.0/len(settings.EMOTION_LABELS))
            logger.info(f"Entropia predykcji: {entropy:.4f}/{max_entropy:.4f} " 
                      f"({entropy/max_entropy*100:.1f}% maksymalnej entropii)")
            
            # Wykryj przypadek, gdy model jest zbyt pewny jednej klasy
            if probabilities[0] > 0.65 and probabilities[1] < 0.05:
                logger.warning("Wykryto podejrzany wzorzec: bardzo wysokie prawdopodobieństwo dla 'anger' i niskie dla innych klas")
                
                # Sprawdź zgodność modeli - czy wszystkie modele dają taką samą predykcję
                predictions = [info["prediction"] for info in individual_predictions.values()]
                if len(set(predictions)) < len(predictions):
                    logger.info("Modele nie są zgodne w predykcjach, stosowanie korekcji prawdopodobieństw")
                    
                    # Oblicz średnią z indywidualnych modeli z wyłączeniem modelu, który zawsze daje "anger"
                    avg_probs = np.zeros_like(probabilities)
                    valid_predictions = 0
                    
                    for feature_type, info in individual_predictions.items():
                        model_probs = info["probabilities"]
                        if np.argmax(model_probs) != 0 or model_probs[0] < 0.6:
                            avg_probs += model_probs
                            valid_predictions += 1
                    
                    if valid_predictions > 0:
                        avg_probs /= valid_predictions
                        logger.info(f"Skorygowane prawdopodobieństwa: {avg_probs}")
                        
                        # Użyj skorygowanych prawdopodobieństw jeśli są różne od oryginalnych
                        if np.argmax(avg_probs) != np.argmax(probabilities):
                            probabilities = avg_probs
                            logger.warning(f"Zastosowano korektę prawdopodobieństw. Nowa predykcja: {settings.EMOTION_LABELS[np.argmax(probabilities)]}")
            
            pred_idx = np.argmax(probabilities)
            
            # Map to emotion labels
            emotion_labels = settings.EMOTION_LABELS
            logger.info(f"Dostępne etykiety emocji: {emotion_labels}")
            pred_label = emotion_labels[pred_idx] if pred_idx < len(emotion_labels) else "unknown"
            
            # Format the result
            result = {
                "emotion": pred_label,
                "confidence": float(probabilities[pred_idx]),
                "probabilities": {
                    emotion: float(prob) 
                    for emotion, prob in zip(emotion_labels, probabilities)
                },
                "individual_model_predictions": individual_predictions
            }
            
            logger.info(f"Wynik predykcji: {result['emotion']} ({result['confidence']:.4f})")
            logger.info(f"Wagi modeli: {self.ensemble_model.get_weights()}")
            
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