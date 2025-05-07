from app.models.helpers.ensemble_model import WeightedEnsembleModel
from app.models.resnet_model import AudioResNet
from app.models.feature_extraction import extract_features, prepare_audio_features

# Dodanie klasy WeightedEnsembleModel do bezpiecznych globalnych zmiennych PyTorch 2.6+
import torch
try:
    # Dla PyTorch 2.6+
    from torch.serialization import add_safe_globals
    add_safe_globals([WeightedEnsembleModel])
except ImportError:
    # Dla starszych wersji PyTorch
    pass

__all__ = [
    "WeightedEnsembleModel",
    "AudioResNet",
    "extract_features",
    "prepare_audio_features",
] 