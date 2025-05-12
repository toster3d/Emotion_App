# from app.models.helpers.ensemble_model import WeightedEnsembleModel
# from app.models.resnet_model import AudioResNet
# from app.models.feature_extraction import extract_features, prepare_audio_features
# import logging

# logger = logging.getLogger(__name__)

# # Dodanie bezpiecznych globali dla PyTorch 2.6+
# import torch
# try:
#     # Dla PyTorch 2.6+
#     import numpy as np
#     from torch.serialization import add_safe_globals
    
#     # Lista bezpiecznych typów
#     safe_types = [
#         WeightedEnsembleModel,
#         AudioResNet,
#         torch.torch_version.TorchVersion,
#         np.ndarray,
#         np.dtype,
#         np.core.multiarray._reconstruct
#     ]
    
#     # Rejestracja bezpiecznych typów
#     for safe_type in safe_types:
#         try:
#             add_safe_globals([safe_type])
#             logger.info(f"Registered {safe_type.__name__} as safe global for PyTorch")
#         except Exception as e:
#             logger.warning(f"Could not register {getattr(safe_type, '__name__', str(safe_type))} as safe global: {str(e)}")
# except ImportError as e:
#     # Dla starszych wersji PyTorch
#     logger.info(f"PyTorch serialization module not available or doesn't support add_safe_globals: {str(e)}")
#     pass

# __all__ = [
#     "WeightedEnsembleModel",
#     "AudioResNet",
#     "extract_features",
#     "prepare_audio_features",
# ] 