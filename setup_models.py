#!/usr/bin/env python
"""
Script to create empty model files for testing the application.
Run this script to create dummy model files that can be used to test the application
without having actual trained models.
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
from pathlib import Path

from app.models.resnet_model import AudioResNet
from app.models.helpers.ensemble_model import WeightedEnsembleModel

# Define paths
MODELS_DIR = Path("app/models/saved_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def create_dummy_model(output_path, feature_type="melspectrogram", num_classes=6):
    """Create a dummy ResNet model and save it to disk."""
    print(f"Creating dummy {feature_type} model...")
    
    # Initialize model
    model = AudioResNet(num_classes=num_classes)
    
    # Save model weights
    torch.save(model.state_dict(), output_path)
    print(f"Saved dummy model to {output_path}")
    
    return model

def create_dummy_ensemble(output_path, feature_types=None):
    """Create a dummy ensemble model using the individual models."""
    if feature_types is None:
        feature_types = ["melspectrogram", "mfcc", "chroma"]
    
    print("Creating dummy ensemble model...")
    
    # Create individual models
    models_dict = {}
    for ft in feature_types:
        model_path = MODELS_DIR / f"{ft}_model.pt"
        # Use existing model if it exists, otherwise create a new one
        if not model_path.exists():
            model = create_dummy_model(model_path, feature_type=ft)
            models_dict[ft] = model
        else:
            model = AudioResNet()
            model.load_state_dict(torch.load(model_path))
            models_dict[ft] = model
    
    # Create ensemble model
    weights = {ft: 1.0 / len(feature_types) for ft in feature_types}
    ensemble = WeightedEnsembleModel(models_dict, weights=weights)
    
    # Save ensemble model
    ensemble.save(output_path)
    print(f"Saved dummy ensemble model to {output_path}")
    
    return ensemble

def main():
    """Main function to create all dummy models."""
    print("Setting up dummy models for testing the application...")
    
    # Create individual feature models
    feature_types = ["melspectrogram", "mfcc", "chroma"]
    for ft in feature_types:
        model_path = MODELS_DIR / f"{ft}_model.pt"
        if not model_path.exists():
            create_dummy_model(model_path, feature_type=ft)
        else:
            print(f"Model {model_path} already exists, skipping...")
    
    # Create ensemble model
    ensemble_path = MODELS_DIR / "ensemble_model.pt"
    if not ensemble_path.exists():
        create_dummy_ensemble(ensemble_path, feature_types)
    else:
        print(f"Ensemble model {ensemble_path} already exists, skipping...")
    
    print("Setup complete! You can now run the application.")

if __name__ == "__main__":
    main() 