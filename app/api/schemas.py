from pydantic import BaseModel, Field

class EmotionProbability(BaseModel):
    """Individual emotion probability."""
    emotion: str
    probability: float = Field(..., ge=0.0, le=1.0)


class EmotionPrediction(BaseModel):
    """Response model for emotion prediction."""
    emotion: str = Field(..., description="Predicted emotion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for the prediction")
    probabilities: dict[str, float] = Field(..., description="Probabilities for all emotions")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "emotion": "happy",
                "confidence": 0.85,
                "probabilities": {
                    "anger": 0.02,
                    "surprised": 0.01,
                    "fear": 0.05,
                    "happiness": 0.85,
                    "neutral": 0.05,
                    "sadness": 0.02
                }
            }
        }
    }


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = "ok"
    models_loaded: bool
    available_models: list[str]
    device: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "ok",
                "models_loaded": True,
                "available_models": ["melspectrogram", "mfcc", "chroma"],
                "device": "cpu"
            }
        }
    }


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Invalid audio file format"
            }
        }
    } 